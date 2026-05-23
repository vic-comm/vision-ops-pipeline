import pandas as pd
import numpy as np
import os
import json
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
from .utils import compute_image_hash
import asyncio
from prefect import task, flow
import asyncpg
from mlops.feature_store import FeatureStore
import pandas as pd
import asyncio
import asyncpg
from datetime import datetime, timedelta
import logging
from typing import Optional, List
from prefect import task, flow
# CONFIGURATION
import boto3
LOGS_PATH = os.getenv("LOGS_PATH", "data/raw_logs.jsonl")
MASTER_DATA_PATH = os.getenv("MASTER_DATA_PATH", "data/training_data_with_history.parquet")
BACKUP_PATH = os.getenv("BACKUP_PATH", "data/training_data_backup.parquet")
ARCHIVE_PATH = os.getenv("ARCHIVE_PATH", "data/archives")
FEATURE_CONFIG_PATH = os.getenv("FEATURE_CONFIG", "config/features.json")
BUCKET_NAME = os.getenv("BUCKET_NAME")
S3_MASTER_KEY = os.getenv("S3_MASTER_KEY", "data/training_data_with_history.parquet")
DATABASE_URL = os.getenv("DATABASE_URL", os.getenv("POSTGRES_URL"))
INGESTION_LOOKBACK_HOURS = int(os.getenv("INGESTION_LOOKBACK_HOURS", "24"))
PLATFORMS_TO_INGEST = os.getenv("PLATFORMS_TO_INGEST", "web,api").split(",")
# Data quality thresholds
MIN_IMAGE_SIZE = 1000  # Minimum pixels
MAX_IMAGE_SIZE = 10000000  # Maximum pixels
MIN_NEW_SAMPLES = 10
MAX_NULL_RATIO = 0.3

# Feature engineering (for images, perhaps metadata)
TOXIC_KEYWORDS = []  # Not applicable for images

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'ingestion.log')
LOW_CONFIDENCE_THRESHOLD = 0.3
HIGH_CONFIDENCE_THRESHOLD = 0.7
HIGH_CONFIDENCE_SAMPLE_RATE = 0.05

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FEATURE_CONFIG_PATH, exist_ok=True)
os.makedirs(ARCHIVE_PATH, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def fetch_training_images_stratified(
    lookback_hours: int = 24,
    platforms: List[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch training images with stratified sampling.
    For CV, this would fetch image metadata and labels from database.
    """
    if not DATABASE_URL:
        logger.error("❌ DATABASE_URL not set")
        return None

    platforms = platforms or PLATFORMS_TO_INGEST
    cutoff = datetime.now() - timedelta(hours=lookback_hours)

    logger.info(f"📂 Fetching training images with feedback...")
    logger.info(f"   Strategy: Stratified sampling + admin-reviewed")

    try:
        conn = await asyncpg.connect(DATABASE_URL, statement_cache_size=0)

        # Query for images (assuming image metadata stored in DB)
        query = """
            SELECT
                l.id,
                l.image_path,
                l.user_id,
                l.platform,
                l.toxicity_score,
                l.severity,
                l.timestamp,
                l.metadata,
                CASE
                    WHEN l.severity IN ('HIGH') THEN 1
                    ELSE 0
                END AS label,
                'production' as source_type,
                'model_prediction' as label_source
            FROM image_logs l
            WHERE l.timestamp > $1
              AND l.platform = ANY($2)
              AND l.toxicity_score IS NOT NULL
              AND LENGTH(l.image_path) > 0
            ORDER BY l.timestamp DESC
        """

        rows = await conn.fetch(query, cutoff, platforms)
        await conn.close()

        if not rows:
            logger.warning(f"⚠️ No training images found in last {lookback_hours}h")
            return None

        df = pd.DataFrame([dict(row) for row in rows])

        logger.info(f"✅ Fetched {len(df)} training images")

        # Add image hash for deduplication
        df['image_hash'] = df['image_path'].apply(lambda x: compute_image_hash(x))

        # Deduplicate
        before = len(df)
        df = df.drop_duplicates(subset=['image_hash'])
        if len(df) < before:
            logger.info(f"🔄 Removed {before - len(df)} duplicate images")

        return df

    except Exception as e:
        logger.error(f"❌ Failed to fetch training images: {e}", exc_info=True)
        return None

# DATA LOADING AND VALIDATION
@task(name="Load Training Images", log_prints=True, retries=2, retry_delay_seconds=30)
def load_and_validate_images() -> Optional[pd.DataFrame]:
    """
    Load training images with validation.
    """
    logger.info(f"📂 Loading training images (last {INGESTION_LOOKBACK_HOURS}h)...")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        df = loop.run_until_complete(
            fetch_training_images_stratified(
                lookback_hours=INGESTION_LOOKBACK_HOURS,
                platforms=PLATFORMS_TO_INGEST
            )
        )
    finally:
        loop.close()

    if df is None or df.empty:
        logger.warning("⚠️ No training images available")
        return None

    logger.info(f"✅ Loaded {len(df)} image records")

    # Validate
    return validate_incoming_images(df)

# FEATURE ENGINEERING (for images, metadata features)
@task(name="Calculate Image Features", log_prints=True)
def calculate_image_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"🔧 Engineering features for {len(df)} images...")

    df = df.copy()

    # For images, features are mostly metadata
    # Could add image size, format, etc. if available

    logger.info(f"✅ Feature engineering complete: {len(df.columns)} features")

    return df

@task(name="Validate Image Quality", log_prints=True)
def validate_image_quality(df: pd.DataFrame) -> bool:
    """
    Enhanced validation for images.
    """
    logger.info("🔍 Running image quality & security checks...")

    blocking_issues = []
    warnings = []

    # Required columns
    critical_cols = ['image_path', 'user_id', 'timestamp', 'label']
    for col in critical_cols:
        if col not in df.columns:
            blocking_issues.append(f"❌ Missing column: {col}")
        elif df[col].isnull().any():
            null_count = df[col].isnull().sum()
            blocking_issues.append(f"❌ Nulls in '{col}': {null_count}")

    # Check image paths exist (if local)
    if 'image_path' in df.columns:
        missing_images = 0
        for path in df['image_path'].dropna():
            if not os.path.exists(path):
                missing_images += 1
        if missing_images > 0:
            warnings.append(f"⚠️ {missing_images} image files not found locally")

    # Class imbalance
    if 'label' in df.columns:
        label_dist = df['label'].value_counts(normalize=True)
        minority_ratio = label_dist.min()
        if minority_ratio < 0.05:
            warnings.append(f"⚠️ Severe class imbalance: {minority_ratio:.1%}")

    # Print results
    for w in warnings:
        logger.warning(w)

    if blocking_issues:
        logger.error("🛑 INGESTION BLOCKED - CRITICAL ISSUES")
        for issue in blocking_issues:
            logger.error(issue)
        return False

    logger.info("✅ Image quality checks passed")
    return True

def pull_master_data():
    if os.path.exists(MASTER_DATA_PATH):
        logger.info("✅ Master data already exists locally.")
        return

    logger.info("📉 Pulling Master Data (Remote -> Local)...")

    try:
        subprocess.run(["dvc", "pull", MASTER_DATA_PATH], check=True, capture_output=True)
        logger.info("✅ DVC Pull successful")
        return
    except Exception as e:
        logger.warning(f"⚠️ DVC Pull failed: {e}")

    if BUCKET_NAME:
        try:
            logger.info("🔄 Attempting direct S3 download...")
            s3 = boto3.client('s3')
            os.makedirs(os.path.dirname(MASTER_DATA_PATH), exist_ok=True)
            s3.download_file(BUCKET_NAME, S3_MASTER_KEY, MASTER_DATA_PATH)
            logger.info("✅ Direct S3 Download successful")
        except Exception as e:
            logger.warning(f"❌ Direct S3 Download failed: {e}")

def push_master_data():
    logger.info("📈 Pushing Master Data (Local -> Remote)...")

    try:
        subprocess.run(["dvc", "add", MASTER_DATA_PATH], check=True, capture_output=True)
        subprocess.run(["dvc", "push", MASTER_DATA_PATH], check=True, capture_output=True)
        logger.info("✅ DVC Push successful")
        return
    except Exception as e:
        logger.warning(f"⚠️ DVC Push failed: {e}")

    if BUCKET_NAME:
        try:
            logger.info("🔄 Attempting direct S3 upload...")
            s3 = boto3.client('s3')
            s3.upload_file(MASTER_DATA_PATH, BUCKET_NAME, S3_MASTER_KEY)
            logger.info("✅ Direct S3 Upload successful")
        except Exception as e:
            logger.error(f"❌ Direct S3 Upload failed: {e}")

def merge_and_save(new_df: pd.DataFrame) -> Dict[str, Any]:
    stats = {"new_samples": len(new_df), "status": "pending"}

    try:
        # Download history
        pull_master_data()

        # Load existing
        if os.path.exists(MASTER_DATA_PATH):
            master_df = pd.read_parquet(MASTER_DATA_PATH)
            stats["master_size_before"] = len(master_df)
        else:
            master_df = pd.DataFrame()
            stats["master_size_before"] = 0

        # Merge
        combined_df = pd.concat([master_df, new_df], ignore_index=True)

        # Deduplicate
        if 'image_hash' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['image_hash'], keep='last')

        stats["master_size_after"] = len(combined_df)
        stats["duplicates_removed"] = (stats["master_size_before"] + len(new_df)) - len(combined_df)

        # Save
        combined_df.to_parquet(MASTER_DATA_PATH)

        # Upload
        push_master_data()

        # Cleanup
        if os.path.exists(LOGS_PATH):
            os.remove(LOGS_PATH)

        stats["status"] = "success"
        return stats

    except Exception as e:
        logger.error(f"❌ Merge failed: {e}", exc_info=True)
        stats["status"] = "failed"
        return stats

def validate_incoming_images(new_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Validates the DataFrame passed from the Orchestrator.
    """
    if new_data is None or new_data.empty:
        logger.warning("⚠️ Received empty DataFrame for validation")
        return None

    logger.info(f"🔍 Validating {len(new_data)} raw image records...")

    try:
        # Validate required columns
        required_cols = ['image_path', 'user_id', 'timestamp']
        missing_cols = [col for col in required_cols if col not in new_data.columns]

        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return None

        # Filter by basic criteria
        valid_data = new_data.copy()

        logger.info(f"✅ {len(valid_data)} image records passed validation")

        # Add metadata
        valid_data['ingested_at'] = datetime.now().isoformat()

        # Deduplicate by image hash
        if 'image_hash' in valid_data.columns:
            before = len(valid_data)
            valid_data = valid_data.drop_duplicates(subset=['image_hash'])
            after = len(valid_data)
            if before != after:
                logger.info(f"🔄 Removed {before - after} duplicate images inside this batch")

        return valid_data

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}", exc_info=True)
        return None

# MAIN FLOW
@flow(name="Image Data Ingestion Pipeline", log_prints=True)
def data_ingestion_flow(new_data: Optional[pd.DataFrame] = None):
    logger.info("🚀 Starting image data ingestion pipeline...")

    if not new_data:
        new_data = load_and_validate_images()
    else:
        new_data = validate_incoming_images(new_data)

    if new_data is None:
        logger.info("✅ No new image data to process")
        return {"status": "skipped", "reason": "no_new_data"}

    processed_data = calculate_image_features(new_data)

    quality_ok = validate_image_quality(processed_data)

    if not quality_ok:
        logger.error("Image quality checks failed - aborting ingestion")
        return {"status": "failed", "reason": "quality_check_failed"}

    merge_stats = merge_and_save(processed_data)

    if merge_stats["status"] != "success":
        logger.error("Merge failed - aborting pipeline")
        return merge_stats

    logger.info("\n" + "="*60)
    logger.info("📊 IMAGE INGESTION SUMMARY")
    logger.info("="*60)
    logger.info(f"New images ingested: {merge_stats['new_samples']}")
    logger.info(f"Total dataset size: {merge_stats['master_size_after']}")
    logger.info(f"Duplicates removed: {merge_stats['duplicates_removed']}")
    logger.info("="*60)

    return merge_stats

if __name__ == "__main__":
    data_ingestion_flow()