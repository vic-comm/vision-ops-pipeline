"""
Casting Defect Detection — Hybrid Drift Detection Pipeline
==========================================================
Production-grade, two-tier drift detection system.

Tier 1 — Heuristic / Hardware Checks  (fast, cheap)
    Extract tabular image statistics (brightness, contrast, file size, aspect
    ratio) and run Evidently DataDriftPreset + KS-test + PSI.
    Purpose: catch camera failures, lighting changes, sensor degradation.
    Limitation: semantically blind — identical stats != identical content.

Tier 2 — Semantic / Embedding Checks  (slower, meaningful)
    Pass images through a batched, headless ResNet50.
    Apply PCA to reduce 2048-D embeddings to top-K principal components that
    retain >= 95% of variance (typically 10-30 PCs).
    Run per-PC KS-test + PSI — the same rigorous tests used in Tier 1.
    Purpose: detect new defect morphologies or manufacturing process changes
    that leave pixel statistics unchanged.
    Fixes the critical flaw of averaging raw high-dimensional embeddings, which
    collapses the distribution shape and produces massive false negatives.

Architecture decision log
    - Raw cosine-similarity-between-means is mathematically unsound for
      multimodal manifolds (OK cluster vs defect cluster).  Discarded.
    - PCA on embeddings followed by per-component KS/PSI preserves
      distributional shape and is statistically valid.
    - ResNet50 runs batched (not per-image) to avoid compute waste.
    - Severity scoring merges both tiers; either tier alone can trigger
      escalation (union of evidence, not intersection).
    - Evidently report covers Tier-1 features for human-readable HTML artefact.
    - All orchestration, MLflow logging, and alerting from the operationally
      mature Tier-1 approach are preserved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import boto3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from PIL import Image, ImageStat
from scipy import stats
from sklearn.decomposition import PCA
from torchvision import models, transforms

import mlflow

try:
    from prefect import flow, get_run_logger, task
except ImportError:
    def task(name=None, log_prints=False):
        def _d(f): return f
        return _d
    def flow(*a, **kw):
        def _d(f): return f
        return _d
    def get_run_logger():
        return logging.getLogger(__name__)

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

REFERENCE_STATS_PATH      = os.getenv("REFERENCE_DATA_PATH",       "data/reference_image_stats.parquet")
REFERENCE_EMBEDDINGS_PATH = os.getenv("REFERENCE_EMBEDDINGS_PATH", "data/reference_embeddings.npz")
REPORT_OUTPUT_PATH        = os.getenv("REPORT_OUTPUT_PATH",        "reports/drift/drift_report.html")
METRICS_OUTPUT_PATH       = os.getenv("METRICS_OUTPUT_PATH",       "reports/drift/drift_metrics.json")
HISTORICAL_METRICS_PATH   = os.getenv("HISTORICAL_METRICS_PATH",   "reports/drift/drift_history.jsonl")
DRIFT_LOOKBACK_HOURS      = int(os.getenv("DRIFT_LOOKBACK_HOURS",  "24"))
BUCKET_NAME               = os.getenv("BUCKET_NAME")
DATABASE_URL              = os.getenv("DATABASE_URL", os.getenv("POSTGRES_URL"))
SLACK_WEBHOOK_URL         = os.getenv("SLACK_WEBHOOK_URL")
PAGERDUTY_TOKEN           = os.getenv("PAGERDUTY_TOKEN")
ALERT_EMAIL               = os.getenv("ALERT_EMAIL")

MIN_SAMPLES_FOR_DRIFT = 50
EMBEDDING_SAMPLE_CAP  = 300      # max images to embed per run (cost control)
PCA_VARIANCE_TARGET   = 0.95     # retain 95% of embedding variance
EMBEDDING_BATCH_SIZE  = 32

# Tier-1 heuristic thresholds
DRIFT_THRESHOLDS: Dict[str, float] = {
    "dataset_drift_share":      0.30,   # >30% of heuristic features drifted
    "label_distribution_shift": 0.15,   # >15% change in defect rate
    "psi_threshold":            0.20,   # PSI >= 0.20 -> significant shift
    "performance_drop":         0.05,
}

# Tier-2 semantic thresholds (per PCA component)
EMBEDDING_DRIFT_THRESHOLDS: Dict[str, float] = {
    "psi_threshold":         0.10,   # tighter than heuristic
    "ks_alpha":              0.05,   # significance level for KS test
    "component_drift_share": 0.30,   # >30% of PCs drifted -> semantic alert
}

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "drift_detection.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# TIER 1 — IMAGE HEURISTIC FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_image_features(image_path: str) -> Dict[str, float]:
    """
    Extract cheap tabular statistics from a single image.

    These are proxies for hardware / environment health:
      brightness_*  -> lighting / exposure
      contrast_*    -> lens focus, depth of field
      aspect_ratio  -> camera orientation / crop settings
      file_size_kb  -> image entropy (blurry images compress more)
      is_grayscale  -> colour-channel failure detection

    NOT a semantic check — two completely different images can share
    identical values here, which is exactly why Tier 2 exists.
    """
    try:
        img  = Image.open(image_path).convert("RGB")
        stat = ImageStat.Stat(img)
        w, h = img.size
        size_kb    = os.path.getsize(image_path) / 1024
        r_m, g_m, b_m = stat.mean
        chroma_var = float(np.std([r_m, g_m, b_m]))

        return {
            "brightness_r":    float(stat.mean[0]),
            "brightness_g":    float(stat.mean[1]),
            "brightness_b":    float(stat.mean[2]),
            "brightness_mean": float(np.mean(stat.mean)),
            "contrast_r":      float(stat.stddev[0]),
            "contrast_g":      float(stat.stddev[1]),
            "contrast_b":      float(stat.stddev[2]),
            "contrast_mean":   float(np.mean(stat.stddev)),
            "aspect_ratio":    float(w / max(h, 1)),
            "width":           float(w),
            "height":          float(h),
            "file_size_kb":    float(size_kb),
            "is_grayscale":    float(chroma_var < 5.0),
        }
    except Exception as exc:
        logger.warning("Could not extract heuristic features from %s: %s", image_path, exc)
        return {}


def build_reference_stats(image_dir: str, save_path: str) -> pd.DataFrame:
    """
    One-time bootstrap: scan training images -> extract Tier-1 features ->
    save as the reference parquet used by every future drift run.
    Run this once after finalising the training dataset.
    """
    IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".bmp"}
    records: List[Dict] = []

    for class_name, label in [("ok_front", 0), ("def_front", 1)]:
        class_dir = Path(image_dir) / class_name
        if not class_dir.exists():
            continue
        for p in class_dir.iterdir():
            if p.suffix.lower() in IMAGE_EXTS:
                feats = extract_image_features(str(p))
                if feats:
                    feats["label"]      = label
                    feats["class_name"] = class_name
                    feats["image_path"] = str(p)
                    records.append(feats)

    df = pd.DataFrame(records)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, index=False)
    logger.info("Reference stats saved -> %s (%d images)", save_path, len(df))
    return df


# ─────────────────────────────────────────────────────────────
# TIER 2 — SEMANTIC EMBEDDING EXTRACTOR  (batched ResNet50)
# ─────────────────────────────────────────────────────────────

class EmbeddingExtractor:
    """
    Batched, inference-only ResNet50 feature extractor.

    Design decisions:
      - The final FC layer is removed -> 2048-D global average pool output.
      - Images processed in batches (not one-by-one) to amortise GPU/CPU
        transfer overhead. A per-image loop is 10-30x slower.
      - Model instantiated once per drift run and reused across both splits.
      - L2-normalisation applied after extraction to unit-normalise vectors
        (standard practice before PCA on embeddings).
    """

    _TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, batch_size: int = EMBEDDING_BATCH_SIZE):
        self.batch_size = batch_size
        self.device     = self._pick_device()
        self.model      = self._build_model()

    @staticmethod
    def _pick_device() -> torch.device:
        if torch.cuda.is_available():   return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    def _build_model(self) -> nn.Module:
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the classifier head; output is (B, 2048) after avgpool + flatten
        model = nn.Sequential(*list(backbone.children())[:-1])
        model = model.to(self.device)
        model.eval()
        return model

    def compute_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """
        Return float32 array of shape (N_valid, 2048).
        Invalid images are silently skipped; caller validates length.
        """
        valid_tensors: List[torch.Tensor] = []

        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                valid_tensors.append(self._TRANSFORM(img))
            except Exception as exc:
                logger.debug("Skipping %s: %s", path, exc)

        if not valid_tensors:
            return np.empty((0, 2048), dtype=np.float32)

        all_embeddings: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(valid_tensors), self.batch_size):
                batch = torch.stack(valid_tensors[start : start + self.batch_size])
                batch = batch.to(self.device)
                feats = self.model(batch)               # (B, 2048, 1, 1)
                feats = feats.squeeze(-1).squeeze(-1)   # (B, 2048)
                feats = feats.cpu().numpy()
                # L2-normalise each vector
                norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-10
                all_embeddings.append(feats / norms)

        return np.vstack(all_embeddings).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# PCA-BASED SEMANTIC DRIFT  (the key mathematical fix)
# ─────────────────────────────────────────────────────────────

class PCAEmbeddingDriftDetector:
    """
    Mathematically sound distribution drift detection on high-dimensional embeddings.

    Why raw mean cosine-similarity fails
    ─────────────────────────────────────
    ResNet embeddings form a multi-modal manifold. "OK" images cluster in one
    region of the 2048-D sphere; defective images cluster in another. Averaging
    collapses both clusters into a single centroid. Two datasets with entirely
    different class balances can yield the same mean vector -> false negative.
    This was the critical flaw in the original Script 2.

    The correct approach
    ─────────────────────
    1. Fit PCA on the reference embeddings -> learn principal axes of variance.
    2. Project both reference and current embeddings onto the top-K PCs that
       collectively explain >= 95% of variance (typically 10-30 dimensions).
    3. Each PC is now a 1-D distribution. Run KS-test + PSI on every PC
       independently — the same validated approach used for tabular features.
    4. Aggregate: if >30% of PCs are individually drifted -> semantic drift alert.

    This correctly handles the multimodal manifold structure and is
    statistically equivalent to the Tier-1 approach — making the combined
    decision logic symmetric and interpretable.
    """

    def __init__(self, variance_threshold: float = PCA_VARIANCE_TARGET):
        self.variance_threshold = variance_threshold
        self.pca: Optional[PCA] = None
        self.n_components_: int = 0

    def fit(self, ref_embeddings: np.ndarray) -> "PCAEmbeddingDriftDetector":
        """
        Fit PCA on the reference embedding set.
        Automatically determines K = min components explaining >= variance_threshold.
        """
        max_k    = min(ref_embeddings.shape[0] - 1, ref_embeddings.shape[1])
        pca_full = PCA(n_components=max_k, random_state=42)
        pca_full.fit(ref_embeddings)

        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        k      = int(np.searchsorted(cumvar, self.variance_threshold)) + 1
        k      = max(k, 5)   # always keep at least 5 components

        self.pca           = PCA(n_components=k, random_state=42)
        self.pca.fit(ref_embeddings)
        self.n_components_ = k

        logger.info(
            "PCA fitted: %d components explain %.1f%% of embedding variance",
            k, float(cumvar[k - 1]) * 100,
        )
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("Call fit() before transform()")
        return self.pca.transform(embeddings)

    def detect_drift(
        self,
        ref_embeddings: np.ndarray,
        cur_embeddings: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Project both sets onto the PCA basis and run per-component tests.
        Returns a structured result dict compatible with the Tier-1 output schema.
        """
        if self.pca is None:
            self.fit(ref_embeddings)

        ref_proj = self.transform(ref_embeddings)   # (N_ref, K)
        cur_proj = self.transform(cur_embeddings)   # (N_cur, K)

        component_results: Dict[str, Dict] = {}
        drifted_components: List[str]       = []

        for i in range(self.n_components_):
            ref_pc = ref_proj[:, i]
            cur_pc = cur_proj[:, i]

            ks_stat, ks_p = stats.ks_2samp(ref_pc, cur_pc)
            ks_drifted    = ks_p < EMBEDDING_DRIFT_THRESHOLDS["ks_alpha"]

            psi         = self._psi(ref_pc, cur_pc)
            psi_drifted = psi is not None and psi > EMBEDDING_DRIFT_THRESHOLDS["psi_threshold"]

            pc_drifted = ks_drifted or psi_drifted
            pc_name    = f"pc_{i:02d}"

            component_results[pc_name] = {
                "ks_statistic": float(ks_stat),
                "ks_p_value":   float(ks_p),
                "psi":          psi,
                "ref_mean":     float(ref_pc.mean()),
                "cur_mean":     float(cur_pc.mean()),
                "drifted":      pc_drifted,
            }
            if pc_drifted:
                drifted_components.append(pc_name)

        drift_share = len(drifted_components) / max(self.n_components_, 1)
        overall     = drift_share > EMBEDDING_DRIFT_THRESHOLDS["component_drift_share"]

        logger.info(
            "Semantic drift — %d/%d PCs drifted (%.1f%%) | overall_drift=%s",
            len(drifted_components), self.n_components_,
            drift_share * 100, overall,
        )

        return {
            "n_components":           self.n_components_,
            "drifted_components":     drifted_components,
            "component_drift_share":  float(drift_share),
            "overall_drift_detected": overall,
            "components":             component_results,
            "ref_samples":            int(ref_embeddings.shape[0]),
            "cur_samples":            int(cur_embeddings.shape[0]),
            "timestamp":              datetime.now().isoformat(),
        }

    @staticmethod
    def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> Optional[float]:
        try:
            _, edges = np.histogram(ref, bins=bins)
            eps      = 1e-10
            ref_d    = np.histogram(ref, bins=edges)[0] / len(ref) + eps
            cur_d    = np.histogram(cur, bins=edges)[0] / len(cur) + eps
            return float(np.sum((cur_d - ref_d) * np.log(cur_d / ref_d)))
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────
# SHARED STATISTICAL HELPERS
# ─────────────────────────────────────────────────────────────

def ks_test(ref: pd.Series, cur: pd.Series) -> Dict[str, Any]:
    try:
        s, p = stats.ks_2samp(ref.dropna().values, cur.dropna().values)
        return {"statistic": float(s), "p_value": float(p), "drifted": bool(p < 0.05)}
    except Exception as exc:
        logger.warning("KS test failed: %s", exc)
        return {"statistic": None, "p_value": None, "drifted": None}


def population_stability_index(ref: pd.Series, cur: pd.Series, bins: int = 10) -> Optional[float]:
    try:
        _, edges = np.histogram(ref.dropna(), bins=bins)
        eps      = 1e-10
        ref_d    = np.histogram(ref.dropna(), bins=edges)[0] / len(ref) + eps
        cur_d    = np.histogram(cur.dropna(), bins=edges)[0] / len(cur) + eps
        return float(np.sum((cur_d - ref_d) * np.log(cur_d / ref_d)))
    except Exception as exc:
        logger.warning("PSI failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def _s3_download(s3_key: str, local_path: str) -> bool:
    if not BUCKET_NAME:
        return False
    try:
        boto3.client("s3").download_file(BUCKET_NAME, s3_key, local_path)
        logger.info("Downloaded s3://%s/%s", BUCKET_NAME, s3_key)
        return True
    except Exception as exc:
        logger.warning("S3 download failed (%s): %s", s3_key, exc)
        return False


def load_reference_stats() -> Optional[pd.DataFrame]:
    """Load reference image stats, falling back to S3 then DVC."""
    if os.path.exists(REFERENCE_STATS_PATH):
        df = pd.read_parquet(REFERENCE_STATS_PATH)
        logger.info("Loaded reference stats: %d rows", len(df))
        return df

    if _s3_download("data/reference_image_stats.parquet", REFERENCE_STATS_PATH):
        return pd.read_parquet(REFERENCE_STATS_PATH)

    if Path(".dvc").exists():
        try:
            subprocess.run(
                ["dvc", "pull", REFERENCE_STATS_PATH],
                check=True, capture_output=True,
            )
            return pd.read_parquet(REFERENCE_STATS_PATH)
        except Exception:
            pass

    logger.error("No reference stats found. Run build_reference_stats() first.")
    return None


async def _fetch_production_logs(lookback_hours: int) -> Optional[pd.DataFrame]:
    """
    Pull recent inference logs from production Postgres / Supabase.

    Expected inference_logs schema:
        image_id        TEXT
        image_path      TEXT        (local path or S3 key)
        prediction      INT         (0=ok, 1=defective)
        score           FLOAT
        brightness_mean FLOAT
        contrast_mean   FLOAT
        aspect_ratio    FLOAT
        file_size_kb    FLOAT
        is_grayscale    INT
        timestamp       TIMESTAMPTZ
        human_label     INT NULL    (delayed ground truth from review team)
    """
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set — skipping DB fetch")
        return None

    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        rows = await conn.fetch(
            """
            SELECT image_id, image_path, prediction, score,
                   brightness_mean, contrast_mean, aspect_ratio,
                   file_size_kb, is_grayscale, timestamp, human_label
            FROM   inference_logs
            WHERE  timestamp > $1
            ORDER  BY timestamp DESC
            """,
            cutoff,
        )
        await conn.close()

        if not rows:
            return None

        df          = pd.DataFrame([dict(r) for r in rows])
        df["label"] = df["human_label"].fillna(df["prediction"])
        logger.info("Fetched %d inference logs from DB", len(df))
        return df

    except Exception as exc:
        logger.error("DB fetch failed: %s", exc, exc_info=True)
        return None


def load_current_data() -> Optional[pd.DataFrame]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        df = loop.run_until_complete(_fetch_production_logs(DRIFT_LOOKBACK_HOURS))
    finally:
        loop.close()

    if df is None or df.empty:
        logger.warning("No current production data available")
        return None
    if len(df) < MIN_SAMPLES_FOR_DRIFT:
        logger.warning(
            "Low sample count: %d < %d — drift estimates may be noisy",
            len(df), MIN_SAMPLES_FOR_DRIFT,
        )
    return df


# ─────────────────────────────────────────────────────────────
# PREFECT TASKS
# ─────────────────────────────────────────────────────────────

@task(name="Tier1: Heuristic Feature Drift", log_prints=True)
def detect_heuristic_drift(
    reference: pd.DataFrame,
    current:   pd.DataFrame,
) -> Dict[str, Any]:
    """
    Tier-1 fast check: KS-test + PSI on tabular image statistics.

    Catches hardware/environment failures (dirty lens, broken lighting,
    image codec changes) cheaply before committing GPU time for Tier 2.

    Key features monitored:
      - brightness channels: detect lighting rig failures
      - contrast channels:   detect focal length / depth-of-field changes
      - aspect_ratio:        detect crop/resize pipeline changes
      - file_size_kb:        detect compression setting changes
      - label_distribution:  detect manufacturing line behavioural shift
    """
    logger.info("── Tier 1: Heuristic drift ──────────────────────────────")

    NUMERICAL_FEATURES = [
        "brightness_mean", "brightness_r", "brightness_g", "brightness_b",
        "contrast_mean",   "contrast_r",   "contrast_g",   "contrast_b",
        "aspect_ratio",    "file_size_kb",
    ]

    results: Dict[str, Any] = {
        "timestamp":         datetime.now().isoformat(),
        "reference_samples": len(reference),
        "current_samples":   len(current),
        "features":          {},
        "drifted_features":  [],
    }

    for feat in NUMERICAL_FEATURES:
        if feat not in reference.columns or feat not in current.columns:
            continue

        ks         = ks_test(reference[feat], current[feat])
        psi        = population_stability_index(reference[feat], current[feat])
        ref_mean   = float(reference[feat].mean())
        cur_mean   = float(current[feat].mean())
        mean_shift = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-10)
        drifted    = (ks.get("drifted") is True) or (
            psi is not None and psi > DRIFT_THRESHOLDS["psi_threshold"]
        )

        results["features"][feat] = {
            "ks_test": ks, "psi": psi,
            "ref_mean": ref_mean, "cur_mean": cur_mean,
            "mean_shift": float(mean_shift), "drifted": drifted,
        }
        if drifted:
            results["drifted_features"].append(feat)
            logger.warning(
                "  Heuristic drift %-22s | PSI=%.3f | KS p=%.4f",
                feat, psi or -1, ks.get("p_value") or -1,
            )

    # Defect-rate (label) shift check
    if "label" in reference.columns and "label" in current.columns:
        ref_rate      = float(reference["label"].mean())
        cur_rate      = float(current["label"].mean())
        shift         = abs(cur_rate - ref_rate)
        label_drifted = shift > DRIFT_THRESHOLDS["label_distribution_shift"]
        results["label_distribution"] = {
            "ref_defect_rate": ref_rate,
            "cur_defect_rate": cur_rate,
            "shift":           shift,
            "drifted":         label_drifted,
        }
        if label_drifted:
            results["drifted_features"].append("label_distribution")
            logger.warning(
                "  Defect-rate shift: %.1f%% -> %.1f%%",
                ref_rate * 100, cur_rate * 100,
            )

    n_drifted   = len([f for f in results["features"] if results["features"][f]["drifted"]])
    n_total     = max(len(results["features"]), 1)
    drift_share = n_drifted / n_total

    results["drift_share"]            = drift_share
    results["overall_drift_detected"] = drift_share > DRIFT_THRESHOLDS["dataset_drift_share"]
    logger.info("  Tier-1 drift share: %.1f%%", drift_share * 100)
    return results


@task(name="Tier1: Evidently Report", log_prints=True)
def generate_evidently_report(
    reference: pd.DataFrame,
    current:   pd.DataFrame,
) -> Dict[str, Any]:
    """
    Run Evidently DataDriftPreset on Tier-1 tabular features.
    Produces an HTML report committed to MLflow artifacts for human review.
    Evidently provides a second, independent statistical opinion on Tier-1 features.
    """
    logger.info("── Evidently HTML report ────────────────────────────────")
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset

        FEATURE_COLS = [
            "brightness_mean", "contrast_mean", "aspect_ratio",
            "file_size_kb", "brightness_r", "brightness_g", "brightness_b",
            "is_grayscale",
        ]
        use_cols = [c for c in FEATURE_COLS
                    if c in reference.columns and c in current.columns]

        schema   = DataDefinition()
        ref_ds   = Dataset.from_pandas(reference[use_cols], data_definition=schema)
        cur_ds   = Dataset.from_pandas(current[use_cols],   data_definition=schema)
        report   = Report(metrics=[
            DataSummaryPreset(include_tests=True),
            DataDriftPreset(drift_share=DRIFT_THRESHOLDS["dataset_drift_share"]),
        ], include_tests=True)
        snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

        Path(REPORT_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(REPORT_OUTPUT_PATH)
        logger.info("Evidently HTML -> %s", REPORT_OUTPUT_PATH)

        report_dict      = snapshot.dict()
        drifted_features: List[str] = []
        drift_share      = 0.0
        dataset_drift    = False

        for entry in report_dict.get("metrics", []):
            result = entry.get("result", {})
            if "drift_by_columns" in result:
                for col, col_stat in result["drift_by_columns"].items():
                    if col_stat.get("drift_detected"):
                        drifted_features.append(col)
                drift_share   = result.get("share_of_drifted_columns", 0.0)
                dataset_drift = result.get("dataset_drift", False)
                break

        return {
            "drift_share":            float(drift_share),
            "drifted_features":       drifted_features,
            "drifted_count":          len(drifted_features),
            "dataset_drift_detected": bool(
                dataset_drift or drift_share >= DRIFT_THRESHOLDS["dataset_drift_share"]
            ),
            "report_path": REPORT_OUTPUT_PATH,
            "timestamp":   datetime.now().isoformat(),
        }

    except Exception as exc:
        logger.error("Evidently report failed: %s", exc, exc_info=True)
        return {
            "drift_share": 0.0, "drifted_features": [], "drifted_count": 0,
            "dataset_drift_detected": False,
            "report_path": None, "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        }


@task(name="Tier2: Semantic Embedding Drift (PCA-KS)", log_prints=True)
def detect_semantic_drift(
    reference: pd.DataFrame,
    current:   pd.DataFrame,
) -> Dict[str, Any]:
    """
    Tier-2 semantic check using PCA-projected ResNet50 embeddings.

    Why this is needed
    ───────────────────
    Tier-1 heuristics can be identical even when the factory starts producing
    a completely new category of defect. Tier-2 catches semantic distribution
    shift invisible to pixel statistics.

    Algorithm
    ─────────
    1. Sample up to EMBEDDING_SAMPLE_CAP images from each split
       (cost-controlled: avoid unbounded GPU time in scheduled jobs).
    2. Extract 2048-D ResNet50 embeddings (batched for efficiency).
    3. Fit PCA on reference embeddings -> K components explaining >= 95% var.
    4. Project both sets onto K principal components.
    5. Run per-component KS + PSI tests (same as Tier-1, statistically valid).
    6. Flag semantic drift if >30% of PCs are individually drifted.

    This correctly handles the multimodal manifold (OK cluster vs defect cluster)
    where mean-based cosine similarity completely fails.
    """
    logger.info("── Tier 2: Semantic (PCA-embedding) drift ───────────────")

    if "image_path" not in reference.columns or "image_path" not in current.columns:
        logger.warning("image_path column missing — skipping Tier-2")
        return _tier2_skipped("no_image_path_column")

    ref_paths = [p for p in reference["image_path"].dropna() if os.path.exists(str(p))]
    cur_paths = [p for p in current["image_path"].dropna()   if os.path.exists(str(p))]

    if len(ref_paths) < 30 or len(cur_paths) < 30:
        logger.warning(
            "Insufficient accessible images (ref=%d, cur=%d) — skipping Tier-2",
            len(ref_paths), len(cur_paths),
        )
        return _tier2_skipped("insufficient_images")

    # Cap sample size for cost control; deterministic seed for reproducibility
    rng        = np.random.default_rng(42)
    ref_sample = list(rng.choice(ref_paths, min(EMBEDDING_SAMPLE_CAP, len(ref_paths)), replace=False))
    cur_sample = list(rng.choice(cur_paths, min(EMBEDDING_SAMPLE_CAP, len(cur_paths)), replace=False))

    logger.info(
        "Extracting embeddings: %d reference, %d current images ...",
        len(ref_sample), len(cur_sample),
    )

    extractor      = EmbeddingExtractor()
    ref_embeddings = extractor.compute_embeddings(ref_sample)
    cur_embeddings = extractor.compute_embeddings(cur_sample)

    if ref_embeddings.shape[0] < 20 or cur_embeddings.shape[0] < 20:
        logger.warning("Too few valid embeddings computed — skipping Tier-2")
        return _tier2_skipped("embedding_computation_failed")

    detector = PCAEmbeddingDriftDetector()
    result   = detector.detect_drift(ref_embeddings, cur_embeddings)
    return result


def _tier2_skipped(reason: str) -> Dict[str, Any]:
    return {
        "status": "unavailable", "reason": reason,
        "overall_drift_detected": False,
        "component_drift_share":  0.0,
        "drifted_components":     [],
        "n_components":           0,
        "timestamp":              datetime.now().isoformat(),
    }


@task(name="Combined Drift Decision", log_prints=True)
def make_drift_decision(
    tier1_heuristic: Dict[str, Any],
    tier1_evidently: Dict[str, Any],
    tier2_semantic:  Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge all tier signals into a single actionable severity decision.

    Severity escalation logic
    ─────────────────────────
    - Base severity derived from Tier-1 combined drift share.
    - Tier-2 semantic drift independently escalates to at least HIGH.
      Rationale: a new defect morphology with no heuristic signal is the
      highest-risk blind spot in any industrial CV system.
    - Label-distribution drift independently escalates to at least HIGH.
    - Heuristic feature-count pressure adds secondary escalation.

    Union-of-evidence model: either tier alone can trigger escalation.
    Intersection-of-evidence would mask single-tier blind spots.
    """
    logger.info("── Combined Drift Decision ──────────────────────────────")

    SEVERITY_LEVELS  = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    SEVERITY_ACTIONS = {
        "none":     "no_action_needed",
        "low":      "continue_monitoring",
        "medium":   "monitor_closely",
        "high":     "retrain_recommended",
        "critical": "immediate_retrain",
    }

    t1_stat_share = tier1_heuristic.get("drift_share", 0.0)
    t1_evid_share = tier1_evidently.get("drift_share", 0.0)
    t2_sem_share  = tier2_semantic.get("component_drift_share", 0.0)

    t1_stat_feats = set(tier1_heuristic.get("drifted_features", []))
    t1_evid_feats = set(tier1_evidently.get("drifted_features", []))
    t2_sem_pcs    = set(tier2_semantic.get("drifted_components", []))

    all_t1_features = t1_stat_feats | t1_evid_feats
    combined_t1     = max(t1_stat_share, t1_evid_share)

    # Base severity from Tier-1
    if   combined_t1 >= 0.50: severity = "critical"
    elif combined_t1 >= 0.30: severity = "high"
    elif combined_t1 >= 0.15: severity = "medium"
    elif combined_t1 >= 0.05: severity = "low"
    else:                      severity = "none"

    score = SEVERITY_LEVELS[severity]

    # Tier-2 independent escalation
    if tier2_semantic.get("overall_drift_detected"):
        score = max(score, SEVERITY_LEVELS["high"])
        logger.warning(
            "  Tier-2 semantic drift (%.0f%% PCs) -> escalating to at least HIGH",
            t2_sem_share * 100,
        )

    # Label distribution shift escalation
    label_drift = tier1_heuristic.get("label_distribution", {}).get("drifted", False)
    if label_drift:
        score = max(score, SEVERITY_LEVELS["high"])

    # Feature count pressure
    n_heuristic = len(all_t1_features)
    if   n_heuristic >= 8: score = max(score, SEVERITY_LEVELS["high"])
    elif n_heuristic >= 4: score = max(score, SEVERITY_LEVELS["medium"])

    severity = next(k for k, v in SEVERITY_LEVELS.items() if v == score)
    action   = SEVERITY_ACTIONS[severity]

    drift_detected = (
        combined_t1 >= DRIFT_THRESHOLDS["dataset_drift_share"]
        or tier2_semantic.get("overall_drift_detected", False)
    )

    decision = {
        "timestamp":     datetime.now().isoformat(),
        "drift_detected": drift_detected,
        "severity":      severity,
        "action":        action,
        "metrics": {
            "tier1_statistical_drift_share":  t1_stat_share,
            "tier1_evidently_drift_share":    t1_evid_share,
            "tier2_semantic_drift_share":     t2_sem_share,
            "combined_tier1_drift_share":     combined_t1,
            "tier1_heuristic_features_count": n_heuristic,
            "tier2_drifted_pc_count":         len(t2_sem_pcs),
            "tier2_total_pcs":                tier2_semantic.get("n_components", 0),
            "label_drift_detected":           label_drift,
        },
        "drifted_features": {
            "tier1_statistical":  list(t1_stat_feats),
            "tier1_evidently":    list(t1_evid_feats),
            "tier1_union":        list(all_t1_features),
            "tier2_semantic_pcs": list(t2_sem_pcs),
        },
        "tier2_available": tier2_semantic.get("status") != "unavailable",
    }

    logger.info(
        "Decision -> severity=%-8s | action=%s | T1=%.0f%% | T2=%.0f%% | label_drift=%s",
        severity.upper(), action,
        combined_t1 * 100, t2_sem_share * 100, label_drift,
    )
    return decision


# ─────────────────────────────────────────────────────────────
# ALERTING
# ─────────────────────────────────────────────────────────────

def _build_slack_payload(decision: Dict[str, Any]) -> Dict:
    m        = decision["metrics"]
    severity = decision["severity"].upper()
    emoji    = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(severity, "⚪")
    return {
        "text": f"{emoji} *Casting CV Drift [{severity}]* — {decision['action']}",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text",
                         "text": f"{emoji} Casting Defect Drift Alert [{severity}]"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Severity:* {severity}"},
                    {"type": "mrkdwn", "text": f"*Action:* {decision['action']}"},
                    {"type": "mrkdwn", "text": f"*Tier-1 Drift:* {m['combined_tier1_drift_share']:.0%}"},
                    {"type": "mrkdwn", "text": f"*Tier-2 Semantic:* {m['tier2_semantic_drift_share']:.0%}"},
                    {"type": "mrkdwn", "text": f"*Label Drift:* {'YES' if m['label_drift_detected'] else 'No'}"},
                    {"type": "mrkdwn", "text": f"*Heuristic Features Drifted:* {m['tier1_heuristic_features_count']}"},
                ],
            },
        ],
    }


def send_slack_alert(decision: Dict[str, Any]) -> None:
    if not SLACK_WEBHOOK_URL:
        logger.info("SLACK_WEBHOOK_URL not configured — skipping Slack alert")
        return
    import requests
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json=_build_slack_payload(decision), timeout=10)
        r.raise_for_status()
        logger.info("Slack alert sent")
    except Exception as exc:
        logger.error("Slack alert failed: %s", exc)


@task(name="Send Alerts", log_prints=True)
def send_alerts(decision: Dict[str, Any]) -> None:
    if decision.get("severity") not in ("high", "critical"):
        logger.info("Severity=%s — no alerts dispatched", decision.get("severity"))
        return
    logger.info("Dispatching alerts (severity=%s) ...", decision["severity"].upper())
    send_slack_alert(decision)
    if ALERT_EMAIL:
        logger.info("Email alert placeholder -> %s", ALERT_EMAIL)
    if PAGERDUTY_TOKEN:
        logger.info("PagerDuty alert placeholder")


@task(name="Save Drift Metrics", log_prints=True)
def save_drift_metrics(payload: Dict[str, Any]) -> None:
    try:
        Path(METRICS_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_OUTPUT_PATH, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        Path(HISTORICAL_METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORICAL_METRICS_PATH, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")

        logger.info("Drift metrics persisted -> %s", METRICS_OUTPUT_PATH)
    except Exception as exc:
        logger.error("Failed to save metrics: %s", exc)


# ─────────────────────────────────────────────────────────────
# MAIN FLOW
# ─────────────────────────────────────────────────────────────

@flow(name="Casting Defect CV Drift Detection v2")
def drift_detection_flow(
    current_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Full two-tier drift detection flow.

    Tier 1a  Heuristic KS/PSI on image statistics   <- fast hardware check
    Tier 1b  Evidently DataDrift HTML report         <- human-readable artefact
    Tier 2   PCA-projected embedding KS/PSI          <- semantic / defect-type check
    Decision Union-of-evidence severity scoring
    Persist  JSON metrics + append to historical log
    Alert    Slack / PagerDuty on HIGH or CRITICAL
    """
    start_time = datetime.now()
    log = get_run_logger()
    log.info("Casting Defect Drift Detection v2 — %s", start_time.isoformat())

    mlflow_run = None
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
        mlflow.set_experiment("Drift_Monitoring_CV_v2")
        mlflow_run = mlflow.start_run(
            run_name=f"drift_{start_time.strftime('%Y%m%d_%H%M%S')}"
        )

        # STEP 1: Load data
        reference = load_reference_stats()
        current   = current_data if current_data is not None else load_current_data()

        if reference is None:
            mlflow.end_run(status="FAILED")
            return {"status": "failed", "reason": "no_reference_data"}

        if current is None or current.empty:
            mlflow.end_run(status="FINISHED")
            return {"status": "skipped", "reason": "no_current_data"}

        common_cols = list(set(reference.columns) & set(current.columns))
        ref_aligned = reference[common_cols]
        cur_aligned = current[common_cols]

        mlflow.log_params({
            "reference_samples": len(reference),
            "current_samples":   len(current),
            "pipeline_version":  "2.0.0",
        })

        # STEP 2: Tier-1 heuristic checks
        t1_stat = detect_heuristic_drift(ref_aligned, cur_aligned)
        t1_evid = generate_evidently_report(ref_aligned, cur_aligned)

        mlflow.log_metrics({
            "t1_statistical_drift_share": t1_stat.get("drift_share", 0.0),
            "t1_evidently_drift_share":   t1_evid.get("drift_share", 0.0),
            "t1_drifted_feature_count":   len(t1_stat.get("drifted_features", [])),
        })

        if t1_evid.get("report_path") and Path(t1_evid["report_path"]).exists():
            mlflow.log_artifact(t1_evid["report_path"], artifact_path="drift_reports")

        # STEP 3: Tier-2 semantic checks (uses full reference/current with image_path)
        t2_semantic = detect_semantic_drift(reference, current)

        if t2_semantic.get("status") != "unavailable":
            mlflow.log_metrics({
                "t2_component_drift_share": t2_semantic.get("component_drift_share", 0.0),
                "t2_n_components":          t2_semantic.get("n_components", 0),
                "t2_drifted_pc_count":      len(t2_semantic.get("drifted_components", [])),
            })

        # STEP 4: Combined decision
        decision = make_drift_decision(t1_stat, t1_evid, t2_semantic)

        mlflow.log_params({
            "drift_detected": decision["drift_detected"],
            "severity":       decision["severity"],
            "action":         decision["action"],
        })
        mlflow.log_metrics({
            "combined_t1_drift_share":  decision["metrics"]["combined_tier1_drift_share"],
            "t2_semantic_drift_share":  decision["metrics"]["tier2_semantic_drift_share"],
            "n_heuristic_features":     decision["metrics"]["tier1_heuristic_features_count"],
        })
        if decision["drifted_features"]["tier2_semantic_pcs"]:
            mlflow.set_tag(
                "drifted_pcs",
                ",".join(decision["drifted_features"]["tier2_semantic_pcs"][:10]),
            )

        # STEP 5: Persist metrics
        end_time = datetime.now()
        save_drift_metrics({
            "pipeline_metadata": {
                "start_time": start_time.isoformat(),
                "end_time":   end_time.isoformat(),
                "duration_s": (end_time - start_time).total_seconds(),
                "version":    "2.0.0",
                "mlflow_run": mlflow_run.info.run_id if mlflow_run else None,
            },
            "decision":   decision,
            "tier1_stat": t1_stat,
            "tier1_evid": t1_evid,
            "tier2_sem":  t2_semantic,
        })
        mlflow.log_metric("pipeline_duration_s", (end_time - start_time).total_seconds())

        # STEP 6: Alert
        send_alerts(decision)
        mlflow.set_tag(
            "alert_triggered",
            "true" if decision["severity"] in ("high", "critical") else "false",
        )

        log.info("=" * 60)
        log.info("DRIFT DETECTION SUMMARY")
        log.info("  Drift detected : %s", decision["drift_detected"])
        log.info("  Severity       : %s", decision["severity"].upper())
        log.info("  Action         : %s", decision["action"])
        log.info("  T1 drift share : %.1f%%", decision["metrics"]["combined_tier1_drift_share"] * 100)
        log.info("  T2 drift share : %.1f%%", decision["metrics"]["tier2_semantic_drift_share"] * 100)
        log.info("  Label drift    : %s", decision["metrics"]["label_drift_detected"])
        log.info("=" * 60)

        mlflow.end_run(status="FINISHED")
        return decision

    except Exception as exc:
        log.error("Pipeline failed: %s", exc, exc_info=True)
        if mlflow_run:
            mlflow.log_param("error", str(exc)[:250])
            mlflow.end_run(status="FAILED")
        return {
            "status":    "failed",
            "error":     str(exc),
            "timestamp": start_time.isoformat(),
        }


if __name__ == "__main__":
    drift_detection_flow()