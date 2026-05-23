"""
Casting Defect Detection — MLOps Master Orchestrator
=====================================================
Controls the full automated retraining lifecycle:

  STEP 1  Drift detection   (Tier-1 heuristic + Tier-2 semantic)
  STEP 2  Data ingestion    (validate & merge new labelled images)
  STEP 3  Retraining gate   (only retrain if drift warrants it)
  STEP 4  Model promotion   (promote winner if it beats production F2)

The severity-gated retraining logic prevents unnecessary compute spend
while ensuring the model retrains promptly when it matters.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

try:
    from prefect import flow, get_run_logger
except ImportError:
    def flow(*a, **kw):
        def _d(f): return f
        return _d
    def get_run_logger():
        return logging.getLogger(__name__)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/orchestrator.log"),
    ],
)
logger = logging.getLogger(__name__)


@flow(name="Casting Defect MLOps Master Pipeline")
def mlops_orchestrator(
    force_retrain: bool = False,
    n_trials: int = 20,
    models: Optional[str] = None,
):
    """
    Master pipeline. Runs on a schedule (e.g. daily via Prefect/Airflow/ECS
    Scheduled Tasks) and decides whether a retrain is warranted.

    Severity-gated retraining decision table:
        critical  -> immediate retrain
        high      -> immediate retrain
        medium    -> log for scheduled retrain (next weekly run)
        low/none  -> no retrain; continue monitoring
    """
    log = get_run_logger()
    start = datetime.now()
    log.info("=" * 70)
    log.info("CASTING DEFECT MLOPS MASTER PIPELINE — %s", start.isoformat())
    log.info("=" * 70)

    # ── STEP 1: Drift detection ───────────────────────────────────────
    log.info("\n[STEP 1] Drift Detection ...")
    from mlops.drift_pipeline import drift_detection_flow
    drift_decision = drift_detection_flow()

    severity = drift_decision.get("severity", "none")
    action   = drift_decision.get("action",   "no_action_needed")

    log.info("  Severity : %s", severity.upper())
    log.info("  T1 drift : %.1f%%",
             drift_decision.get("metrics", {}).get("combined_tier1_drift_share", 0) * 100)
    log.info("  T2 drift : %.1f%%",
             drift_decision.get("metrics", {}).get("tier2_semantic_drift_share", 0) * 100)
    log.info("  Action   : %s", action)

    # ── STEP 2: Data ingestion ────────────────────────────────────────
    log.info("\n[STEP 2] Data Ingestion ...")
    try:
        from mlops.ingest_data import data_ingestion_flow
        ingest_stats = data_ingestion_flow()
        ingestion_status = ingest_stats.get("status", "unknown")
        log.info("  Ingestion status : %s", ingestion_status)

        if ingestion_status == "failed":
            log.error("  Ingestion FAILED — data rejected (quality / adversarial check).")
            log.error("  Stopping pipeline. Do NOT retrain on rejected data.")
            return {"status": "stopped", "reason": "ingestion_failed"}

        if ingestion_status == "skipped":
            log.warning("  Ingestion SKIPPED — no valid new data found.")
            if not force_retrain:
                log.info("  Stopping pipeline (no new data + no force_retrain flag).")
                return {"status": "stopped", "reason": "no_new_data"}

    except ImportError:
        log.warning("  ingest_data module not found — skipping ingestion step.")
        ingestion_status = "skipped"

    # ── STEP 3: Retraining gate ───────────────────────────────────────
    log.info("\n[STEP 3] Retraining Decision ...")

    should_retrain = force_retrain or severity in ("high", "critical")

    if not should_retrain:
        if severity == "medium":
            log.info(
                "  Medium drift detected. Flagging for next scheduled retrain window."
            )
            _write_retrain_flag("medium")
        else:
            log.info("  Low/no drift. Model is stable. No retrain needed.")
        return {
            "status":   "stable",
            "severity": severity,
            "action":   action,
            "retrained": False,
        }

    log.warning("  %s drift → triggering retraining.", severity.upper())

    # ── STEP 4: Retrain + promote ─────────────────────────────────────
    log.info("\n[STEP 4] Model Retraining ...")
    from mlops.pipeline import main_flow
    result = main_flow(
        use_dvc=True,
        n_trials=n_trials,
        force_recompute=False,
        models=models,
    )

    duration = (datetime.now() - start).total_seconds()
    log.info("\n" + "=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info("  Model      : %s", result.get("model_name", "unknown"))
    log.info("  F2 Score   : %.4f", result.get("f2_score", 0))
    log.info("  Version    : %s", result.get("version", "?"))
    log.info("  Duration   : %.1f s", duration)
    log.info("=" * 70)

    return {
        "status":    "retrained",
        "severity":  severity,
        "result":    result,
        "duration_s": duration,
    }


def _write_retrain_flag(severity: str) -> None:
    """Write a flag file consumed by the weekly scheduled retrain job."""
    flag_path = Path("reports") / "retrain_flag.json"
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(flag_path, "w") as f:
        json.dump({
            "flagged_at": datetime.now().isoformat(),
            "severity": severity,
            "reason": "medium_drift_scheduled_retrain",
        }, f, indent=2)
    logger.info("  Retrain flag written -> %s", flag_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MLOps Master Orchestrator")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--models", type=str, default=None)
    args = parser.parse_args()

    mlops_orchestrator(
        force_retrain=args.force_retrain,
        n_trials=args.trials,
        models=args.models,
    )