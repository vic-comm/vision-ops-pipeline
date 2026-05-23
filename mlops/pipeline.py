"""
Casting Defect Detection — Training Pipeline Entrypoint
=======================================================
CLI entrypoint that wires together:
  - Environment validation
  - MLflow / DagsHub setup
  - Data preparation
  - Multi-model tournament (Optuna)
  - Model promotion
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

sys.path.append(str(Path(__file__).parent))
load_dotenv()

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from mlops.model_training import (
    Config,
    ModelType,
    load_and_prepare_data,
    promote_best_model,
    run_all_experiments,
)


def validate_environment(config: Config) -> bool:
    print("\n" + "=" * 60)
    print("ENVIRONMENT VALIDATION")
    print("=" * 60)

    issues = []

    required = [
        ("torch",        "PyTorch"),
        ("torchvision",  "TorchVision"),
        ("sklearn",      "scikit-learn"),
        ("mlflow",       "MLflow"),
        ("optuna",       "Optuna"),
        ("PIL",          "Pillow"),
    ]
    for pkg, name in required:
        try:
            __import__(pkg)
            print(f"   {name}")
        except ImportError:
            print(f"   {name} NOT FOUND")
            issues.append(f"pip install {pkg}")

    data_ok = any(
        (config.DATA_DIR / split / cls).exists()
        for split in ("train", "test")
        for cls  in ("ok_front", "def_front")
    )
    if data_ok:
        print(f"   Dataset found under {config.DATA_DIR}")
    else:
        print(f"   Dataset NOT found under {config.DATA_DIR}")
        issues.append(
            "Download from Kaggle: ravirajsinh45/real-life-industrial-dataset-of-casting-product"
        )

    try:
        config.setup_directories()
        print("   Output directories OK")
    except Exception as e:
        print(f"   Cannot create directories: {e}")
        issues.append("Check filesystem permissions")

    print("=" * 60)
    if issues:
        print("\nIssues found:")
        for i in issues:
            print(f"  - {i}")
        return False
    return True


def setup_mlflow(config: Config) -> MlflowClient:
    print("\n" + "=" * 60)
    print("MLFLOW SETUP")
    print("=" * 60)

    dagshub_repo  = os.getenv("DAGSHUB_REPO")
    dagshub_owner = os.getenv("DAGSHUB_OWNER")

    if dagshub_repo and dagshub_owner:
        try:
            import dagshub
            dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)
            print(f"   DagsHub: {dagshub_owner}/{dagshub_repo}")
        except Exception as e:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)
            print(f"   Fallback MLflow URI: {tracking_uri}")
    else:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"   MLflow URI: {tracking_uri}")

    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(config.EXPERIMENT_NAME)
        if not experiment:
            eid = mlflow.create_experiment(
                name=config.EXPERIMENT_NAME,
                artifact_location=os.getenv("MLFLOW_ARTIFACT_LOCATION", config.S3_BUCKET),
            )
            print(f"   Created experiment: {config.EXPERIMENT_NAME}")
        else:
            print(f"   Using experiment:   {config.EXPERIMENT_NAME}")
        mlflow.set_experiment(config.EXPERIMENT_NAME)
    except Exception as e:
        print(f"   Experiment setup failed: {e}")
        raise

    print("=" * 60)
    return client


def main_flow(
    use_dvc:         bool  = False,
    n_trials:        int   = 20,
    force_recompute: bool  = False,
    models:          Optional[str] = None,
) -> dict:
    print("\n" + "=" * 80)
    print(" " * 18 + "CASTING DEFECT DETECTION")
    print(" " * 20 + "TRAINING PIPELINE v2")
    print("=" * 80 + "\n")

    config = Config()
    config.setup_directories()

    if not validate_environment(config):
        print("\nEnvironment validation failed.")
        sys.exit(1)

    client = setup_mlflow(config)

    # Resolve model list
    model_map = {
        "efficientnet_b3": ModelType.EFFICIENTNET_B3,
        "resnet50":        ModelType.RESNET50,
        "mobilenet_v3":    ModelType.MOBILENET_V3,
    }
    models_to_train = None
    if models:
        models_to_train = [
            model_map[m.strip().lower()]
            for m in models.split(",")
            if m.strip().lower() in model_map
        ] or None
        if models_to_train:
            print(f"   Training: {[m.value for m in models_to_train]}")

    try:
        # Step 1: Data preparation
        training_data = load_and_prepare_data()

        # Step 2: Tournament
        winner_id, winner_f2, winner_name = run_all_experiments(
            data=training_data,
            n_trials=n_trials,
            models_to_train=models_to_train,
        )

        # Step 3: Promotion
        version = promote_best_model(
            winner_id=winner_id,
            winner_f2=winner_f2,
            winner_name=winner_name,
            client=client,
            experiment_name=config.EXPERIMENT_NAME,
        )

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"   Model   : {winner_name}")
        print(f"   F2      : {winner_f2:.4f}")
        print(f"   Version : {version}")
        print(f"   Run ID  : {winner_id}")
        print("=" * 80 + "\n")

        return {
            "model_name": winner_name,
            "f2_score":   winner_f2,
            "run_id":     winner_id,
            "version":    version,
        }

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Casting Defect Training Pipeline")
    parser.add_argument("--mode",            choices=["serve", "run"], default="run")
    parser.add_argument("--dvc",             action="store_true",
                        default=os.getenv("USE_DVC", "false").lower() == "true")
    parser.add_argument("--trials",          type=int,
                        default=int(os.getenv("OPTUNA_TRIALS", "20")))
    parser.add_argument("--force-recompute", action="store_true",
                        default=os.getenv("FORCE_RECOMPUTE", "false").lower() == "true")
    parser.add_argument("--models",          type=str,
                        default=os.getenv("TRAIN_MODELS", None))
    args = parser.parse_args()

    if args.mode == "serve":
        print("Prefect serve mode ...")
        main_flow.serve(
            name="casting-defect-training",
            parameters={
                "use_dvc": args.dvc,
                "n_trials": args.trials,
                "force_recompute": args.force_recompute,
                "models": args.models,
            },
        )
    else:
        result = main_flow(
            use_dvc=args.dvc,
            n_trials=args.trials,
            force_recompute=args.force_recompute,
            models=args.models,
        )
        if result:
            print(f"Training success — F2={result.get('f2_score', 0):.4f}")
        sys.exit(0)