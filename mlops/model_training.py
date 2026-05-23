"""
Casting Defect Detection - Model Training Pipeline
===================================================

Architecture:
    - EfficientNet-B3 backbone (pretrained on ImageNet)
    - Fine-tuned for binary defect classification (ok_front / def_front)
    - Augmentation-heavy training to handle real industrial variation
    - Optuna hyperparameter optimization
    - MLflow experiment tracking + model registry
    - F2-score primary metric (recall-biased: missing a defect is worse than false alarm)
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    fbeta_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, accuracy_score,
    ConfusionMatrixDisplay, average_precision_score
)
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

try:
    from prefect import task
except ImportError:
    def task(log_prints=True):
        def decorator(func):
            return func
        return decorator


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

class Config:
    SCRIPT_DIR = Path(__file__).parent.absolute()
    DATA_DIR = SCRIPT_DIR / "../data"
    CACHE_DIR = SCRIPT_DIR / "../cache"
    ARTIFACTS_DIR = SCRIPT_DIR / "../artifacts"
    REPORTS_DIR = SCRIPT_DIR / "../reports"

    S3_BUCKET = os.getenv("S3_BUCKET", "s3://casting-defect-artifacts/mlflow")

    # Image settings
    IMAGE_SIZE = 224          # EfficientNet-B3 canonical input
    MEAN = [0.485, 0.456, 0.406]   # ImageNet stats
    STD  = [0.229, 0.224, 0.225]

    # Class mapping
    CLASS_NAMES = ["ok_front", "def_front"]
    NUM_CLASSES = 2

    # Training
    RANDOM_STATE = 42
    TEST_SIZE    = 0.30
    VAL_SIZE     = 0.50      # of the test portion → 15% val, 15% test
    BETA_SCORE   = 2         # F-beta: recall twice as important as precision
    BATCH_SIZE   = 32
    NUM_WORKERS  = 4

    EXPERIMENT_NAME = "casting-defect-detection"

    @classmethod
    def setup_directories(cls):
        for d in [cls.CACHE_DIR, cls.ARTIFACTS_DIR, cls.REPORTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class ModelMetrics:
    f2_score: float
    f1_score: float
    precision: float
    recall: float
    roc_auc: float
    accuracy: float
    avg_precision: float

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}

    def __str__(self) -> str:
        return (
            f"F2: {self.f2_score:.4f} | F1: {self.f1_score:.4f} | "
            f"Precision: {self.precision:.4f} | Recall: {self.recall:.4f} | "
            f"ROC-AUC: {self.roc_auc:.4f} | Accuracy: {self.accuracy:.4f}"
        )


@dataclass
class TrainingData:
    train_paths: List[str]
    val_paths:   List[str]
    test_paths:  List[str]
    train_labels: List[int]
    val_labels:   List[int]
    test_labels:  List[int]
    scale_pos_weight: float

    def get_shapes_summary(self) -> Dict[str, Any]:
        return {
            "train_size": len(self.train_paths),
            "val_size":   len(self.val_paths),
            "test_size":  len(self.test_paths),
            "scale_pos_weight": self.scale_pos_weight,
            "class_names": Config.CLASS_NAMES,
        }


class ModelType(Enum):
    EFFICIENTNET_B3  = "efficientnet_b3"
    RESNET50         = "resnet50"
    MOBILENET_V3     = "mobilenet_v3"


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class CastingDataset(Dataset):
    """
    PyTorch Dataset for casting product images.

    Expected directory layout (Kaggle dataset):
        data/
          train/
            ok_front/   *.jpeg
            def_front/  *.jpeg
          test/
            ok_front/   *.jpeg
            def_front/  *.jpeg
    """

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path  = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            # Return a blank image rather than crashing the training loop
            print(f"   ⚠️  Could not load {path}: {e}")
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(split: str) -> transforms.Compose:
    """
    Returns augmentation pipeline for each split.

    Train: heavy augmentation to simulate industrial variation
      - Random flips, rotations, color jitter, gaussian blur
      - Simulates lighting changes, orientation variation on conveyor
    Val/Test: only resize + normalize (no augmentation → fair eval)
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
            transforms.RandomCrop(Config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),   # simulate occlusions
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD),
        ])


# ─────────────────────────────────────────────
# DATA PREPARATOR
# ─────────────────────────────────────────────

class DataPreparator:
    """
    Walks the Kaggle dataset directory, builds stratified splits,
    and returns a TrainingData object.

    The Kaggle dataset has a train/ and test/ split already.
    We re-split everything ourselves for full control and reproducibility.
    """

    def __init__(self, config: Config):
        self.config = config

    @task(log_prints=True)
    def prepare_data(self) -> TrainingData:
        print("\n" + "=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)

        all_paths, all_labels = self._collect_image_paths()

        # Stratified 70 / 15 / 15 split
        X_train, X_temp, y_train, y_temp = train_test_split(
            all_paths, all_labels,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=all_labels,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.config.VAL_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y_temp,
        )

        n_ok  = sum(1 for l in y_train if l == 0)
        n_def = sum(1 for l in y_train if l == 1)
        scale_pos_weight = n_ok / max(n_def, 1)

        print(f"   Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
        print(f"   Class balance (train) → ok: {n_ok}  defective: {n_def}")
        print(f"   Scale pos weight: {scale_pos_weight:.2f}")

        return TrainingData(
            train_paths=X_train, val_paths=X_val, test_paths=X_test,
            train_labels=y_train, val_labels=y_val, test_labels=y_test,
            scale_pos_weight=scale_pos_weight,
        )

    def _collect_image_paths(self) -> Tuple[List[str], List[int]]:
        """
        Scan data/train and data/test directories for images.
        Label mapping: ok_front → 0,  def_front → 1
        """
        LABEL_MAP = {"ok_front": 0, "def_front": 1}
        IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".bmp"}

        paths, labels = [], []

        for subset in ["train", "test"]:
            subset_dir = self.config.DATA_DIR / subset
            if not subset_dir.exists():
                print(f"   ⚠️  Missing directory: {subset_dir}")
                continue

            for class_name, label in LABEL_MAP.items():
                class_dir = subset_dir / class_name
                if not class_dir.exists():
                    print(f"   ⚠️  Missing class dir: {class_dir}")
                    continue

                found = [
                    str(p) for p in class_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS
                ]
                paths.extend(found)
                labels.extend([label] * len(found))
                print(f"   [{subset}/{class_name}] → {len(found):,} images")

        if not paths:
            raise FileNotFoundError(
                f"No images found under {self.config.DATA_DIR}. "
                "Download the Kaggle dataset and place under data/train and data/test."
            )

        return paths, labels


# ─────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────

def build_model(model_type: ModelType, num_classes: int = 2, dropout: float = 0.3) -> nn.Module:
    """
    Build a pretrained backbone with a custom classification head.

    Design choices:
      - EfficientNet-B3: best accuracy/param trade-off for industrial inspection
      - ResNet50:        robust, widely deployed, easier to debug
      - MobileNetV3:    lightweight, suitable for edge deployment
    """
    if model_type == ModelType.EFFICIENTNET_B3:
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif model_type == ModelType.RESNET50:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif model_type == ModelType.MOBILENET_V3:
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

class MetricsCalculator:

    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> ModelMetrics:
        return ModelMetrics(
            f2_score      = fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            f1_score      = fbeta_score(y_true, y_pred, beta=1, zero_division=0),
            precision     = precision_score(y_true, y_pred, zero_division=0),
            recall        = recall_score(y_true, y_pred, zero_division=0),
            roc_auc       = roc_auc_score(y_true, y_proba),
            accuracy      = accuracy_score(y_true, y_pred),
            avg_precision = average_precision_score(y_true, y_proba),
        )

    @staticmethod
    def optimize_threshold(
        y_proba: np.ndarray,
        y_true:  np.ndarray,
        beta:    int = 2,
    ) -> Tuple[float, float]:
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = [
            fbeta_score(y_true, (y_proba >= t).astype(int), beta=beta, zero_division=0)
            for t in thresholds
        ]
        idx = int(np.argmax(scores))
        return float(thresholds[idx]), float(scores[idx])

    @staticmethod
    def log_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_dir: Path,
    ) -> Path:
        cm   = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=Config.CLASS_NAMES)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap="Blues", ax=ax)
        ax.set_title(f"{model_name} – Confusion Matrix")
        path = save_dir / f"confusion_matrix_{model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    @staticmethod
    def log_training_curves(
        history: Dict[str, List[float]],
        model_name: str,
        save_dir: Path,
    ) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history["train_loss"], label="Train Loss")
        axes[0].plot(history["val_loss"],   label="Val Loss")
        axes[0].set_title("Loss"); axes[0].legend()
        axes[1].plot(history["train_f2"], label="Train F2")
        axes[1].plot(history["val_f2"],   label="Val F2")
        axes[1].set_title("F2 Score"); axes[1].legend()
        fig.suptitle(f"{model_name} Training Curves")
        path = save_dir / f"training_curves_{model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path


# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────

class ModelTrainer:
    """
    End-to-end training loop with:
      - Weighted random sampling (class imbalance)
      - Mixed precision (AMP) for speed
      - Cosine annealing LR schedule
      - Early stopping on val F2
      - Full MLflow tracking
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = self._get_device()
        self.metrics = MetricsCalculator()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @task(log_prints=True)
    def train_model(
        self,
        model_type: ModelType,
        data: TrainingData,
        params: Dict[str, Any],
    ) -> Tuple[str, float]:
        """
        Train one model configuration and return (run_id, test_f2).
        Called either directly or from the Optuna objective.
        """
        print(f"\n{'=' * 60}")
        print(f"TRAINING {model_type.value.upper()}")
        print(f"  LR={params['lr']:.5f}  Epochs={params['epochs']}  "
              f"Dropout={params['dropout']:.2f}  WD={params['weight_decay']:.5f}")
        print("=" * 60)

        with mlflow.start_run(run_name=f"{model_type.value}_final", nested=False) as run:
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_type.value)
            mlflow.log_param("image_size", Config.IMAGE_SIZE)
            mlflow.set_tag("backbone", model_type.value)

            # ── Data loaders ─────────────────────────────
            train_loader = self._make_loader(
                data.train_paths, data.train_labels, split="train", weighted=True
            )
            val_loader = self._make_loader(
                data.val_paths, data.val_labels, split="val"
            )
            test_loader = self._make_loader(
                data.test_paths, data.test_labels, split="test"
            )

            # ── Model + loss + optimizer ──────────────────
            model = build_model(model_type, dropout=params["dropout"]).to(self.device)

            # Class-weighted loss to handle imbalance
            pos_weight = torch.tensor([data.scale_pos_weight]).to(self.device)
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, float(data.scale_pos_weight)]).to(self.device)
            )

            optimizer = optim.AdamW(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params["epochs"], eta_min=1e-6
            )

            # AMP scaler
            use_amp = self.device.type in ("cuda",)
            scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

            # ── Training loop ─────────────────────────────
            history = {"train_loss": [], "val_loss": [], "train_f2": [], "val_f2": []}
            best_val_f2   = 0.0
            best_state    = None
            patience_ctr  = 0
            patience      = params.get("patience", 7)

            for epoch in range(params["epochs"]):
                train_loss, train_f2 = self._train_epoch(
                    model, train_loader, criterion, optimizer, scaler, use_amp
                )
                val_loss, val_f2, _, _ = self._eval_epoch(model, val_loader, criterion)

                scheduler.step()

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["train_f2"].append(train_f2)
                history["val_f2"].append(val_f2)

                mlflow.log_metrics({
                    "train_loss": train_loss, "val_loss": val_loss,
                    "train_f2":   train_f2,   "val_f2":   val_f2,
                    "lr": scheduler.get_last_lr()[0],
                }, step=epoch)

                print(
                    f"   Epoch {epoch + 1:3d}/{params['epochs']}  "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"val_f2={val_f2:.4f}"
                )

                if val_f2 > best_val_f2:
                    best_val_f2 = val_f2
                    best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        print(f"   ⏹  Early stopping at epoch {epoch + 1}")
                        break

            # ── Load best weights & final eval ────────────
            if best_state:
                model.load_state_dict(best_state)

            _, _, y_val_proba, y_val_true = self._eval_epoch(model, val_loader, criterion)
            opt_threshold, _ = self.metrics.optimize_threshold(y_val_proba, y_val_true)

            _, _, y_test_proba, y_test_true = self._eval_epoch(model, test_loader, criterion)
            y_test_pred = (y_test_proba >= opt_threshold).astype(int)
            test_metrics = self.metrics.calculate(y_test_true, y_test_pred, y_test_proba)

            print(f"\n   ✅ Test Metrics: {test_metrics}")
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.to_dict().items()})
            mlflow.log_param("optimal_threshold", opt_threshold)

            # ── Artifacts ─────────────────────────────────
            cm_path = self.metrics.log_confusion_matrix(
                y_test_true, y_test_pred, model_type.value, self.config.ARTIFACTS_DIR
            )
            curves_path = self.metrics.log_training_curves(
                history, model_type.value, self.config.ARTIFACTS_DIR
            )
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(curves_path))

            # ── Save model ────────────────────────────────
            model_path    = self.config.ARTIFACTS_DIR / f"{model_type.value}_model.pth"
            threshold_path = self.config.ARTIFACTS_DIR / f"{model_type.value}_threshold.pkl"
            torch.save({"model_state": best_state or model.state_dict(),
                        "model_type": model_type.value,
                        "params": params}, model_path)
            joblib.dump(opt_threshold, threshold_path)

            mlflow.log_artifact(str(model_path))
            mlflow.set_tag("model_type", model_type.value)
            mlflow.set_tag("f2_score", f"{test_metrics.f2_score:.4f}")

            return run.info.run_id, test_metrics.f2_score

    def _train_epoch(
        self,
        model:     nn.Module,
        loader:    DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scaler,
        use_amp:   bool,
    ) -> Tuple[float, float]:
        model.train()
        total_loss  = 0.0
        all_preds   = []
        all_labels  = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        f2       = fbeta_score(all_labels, all_preds, beta=2, zero_division=0)
        return avg_loss, f2

    @torch.no_grad()
    def _eval_epoch(
        self,
        model:     nn.Module,
        loader:    DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        model.eval()
        total_loss = 0.0
        all_proba  = []
        all_labels = []

        softmax = nn.Softmax(dim=1)

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            proba = softmax(outputs)[:, 1].cpu().numpy()
            all_proba.extend(proba)
            all_labels.extend(labels.cpu().numpy())

        all_proba  = np.array(all_proba)
        all_labels = np.array(all_labels)
        all_preds  = (all_proba >= 0.5).astype(int)
        avg_loss   = total_loss / len(loader.dataset)
        f2         = fbeta_score(all_labels, all_preds, beta=2, zero_division=0)

        return avg_loss, f2, all_proba, all_labels

    def _make_loader(
        self,
        paths:    List[str],
        labels:   List[int],
        split:    str,
        weighted: bool = False,
    ) -> DataLoader:
        transform = get_transforms(split)
        dataset   = CastingDataset(paths, labels, transform=transform)

        sampler = None
        shuffle = split == "train"

        if weighted and split == "train":
            counts  = np.bincount(labels)
            weights = 1.0 / counts[labels]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False

        return DataLoader(
            dataset,
            batch_size  = Config.BATCH_SIZE,
            shuffle     = shuffle,
            sampler     = sampler,
            num_workers = Config.NUM_WORKERS,
            pin_memory  = True,
        )


# ─────────────────────────────────────────────
# HYPERPARAMETER OPTIMIZATION
# ─────────────────────────────────────────────

class HyperparameterOptimizer:
    """Optuna study per model type, optimising val F2."""

    def __init__(self, config: Config, data: TrainingData):
        self.config  = config
        self.data    = data
        self.trainer = ModelTrainer(config)

    def create_objective(self, model_type: ModelType):
        def objective(trial):
            params = {
                "lr":           trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
                "dropout":      trial.suggest_float("dropout", 0.1, 0.5),
                "epochs":       trial.suggest_int("epochs", 5, 25),
                "patience":     7,
            }
            with mlflow.start_run(run_name=f"{model_type.value}_trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                _, val_f2 = self.trainer.train_model(model_type, self.data, params)
            return val_f2
        return objective


# ─────────────────────────────────────────────
# EXPERIMENT RUNNER
# ─────────────────────────────────────────────

def run_all_experiments(
    data: TrainingData,
    n_trials: int = 20,
    models_to_train: Optional[List[ModelType]] = None,
) -> Tuple[str, float, str]:
    """
    Run Optuna tuning for each model, return (best_run_id, best_f2, best_model_name).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if models_to_train is None:
        models_to_train = list(ModelType)

    config  = Config()
    opt     = HyperparameterOptimizer(config, data)
    trainer = ModelTrainer(config)
    results = []

    for model_type in models_to_train:
        print(f"\n{'='*60}\nOPTIMISING {model_type.value.upper()}\n{'='*60}")
        try:
            study = optuna.create_study(
                direction="maximize",
                study_name=f"{model_type.value}_study",
                sampler=optuna.samplers.TPESampler(seed=Config.RANDOM_STATE),
            )
            study.optimize(opt.create_objective(model_type), n_trials=n_trials, show_progress_bar=True)
            best_params = study.best_params
            best_params["patience"] = 7
            best_params["epochs"]   = max(best_params.get("epochs", 15), 15)

            print(f"\n   Best trial val F2: {study.best_value:.4f}")
            print(f"   Best params: {best_params}")

            run_id, test_f2 = trainer.train_model(model_type, data, best_params)
            results.append((model_type.value, run_id, test_f2))

        except Exception as e:
            import traceback
            print(f"\n❌ Error training {model_type.value}: {e}")
            traceback.print_exc()

    if not results:
        raise RuntimeError("All model training attempts failed.")

    results.sort(key=lambda x: x[2], reverse=True)
    winner_name, winner_id, winner_f2 = results[0]

    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    for name, rid, f2 in results:
        emoji = "🏆" if name == winner_name else "  "
        print(f"{emoji} {name:20s} | F2: {f2:.4f} | Run: {rid}")
    print(f"\n🏆 WINNER: {winner_name}  F2={winner_f2:.4f}")
    print("=" * 60)

    return winner_id, winner_f2, winner_name


# ─────────────────────────────────────────────
# MODEL PROMOTION
# ─────────────────────────────────────────────

def promote_best_model(
    winner_id:   str,
    winner_f2:   float,
    winner_name: str,
    client:      MlflowClient,
    experiment_name: str = "casting-defect-detection",
) -> int:
    print("\n" + "=" * 60)
    print("MODEL PROMOTION")
    print("=" * 60)

    model_uri = f"runs:/{winner_id}/artifacts/{winner_name}_model.pth"
    print(f"   Model URI: runs:/{winner_id}/...")

    # Check current production F2
    prod_f2 = 0.0
    try:
        prod_versions = client.get_latest_versions(experiment_name, stages=["Production"])
        if prod_versions:
            prod_run = client.get_run(prod_versions[0].run_id)
            prod_f2  = prod_run.data.metrics.get("test_f2_score", 0.0)
            print(f"   Current production F2: {prod_f2:.4f}")
    except Exception as e:
        print(f"   Could not fetch production F2: {e}")

    # Register
    try:
        mv = mlflow.register_model(f"runs:/{winner_id}/model", name=experiment_name)
    except Exception as e:
        print(f"   Registration failed: {e}")
        raise

    if winner_f2 > prod_f2:
        client.transition_model_version_stage(
            name=experiment_name, version=mv.version,
            stage="Production", archive_existing_versions=True,
        )
        client.set_model_version_tag(experiment_name, mv.version, "model_type", winner_name)
        client.set_model_version_tag(experiment_name, mv.version, "f2_score", str(winner_f2))
        print(f"\n   ✅ Promoted v{mv.version} to Production (F2={winner_f2:.4f})")
    else:
        client.transition_model_version_stage(
            name=experiment_name, version=mv.version, stage="Archived"
        )
        print(f"\n   📦 Archived v{mv.version} (did not beat prod F2={prod_f2:.4f})")

    return mv.version


# ─────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ─────────────────────────────────────────────

def load_and_prepare_data() -> TrainingData:
    config = Config()
    config.setup_directories()
    return DataPreparator(config).prepare_data()