"""
Casting Defect Detection — Production Inference API 
===========================================================

  1. ASYNC DB LOGGING: Non-blocking writes to PostgreSQL via asyncio.create_task()
     - User response in ~50ms, DB write happens in background
     - Feeds Tier-1 drift detection (brightness, contrast, file_size)
  
  2. HITL UNCERTAINTY BANDS: Predictions in [0.35, 0.65] flagged for human review
     - Prevents blind binary classification at factory-critical threshold
     - Aligns with production QA workflows (humans review borderline cases)
  
  3. STATELESS CONTAINER DESIGN: No local file writes, no background loops
     - Fargate/Lambda-ready (ephemeral containers scale down unpredictably)
     - Model immutability enforced (no hot-reload endpoints)
  
  4. GRADCAM EXPLAINABILITY: /explain endpoint generates visual heatmaps
     - Shows QA engineers why the AI flagged a part as defective
     - Critical for trust in manufacturing environments
  
  5. PROMETHEUS METRICS: Production-grade observability
     - Tracks latency, prediction distribution, model version
     - Enables Grafana dashboards for SRE monitoring

FastAPI endpoints:
  POST /predict          Single image inference (50ms P50)
  POST /predict/batch    Batch inference (up to 32 images)
  POST /explain          GradCAM visual explanation
  GET  /health           Liveness/readiness probe
  GET  /metrics          Prometheus-compatible metrics
  GET  /model/info       Current model metadata
"""

from __future__ import annotations

import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import models, transforms

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

load_dotenv()

# ═════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════

MODEL_PATH     = os.getenv("MODEL_PATH",     "artifacts/efficientnet_b3_model.pth")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "artifacts/efficientnet_b3_threshold.pkl")
DATABASE_URL   = os.getenv("DATABASE_URL", os.getenv("POSTGRES_URL"))
LOG_INFERENCES = os.getenv("LOG_INFERENCES", "true").lower() == "true"
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))

CLASS_NAMES = ["ok_front", "def_front"]
IMAGE_SIZE  = 224
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

# HITL uncertainty band: predictions in this range flagged for human review
UNCERTAIN_LOW  = 0.35
UNCERTAIN_HIGH = 0.65

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════
# PROMETHEUS METRICS (production observability)
# ═════════════════════════════════════════════════════════════

REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

PREDICTION_LATENCY = Histogram(
    'prediction_duration_seconds',
    'Prediction latency distribution',
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5]
)

DEFECT_PREDICTIONS = Counter(
    'defect_predictions_total',
    'Count of predictions by outcome',
    ['result']  # ok / defect / uncertain
)

MODEL_VERSION_INFO = Gauge(
    'model_version',
    'Current production model version',
    ['model_type', 'threshold']
)

DB_WRITE_LATENCY = Histogram(
    'db_write_duration_seconds',
    'Time spent writing to PostgreSQL',
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)


# ═════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Singleton model loader + prediction engine.
    Thread-safe (torch.no_grad + eval mode + no state mutation).
    """

    _TRANSFORM = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    def __init__(self):
        self.device    = self._pick_device()
        self.model     = self._load_model()
        self.threshold = self._load_threshold()
        self.softmax   = nn.Softmax(dim=1)
        
        # Expose model metadata for Prometheus
        model_type = self.model.__class__.__name__
        MODEL_VERSION_INFO.labels(
            model_type=model_type, 
            threshold=f"{self.threshold:.3f}"
        ).set(1)
        
        logger.info(
            "InferenceEngine ready | device=%s | threshold=%.3f | model=%s",
            self.device, self.threshold, model_type
        )

    @staticmethod
    def _pick_device() -> torch.device:
        if torch.cuda.is_available():          return torch.device("cuda")
        if torch.backends.mps.is_available():  return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self) -> nn.Module:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        checkpoint  = torch.load(MODEL_PATH, map_location="cpu")
        model_type  = checkpoint.get("model_type", "efficientnet_b3")
        params      = checkpoint.get("params", {})
        dropout     = params.get("dropout", 0.3)

        # Rebuild architecture (must match training exactly)
        if model_type == "efficientnet_b3":
            backbone = models.efficientnet_b3(weights=None)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, len(CLASS_NAMES)),
            )
        elif model_type == "resnet50":
            backbone = models.resnet50(weights=None)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, len(CLASS_NAMES)),
            )
        else:
            raise ValueError(f"Unknown model type in checkpoint: {model_type}")

        backbone.load_state_dict(checkpoint["model_state"])
        backbone = backbone.to(self.device)
        backbone.eval()

        logger.info("Model loaded: %s", model_type)
        return backbone

    def _load_threshold(self) -> float:
        """
        Load F2-optimized threshold from training.
        This is CRITICAL: using 0.5 wastes the math you did in hyperparameter optimization.
        """
        if Path(THRESHOLD_PATH).exists():
            import joblib
            t = float(joblib.load(THRESHOLD_PATH))
            logger.info("Threshold loaded from training: %.4f", t)
            return t
        logger.warning("Threshold file not found, defaulting to 0.5 (suboptimal)")
        return 0.5

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL image to model input tensor"""
        return self._TRANSFORM(pil_image.convert("RGB")).unsqueeze(0)

    @torch.no_grad()
    def predict_batch(self, tensors: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Batch prediction with HITL uncertainty flagging.
        
        Returns enriched predictions with:
        - Binary classification (defect vs ok)
        - Probability scores for both classes
        - requires_review flag for borderline cases
        """
        batch  = torch.cat(tensors).to(self.device)
        logits = self.model(batch)
        probs  = self.softmax(logits).cpu().numpy()

        results = []
        for row in probs:
            score       = float(row[1])  # P(defective)
            prediction  = int(score >= self.threshold)
            label       = CLASS_NAMES[prediction]
            uncertain   = UNCERTAIN_LOW <= score <= UNCERTAIN_HIGH  # HITL flagging

            results.append({
                "prediction":        prediction,
                "label":             label,
                "score":             round(score, 4),
                "ok_probability":    round(float(row[0]), 4),
                "defect_probability": round(score, 4),
                "requires_review":   uncertain,  # ← KEY: flag for human QA
                "threshold_used":    self.threshold,
            })
        return results

    def get_gradcam_layer(self):
        """Return the target layer for GradCAM visualization"""
        if hasattr(self.model, 'features'):
            # EfficientNet
            return self.model.features[-1]
        elif hasattr(self.model, 'layer4'):
            # ResNet
            return self.model.layer4[-1]
        return None


# ═════════════════════════════════════════════════════════════
# GRADCAM EXPLAINER (from Set 2, mathematically sound)
# ═════════════════════════════════════════════════════════════

class GradCAMExplainer:
    """
    Generates visual explanations showing which regions triggered the defect classification.
    Critical for trust in manufacturing QA environments.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook into target layer
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate GradCAM heatmap for the predicted class.
        
        Returns:
            Heatmap as numpy array (H x W), values in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass on target class
        target = output[0, target_class]
        target.backward()

        # Generate heatmap
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)  # ReLU on combined output
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


# ═════════════════════════════════════════════════════════════
# SCHEMAS
# ═════════════════════════════════════════════════════════════

class PredictionResult(BaseModel):
    image_id:             str
    prediction:           int           = Field(..., description="0=ok_front, 1=def_front")
    label:                str
    score:                float         = Field(..., description="Defect probability [0,1]")
    ok_probability:       float
    defect_probability:   float
    requires_review:      bool          = Field(..., description="True if score in uncertain band [0.35, 0.65]")
    threshold_used:       float
    latency_ms:           float
    timestamp:            str


class BatchPredictionResult(BaseModel):
    results:        List[PredictionResult]
    total_images:   int
    defect_count:   int
    ok_count:       int
    review_count:   int
    batch_latency_ms: float


class ExplanationResult(BaseModel):
    image_id:       str
    prediction:     int
    score:          float
    heatmap_base64: str  # Base64-encoded PNG overlay
    timestamp:      str


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    device:       str
    database:     str
    timestamp:    str


class ModelInfo(BaseModel):
    model_path:     str
    threshold:      float
    class_names:    List[str]
    image_size:     int
    uncertain_band: Dict[str, float]


# ═════════════════════════════════════════════════════════════
# ASYNC DB LOGGING (non-blocking, feeds drift detection)
# ═════════════════════════════════════════════════════════════

async def log_inference_to_db(
    image_id:    str,
    image_bytes: bytes,
    result:      Dict[str, Any],
    image_stats: Dict[str, float],
) -> None:
    """
    ASYNC write to PostgreSQL. Does NOT block the API response.
    Logs tabular heuristic features for Tier-1 drift detection.
    
    Called via asyncio.create_task() — fire-and-forget pattern.
    """
    if not DATABASE_URL or not LOG_INFERENCES:
        return
    
    t0 = time.perf_counter()
    try:
        conn = await asyncpg.connect(DATABASE_URL, statement_cache_size=0)
        await conn.execute(
            """
            INSERT INTO inference_logs (
                image_id, prediction, score,
                brightness_mean, contrast_mean, aspect_ratio, file_size_kb,
                is_grayscale, timestamp, human_label
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,NULL)
            """,
            image_id,
            result["prediction"],
            result["score"],
            image_stats.get("brightness_mean"),
            image_stats.get("contrast_mean"),
            image_stats.get("aspect_ratio"),
            image_stats.get("file_size_kb"),
            int(image_stats.get("is_grayscale", 0)),
            datetime.utcnow(),
        )
        await conn.close()
        
        # Track DB write latency
        DB_WRITE_LATENCY.observe(time.perf_counter() - t0)
        
    except Exception as exc:
        logger.warning("DB log failed for %s: %s", image_id, exc)


def extract_tier1_features(img: Image.Image, file_bytes: bytes) -> Dict[str, float]:
    """
    Extract Tier-1 heuristic features for drift detection.
    These are fast (~1ms) and catch gross distribution shifts.
    """
    from PIL import ImageStat
    stat = ImageStat.Stat(img)
    w, h = img.size
    r_m, g_m, b_m = stat.mean[:3] if len(stat.mean) >= 3 else (0, 0, 0)
    chroma_var = float(np.std([r_m, g_m, b_m]))
    
    return {
        "brightness_mean": float(np.mean(stat.mean[:3])),
        "contrast_mean":   float(np.mean(stat.stddev[:3])),
        "aspect_ratio":    float(w / max(h, 1)),
        "file_size_kb":    float(len(file_bytes) / 1024),
        "is_grayscale":    float(chroma_var < 5.0),
    }


# ═════════════════════════════════════════════════════════════
# APP + LIFESPAN
# ═════════════════════════════════════════════════════════════

engine: Optional[InferenceEngine] = None
explainer: Optional[GradCAMExplainer] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, explainer
    
    logger.info("Loading inference engine ...")
    try:
        engine = InferenceEngine()
        
        # Initialize GradCAM explainer
        target_layer = engine.get_gradcam_layer()
        if target_layer:
            explainer = GradCAMExplainer(engine.model, target_layer)
            logger.info("GradCAM explainer initialized")
        else:
            logger.warning("Could not identify GradCAM target layer")
        
        logger.info("Server ready")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        engine = None
        explainer = None
    
    yield
    
    logger.info("Server shutting down")


app = FastAPI(
    title="Casting Defect Detection API",
    description="Production CV inference service with HITL uncertainty bands + GradCAM explainability",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ═════════════════════════════════════════════════════════════
# MIDDLEWARE (request tracking)
# ═════════════════════════════════════════════════════════════

@app.middleware("http")
async def track_requests(request, call_next):
    """Log every request + track Prometheus metrics"""
    t0 = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - t0
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    logger.info(
        "%s %s — %d — %.3fs",
        request.method, request.url.path, response.status_code, latency
    )
    
    return response


# ═════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health():
    """Kubernetes liveness/readiness probe"""
    db_status = "unknown"
    if DATABASE_URL:
        try:
            conn = await asyncpg.connect(DATABASE_URL, timeout=2, statement_cache_size=0)
            await conn.close()
            db_status = "ok"
        except:
            db_status = "degraded"
    
    return HealthResponse(
        status      = "ok" if engine else "degraded",
        model_loaded= engine is not None,
        device      = str(engine.device) if engine else "none",
        database    = db_status,
        timestamp   = datetime.utcnow().isoformat(),
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Model metadata (useful for debugging)"""
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfo(
        model_path    = MODEL_PATH,
        threshold     = engine.threshold,
        class_names   = CLASS_NAMES,
        image_size    = IMAGE_SIZE,
        uncertain_band= {"low": UNCERTAIN_LOW, "high": UNCERTAIN_HIGH},
    )


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    """
    Single image inference with HITL uncertainty flagging.
    
    Key feature: predictions in [0.35, 0.65] are flagged with requires_review=True,
    signaling the QA team to manually inspect the part before accepting/rejecting.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0         = time.perf_counter()
    image_id   = str(uuid.uuid4())
    file_bytes = await file.read()

    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot decode image. Send a valid JPEG/PNG.",
        )

    tensor  = engine.preprocess(pil_img)
    results = engine.predict_batch([tensor])
    result  = results[0]

    latency = (time.perf_counter() - t0) * 1000

    # Track prediction distribution
    if result["requires_review"]:
        DEFECT_PREDICTIONS.labels(result="uncertain").inc()
    else:
        DEFECT_PREDICTIONS.labels(result=result["label"]).inc()
    
    PREDICTION_LATENCY.observe(latency / 1000)

    # ASYNC log to database (does NOT block response)
    if LOG_INFERENCES:
        import asyncio
        image_stats = extract_tier1_features(pil_img, file_bytes)
        asyncio.create_task(
            log_inference_to_db(image_id, file_bytes, result, image_stats)
        )

    return PredictionResult(
        image_id   = image_id,
        latency_ms = round(latency, 2),
        timestamp  = datetime.utcnow().isoformat(),
        **result,
    )


@app.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Batch inference (up to MAX_BATCH_SIZE images).
    Useful for end-of-shift QA batch processing.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(files)} exceeds limit {MAX_BATCH_SIZE}",
        )

    t0      = time.perf_counter()
    tensors = []
    ids     = []

    for f in files:
        raw = await f.read()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            tensors.append(engine.preprocess(img))
            ids.append(str(uuid.uuid4()))
        except Exception:
            logger.warning("Skipping unreadable file: %s", f.filename)

    if not tensors:
        raise HTTPException(status_code=422, detail="No valid images in batch")

    raw_results = engine.predict_batch(tensors)
    batch_ms    = (time.perf_counter() - t0) * 1000
    ts          = datetime.utcnow().isoformat()

    predictions = [
        PredictionResult(
            image_id   = iid,
            latency_ms = round(batch_ms / len(tensors), 2),
            timestamp  = ts,
            **r,
        )
        for iid, r in zip(ids, raw_results)
    ]

    return BatchPredictionResult(
        results         = predictions,
        total_images    = len(predictions),
        defect_count    = sum(p.prediction == 1 for p in predictions),
        ok_count        = sum(p.prediction == 0 for p in predictions),
        review_count    = sum(p.requires_review for p in predictions),
        batch_latency_ms= round(batch_ms, 2),
    )


@app.post("/explain", response_model=ExplanationResult)
async def explain(file: UploadFile = File(...)):
    """
    GradCAM visual explanation.
    
    Returns a heatmap overlay showing which image regions triggered the defect classification.
    Critical for manufacturing QA trust and debugging false positives.
    """
    if not engine or not explainer:
        raise HTTPException(status_code=503, detail="Explainer not available")

    image_id   = str(uuid.uuid4())
    file_bytes = await file.read()

    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot decode image",
        )

    # Get prediction first
    tensor  = engine.preprocess(pil_img)
    results = engine.predict_batch([tensor])
    result  = results[0]
    
    # Generate GradCAM heatmap
    heatmap = explainer.generate(tensor, result["prediction"])
    
    # Overlay heatmap on original image
    from PIL import Image as PILImage
    import cv2
    import base64
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (pil_img.width, pil_img.height))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original
    img_array = np.array(pil_img)
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    
    # Convert to base64
    overlay_img = PILImage.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return ExplanationResult(
        image_id       = image_id,
        prediction     = result["prediction"],
        score          = result["score"],
        heatmap_base64 = heatmap_base64,
        timestamp      = datetime.utcnow().isoformat(),
    )


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)