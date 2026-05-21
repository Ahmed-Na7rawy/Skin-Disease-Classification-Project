"""
Skin Disease Classifier — REST API
====================================
FastAPI server exposing the EfficientNet skin-disease model
(best_skin_model_effnet.h5) as a REST endpoint.

Endpoints:
    POST /predict        — upload an image, get classification results
    GET  /health         — health check
    GET  /classes        — list supported skin disease classes

Run:
    cd api
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

# ── Constants (must match training) ──────────────────────────────────────────
IMG_SIZE = 224
CLASS_NAMES = ["Acne", "Eczema", "Fungal", "Melanoma", "Psoriasis", "Vitiligo"]
TTA_N = 7
MODEL_PATH = os.environ.get(
    "SKIN_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "model" / "best_skin_model_effnet.h5"),
)

# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Skin Disease Classifier API",
    description=(
        "REST API for classifying skin disease images using an EfficientNetV2 model. "
        "Supports 6 classes: Acne, Eczema, Fungal, Melanoma, Psoriasis, Vitiligo."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Response Models ──────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float
    tta_enabled: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_type: str


class ClassesResponse(BaseModel):
    classes: List[str]
    count: int


# ── Global model holder ─────────────────────────────────────────────────────
_model = None


def get_model():
    """Lazy-load the Keras model on first request."""
    global _model
    if _model is None:
        try:
            import tensorflow as tf

            print(f"[INFO] Loading model from: {MODEL_PATH}")
            _model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[INFO] Model loaded successfully. Type: {_model.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}", file=sys.stderr)
            raise RuntimeError(f"Model loading failed: {e}")
    return _model


# ── Pre-processing & TTA ────────────────────────────────────────────────────
def preprocess(img_bgr):
    """Resize to 224×224, BGR→RGB, float32. EfficientNet expects [0, 255]."""
    import cv2

    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return img


def tta_versions(img_bgr, n=TTA_N):
    """Return n augmented versions: original, h-flip, v-flip, 90°/180°/270°, center crop."""
    import cv2

    h, w = img_bgr.shape[:2]
    crops = [
        preprocess(img_bgr),
        preprocess(cv2.flip(img_bgr, 1)),
        preprocess(cv2.flip(img_bgr, 0)),
        preprocess(cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)),
        preprocess(cv2.rotate(img_bgr, cv2.ROTATE_180)),
        preprocess(cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]
    # Center crop (80%)
    cy, cx = h // 2, w // 2
    dy, dx = int(h * 0.4), int(w * 0.4)
    crop = img_bgr[cy - dy : cy + dy, cx - dx : cx + dx]
    crops.append(preprocess(crop))

    return np.stack(crops[:n], axis=0)


def decode_image(raw_bytes: bytes):
    """Decode raw image bytes to BGR numpy array via OpenCV."""
    import cv2

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")
    return img


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API and model are operational."""
    model_loaded = _model is not None
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_path=MODEL_PATH,
        model_type="EfficientNetV2",
    )


@app.get("/classes", response_model=ClassesResponse, tags=["System"])
async def list_classes():
    """Return the list of supported skin disease classes."""
    return ClassesResponse(classes=CLASS_NAMES, count=len(CLASS_NAMES))


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="Skin lesion image (JPEG/PNG)"),
    tta: bool = Query(True, description="Enable Test-Time Augmentation (7 augmented views)"),
):
    """
    Upload a skin lesion image and receive a classification prediction.

    The model outputs probabilities for 6 classes:
    Acne, Eczema, Fungal, Melanoma, Psoriasis, Vitiligo.

    With TTA enabled (default), the prediction averages 7 augmented views
    for more robust results.
    """
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload an image.",
        )

    # Read and decode image
    try:
        raw = await file.read()
        img_bgr = decode_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {e}")

    # Load model
    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Run inference
    t0 = time.perf_counter()

    if tta:
        batch = tta_versions(img_bgr)
        probs = model.predict(batch, verbose=0).mean(axis=0)
    else:
        batch = preprocess(img_bgr)[np.newaxis]
        probs = model.predict(batch, verbose=0)[0]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    probabilities = {cls: round(float(p), 6) for cls, p in zip(CLASS_NAMES, probs)}

    return PredictionResult(
        predicted_class=pred_class,
        confidence=round(confidence, 6),
        probabilities=probabilities,
        inference_time_ms=round(elapsed_ms, 2),
        tta_enabled=tta,
    )


# ── Startup event ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Log that the server is ready. Model loads lazily on first request."""
    print("[INFO] API server started. Model will load on first prediction request.")
