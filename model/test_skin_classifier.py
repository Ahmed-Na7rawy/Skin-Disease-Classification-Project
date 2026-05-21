"""
VGG19 Skin Disease Classifier — Test Script
============================================
Tests a trained VGG19 skin disease model (.keras or .onnx) on single
images, directories of images, or a held-out test dataset directory.

Supported classes:
    0: Acne
    1: Eczema
    2: Fungal
    3: Melanoma
    4: Psoriasis
    5: Vitiligo

Usage examples:
    # Test a single image with the Keras model
    python test_skin_classifier.py --model model.keras --image path/to/img.jpg

    # Test a single image with the ONNX model
    python test_skin_classifier.py --model model.onnx --image path/to/img.jpg

    # Evaluate on a labelled test directory (ImageFolder layout)
    python test_skin_classifier.py --model model.keras --test_dir data/test/

    # Disable TTA for faster (but less robust) inference
    python test_skin_classifier.py --model model.keras --image img.jpg --no_tta

    # Save a detailed CSV report
    python test_skin_classifier.py --model model.keras --test_dir data/test/ --report results.csv
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ── Constants (must match training) ──────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
CLASS_NAMES = ["Acne", "Eczema", "Fungal", "Melanoma", "Psoriasis", "Vitiligo"]
TTA_N       = 7          # number of augmented versions per image
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ── Lazy imports (only load what's needed) ───────────────────────────────────
def _import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        sys.exit("[ERROR] OpenCV not found. Install it:  pip install opencv-python")

def _import_keras():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        sys.exit("[ERROR] TensorFlow not found. Install it:  pip install tensorflow")

def _import_onnx():
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        sys.exit("[ERROR] ONNX Runtime not found. Install it:  pip install onnxruntime")

# ── Pre-processing ────────────────────────────────────────────────────────────
def preprocess(img_bgr, model_type="vgg19"):
    """
    Resize, convert BGR→RGB.
    Apply normalization based on model_type.
    - vgg19: subtract ImageNet channel means.
    - effnet: automatic rescaling [0, 1] is handled by Keras layer or manual if needed.
              For Keras Applications EfficientNetV2, preprocess_input is a pass-through 
              if rescale is included in the model.
    """
    cv2 = _import_cv2()
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    if model_type == "vgg19":
        # VGG19 preprocessing: subtract ImageNet channel means (BGR order after flip)
        img[..., 0] -= 103.939
        img[..., 1] -= 116.779
        img[..., 2] -= 123.68
    else:
        # EfficientNetV2 usually expects [0, 255] for preprocess_input or [0, 1] internally.
        # If using keras.applications.efficientnet_v2.preprocess_input, it's a null op 
        # as it expects pixels in [0, 255].
        pass
        
    return img

# ── Test-Time Augmentation ────────────────────────────────────────────────────
def tta_versions(img_bgr, model_type="vgg19", n=TTA_N):
    """
    Return `n` augmented variants of the image as a numpy array (n, H, W, 3).
    Augmentations: original, h-flip, v-flip, 90°/180°/270° rotations, centre-crop.
    """
    cv2   = _import_cv2()
    h, w  = img_bgr.shape[:2]
    crops = []

    # 1. Original
    crops.append(preprocess(img_bgr, model_type))

    # 2. Horizontal flip
    crops.append(preprocess(cv2.flip(img_bgr, 1), model_type))

    # 3. Vertical flip
    crops.append(preprocess(cv2.flip(img_bgr, 0), model_type))

    # 4. 90°
    crops.append(preprocess(cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE), model_type))

    # 5. 180°
    crops.append(preprocess(cv2.rotate(img_bgr, cv2.ROTATE_180), model_type))

    # 6. 270°
    crops.append(preprocess(cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), model_type))

    # 7. Centre crop (80 %) then resize
    cy, cx = h // 2, w // 2
    dy, dx = int(h * 0.4), int(w * 0.4)
    crop   = img_bgr[cy - dy: cy + dy, cx - dx: cx + dx]
    crops.append(preprocess(crop, model_type))

    return np.stack(crops[:n], axis=0)   # (n, 224, 224, 3)

# ── Model wrappers ────────────────────────────────────────────────────────────
class KerasModel:
    def __init__(self, path):
        tf = _import_keras()
        print(f"[INFO] Loading Keras model from: {path}")
        self.model = tf.keras.models.load_model(path)
        # Heuristic to detect model type
        self.model_type = "effnet" if "effnet" in path.lower() or "efficientnet" in self.model.name.lower() else "vgg19"
        print(f"[INFO] Detected model type: {self.model_type}")
        print("[INFO] Model loaded successfully.")

    def predict_batch(self, batch):
        """batch: numpy (N, 224, 224, 3) → softmax probabilities (N, 6)"""
        return self.model.predict(batch, verbose=0)


class OnnxModel:
    def __init__(self, path):
        ort = _import_onnx()
        print(f"[INFO] Loading ONNX model from: {path}")
        self.session = ort.InferenceSession(
            str(path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        # Heuristic to detect model type
        self.model_type = "effnet" if "effnet" in path.lower() else "vgg19"
        print(f"[INFO] Detected model type: {self.model_type}")
        print("[INFO] ONNX model loaded successfully.")

    def predict_batch(self, batch):
        """batch: numpy (N, 224, 224, 3) → softmax probabilities (N, 6)"""
        # ONNX models often expect NCHW; try both
        inp = batch.astype(np.float32)
        try:
            out = self.session.run([self.output_name], {self.input_name: inp})[0]
        except Exception:
            inp = np.transpose(inp, (0, 3, 1, 2))   # NHWC → NCHW
            out = self.session.run([self.output_name], {self.input_name: inp})[0]
        return out


def load_model(model_path):
    ext = Path(model_path).suffix.lower()
    if ext in (".keras", ".h5"):
        return KerasModel(model_path)
    elif ext == ".onnx":
        return OnnxModel(model_path)
    else:
        sys.exit(f"[ERROR] Unsupported model format '{ext}'. Use .keras or .onnx.")

# ── Inference helpers ─────────────────────────────────────────────────────────
def predict_image(model, img_bgr, use_tta=True):
    """
    Returns:
        probs      : averaged softmax probability array (6,)
        pred_class : predicted class name (str)
        confidence : probability of predicted class (float)
    """
    model_type = getattr(model, "model_type", "vgg19")
    if use_tta:
        batch  = tta_versions(img_bgr, model_type)         # (TTA_N, 224, 224, 3)
        probs  = model.predict_batch(batch)     # (TTA_N, 6)
        probs  = probs.mean(axis=0)             # (6,)
    else:
        batch  = preprocess(img_bgr, model_type)[np.newaxis]  # (1, 224, 224, 3)
        probs  = model.predict_batch(batch)[0]    # (6,)

    idx        = int(np.argmax(probs))
    pred_class = CLASS_NAMES[idx]
    confidence = float(probs[idx])
    return probs, pred_class, confidence


def load_image(path):
    cv2 = _import_cv2()
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img

# ── Single-image mode ─────────────────────────────────────────────────────────
def run_single(model, image_path, use_tta):
    img   = load_image(image_path)
    t0    = time.perf_counter()
    probs, pred_class, confidence = predict_image(model, img, use_tta)
    elapsed = time.perf_counter() - t0

    print("\n" + "═" * 50)
    print(f"  Image      : {image_path}")
    print(f"  Prediction : {pred_class}")
    print(f"  Confidence : {confidence * 100:.2f}%")
    print(f"  Inference  : {elapsed * 1000:.1f} ms  (TTA={'on' if use_tta else 'off'})")
    print("─" * 50)
    print("  Class Probabilities:")
    for cls, p in zip(CLASS_NAMES, probs):
        bar = "█" * int(p * 30)
        print(f"    {cls:<12} {p * 100:5.1f}%  {bar}")
    print("═" * 50 + "\n")

# ── Directory / dataset evaluation mode ──────────────────────────────────────
def run_directory(model, test_dir, use_tta, report_csv=None):
    """
    Expects either:
      (a) flat directory of images → no accuracy reported
      (b) ImageFolder layout: test_dir/<ClassName>/<image.jpg> → accuracy reported
    """
    test_dir   = Path(test_dir)
    subdirs    = [d for d in test_dir.iterdir() if d.is_dir()]
    labelled   = all(d.name in CLASS_NAMES for d in subdirs) and len(subdirs) > 0

    # Build file list
    if labelled:
        files  = []
        labels = []
        for d in subdirs:
            for f in d.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    files.append(f)
                    labels.append(CLASS_NAMES.index(d.name))
        print(f"[INFO] Labelled dataset detected — {len(files)} images across {len(subdirs)} classes.")
    else:
        files  = [f for f in test_dir.rglob("*") if f.suffix.lower() in IMG_EXTS]
        labels = [None] * len(files)
        print(f"[INFO] Unlabelled directory — {len(files)} images found.")

    if not files:
        sys.exit("[ERROR] No images found.")

    results = []   # (filepath, true_label, pred_label, confidence, probs...)
    correct = 0
    t_start = time.perf_counter()

    for i, (fp, true_idx) in enumerate(zip(files, labels), 1):
        try:
            img = load_image(fp)
        except ValueError as e:
            print(f"  [WARN] {e}")
            continue

        probs, pred_class, confidence = predict_image(model, img, use_tta)
        pred_idx = CLASS_NAMES.index(pred_class)

        if true_idx is not None and pred_idx == true_idx:
            correct += 1

        results.append((
            str(fp),
            CLASS_NAMES[true_idx] if true_idx is not None else "?",
            pred_class,
            confidence,
            *probs.tolist(),
        ))

        # Progress
        status = ""
        if true_idx is not None:
            status = "✓" if pred_idx == true_idx else f"✗ (true: {CLASS_NAMES[true_idx]})"
        print(f"  [{i:4d}/{len(files)}]  {fp.name:<35}  → {pred_class} ({confidence*100:.1f}%)  {status}")

    elapsed = time.perf_counter() - t_start

    # Summary
    print("\n" + "═" * 60)
    print("  EVALUATION SUMMARY")
    print("─" * 60)
    print(f"  Total images : {len(results)}")
    print(f"  Total time   : {elapsed:.1f}s  ({elapsed/len(results)*1000:.1f} ms/img)")
    if labelled:
        acc = correct / len(results) * 100
        print(f"  Accuracy     : {correct}/{len(results)}  ({acc:.2f}%)")

    if labelled:
        # Per-class accuracy
        from collections import defaultdict
        per_class_correct = defaultdict(int)
        per_class_total   = defaultdict(int)
        for r in results:
            true_cls, pred_cls = r[1], r[2]
            per_class_total[true_cls] += 1
            if true_cls == pred_cls:
                per_class_correct[true_cls] += 1
        print("\n  Per-class accuracy:")
        for cls in CLASS_NAMES:
            t = per_class_total.get(cls, 0)
            c = per_class_correct.get(cls, 0)
            pct = (c / t * 100) if t > 0 else 0.0
            bar = "█" * int(pct / 5)
            print(f"    {cls:<12} {c:3d}/{t:3d}  ({pct:5.1f}%)  {bar}")
    print("═" * 60 + "\n")

    # Optional CSV report
    if report_csv:
        import csv
        header = ["filepath", "true_class", "pred_class", "confidence"] + [f"prob_{c}" for c in CLASS_NAMES]
        with open(report_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(results)
        print(f"[INFO] Detailed report saved to: {report_csv}")

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Test a VGG19 skin disease classifier (.keras or .onnx).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",    required=True, help="Path to .keras or .onnx model file.")
    p.add_argument("--image",    default=None,  help="Path to a single image for inference.")
    p.add_argument("--test_dir", default=None,  help="Path to test directory (flat or ImageFolder layout).")
    p.add_argument("--no_tta",   action="store_true", help="Disable Test-Time Augmentation.")
    p.add_argument("--report",   default=None,  help="Path to save a CSV report (directory mode only).")
    return p.parse_args()


def main():
    args    = parse_args()
    use_tta = not args.no_tta

    if not args.image and not args.test_dir:
        sys.exit("[ERROR] Provide --image <path> or --test_dir <path>.")

    model = load_model(args.model)

    if args.image:
        run_single(model, args.image, use_tta)

    if args.test_dir:
        run_directory(model, args.test_dir, use_tta, args.report)


if __name__ == "__main__":
    main()