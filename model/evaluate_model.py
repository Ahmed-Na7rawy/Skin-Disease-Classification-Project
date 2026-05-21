"""
Skin Disease Classifier — Batch Evaluation Script
================================================
Evaluates .keras or .onnx models on the test dataset using folder mapping.
Supports both VGG19 and EfficientNetV2 architectures.
"""

import argparse
import os
import sys
import time
import warnings
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
CLASS_NAMES = ["Acne", "Eczema", "Fungal", "Melanoma", "Psoriasis", "Vitiligo"]
TTA_N       = 7
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Folder Mapping from Training Notebook
FOLDER_MAP = {
    'Acne': ['Acne and Rosacea Photos'],
    'Eczema': ['Eczema Photos', 'Atopic Dermatitis Photos'],
    'Psoriasis': ['Psoriasis pictures Lichen Planus and related diseases'],
    'Fungal': ['Tinea Ringworm Candidiasis and other Fungal Infections'],
    'Melanoma': ['Melanoma Skin Cancer Nevi and Moles'],
    'Vitiligo': ['Light Diseases and Disorders of Pigmentation']
}

# ── Lazy imports ─────────────────────────────────────────────────────────────
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

# ── Pre-processing & TTA ─────────────────────────────────────────────────────
def preprocess(img_bgr, model_type="vgg19"):
    cv2 = _import_cv2()
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    if model_type == "vgg19":
        # VGG19 ImageNet normalization (BGR order after flip)
        img[..., 0] -= 103.939
        img[..., 1] -= 116.779
        img[..., 2] -= 123.68
    else:
        # EfficientNetV2 handles its own scaling if using preprocess_input on [0, 255]
        pass
    return img

def tta_versions(img_bgr, model_type="vgg19", n=TTA_N):
    cv2 = _import_cv2()
    h, w = img_bgr.shape[:2]
    crops = []
    crops.append(preprocess(img_bgr, model_type))
    crops.append(preprocess(cv2.flip(img_bgr, 1), model_type))
    crops.append(preprocess(cv2.flip(img_bgr, 0), model_type))
    crops.append(preprocess(cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE), model_type))
    crops.append(preprocess(cv2.rotate(img_bgr, cv2.ROTATE_180), model_type))
    crops.append(preprocess(cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), model_type))
    cy, cx = h // 2, w // 2
    dy, dx = int(h * 0.4), int(w * 0.4)
    crop = img_bgr[cy - dy: cy + dy, cx - dx: cx + dx]
    crops.append(preprocess(crop, model_type))
    return np.stack(crops[:n], axis=0)

# ── Model Wrappers ───────────────────────────────────────────────────────────
class KerasModel:
    def __init__(self, path):
        tf = _import_keras()
        print(f"[INFO] Loading Keras model: {path}")
        self.model = tf.keras.models.load_model(path)
        self.model_type = "effnet" if "effnet" in path.lower() or "efficientnet" in self.model.name.lower() else "vgg19"
        print(f"[INFO] Detected model type: {self.model_type}")

    def predict_batch(self, batch):
        return self.model.predict(batch, verbose=0)

class OnnxModel:
    def __init__(self, path):
        ort = _import_onnx()
        print(f"[INFO] Loading ONNX model: {path}")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.model_type = "effnet" if "effnet" in path.lower() else "vgg19"
        print(f"[INFO] Detected model type: {self.model_type}")

    def predict_batch(self, batch):
        inp = batch.astype(np.float32)
        try:
            out = self.session.run([self.output_name], {self.input_name: inp})[0]
        except Exception:
            inp = np.transpose(inp, (0, 3, 1, 2))
            out = self.session.run([self.output_name], {self.input_name: inp})[0]
        return out

def load_model(model_path):
    ext = Path(model_path).suffix.lower()
    if ext in (".keras", ".h5"): return KerasModel(model_path)
    if ext == ".onnx": return OnnxModel(model_path)
    sys.exit(f"[ERROR] Unsupported model: {ext}")

# ── Dataset Loading ──────────────────────────────────────────────────────────
def build_eval_dataset(test_dir):
    test_dir = Path(test_dir)
    found_folders = {f.name.lower(): f.name for f in test_dir.iterdir() if f.is_dir()}
    
    files = []
    labels = []
    
    for label, folders in FOLDER_MAP.items():
        label_idx = CLASS_NAMES.index(label)
        for folder_target in folders:
            match = found_folders.get(folder_target.lower())
            if match:
                folder_path = test_dir / match
                for f in folder_path.iterdir():
                    if f.suffix.lower() in IMG_EXTS:
                        files.append(f)
                        labels.append(label_idx)
            else:
                print(f"[WARN] Folder not found: {folder_target}")
                
    return files, labels

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    use_tta = not args.no_tta
    model = load_model(args.model)
    model_type = getattr(model, "model_type", "vgg19")
    
    files, labels = build_eval_dataset(args.test_dir)
    
    if not files:
        sys.exit("[ERROR] No images found to evaluate.")
    
    print(f"[INFO] Starting evaluation on {len(files)} images...")
    
    results = []
    correct = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    
    t_start = time.perf_counter()
    cv2 = _import_cv2()

    for i, (fp, true_idx) in enumerate(zip(files, labels), 1):
        img_bgr = cv2.imread(str(fp))
        if img_bgr is None: continue

        if use_tta:
            batch = tta_versions(img_bgr, model_type)
            probs = model.predict_batch(batch).mean(axis=0)
        else:
            batch = preprocess(img_bgr, model_type)[np.newaxis]
            probs = model.predict_batch(batch)[0]

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        true_class = CLASS_NAMES[true_idx]
        confidence = float(probs[pred_idx])

        is_correct = (pred_idx == true_idx)
        if is_correct:
            correct += 1
            per_class_correct[true_class] += 1
        
        per_class_total[true_class] += 1
        
        results.append([str(fp), true_class, pred_class, f"{confidence:.4f}", is_correct])

        # Progress every 50 images or end
        if i % 50 == 0 or i == len(files):
            print(f"  Processed {i}/{len(files)}...")

    t_total = time.perf_counter() - t_start
    accuracy = (correct / len(files)) * 100

    print("\n" + "="*60)
    print("  EVALUATION SUMMARY")
    print("-" * 60)
    print(f"  Total Images  : {len(files)}")
    print(f"  Accuracy      : {accuracy:.2f}% ({correct}/{len(files)})")
    print(f"  Total Time    : {t_total:.1f}s")
    print(f"  Avg Time/Img  : {(t_total/len(files))*1000:.1f}ms")
    print("\n  Per-Class Accuracy:")
    for cls in CLASS_NAMES:
        total = per_class_total[cls]
        curr_correct = per_class_correct[cls]
        pct = (curr_correct / total * 100) if total > 0 else 0
        print(f"    {cls:<12} : {pct:6.2f}% ({curr_correct}/{total})")
    print("="*60 + "\n")

    if args.report:
        with open(args.report, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "true_class", "pred_class", "confidence", "correct"])
            writer.writerows(results)
        print(f"[INFO] Report saved to: {args.report}")

if __name__ == "__main__":
    main()