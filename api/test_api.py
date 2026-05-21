"""
Skin Disease Classifier API — Test Script
===========================================
Sends test images to the FastAPI server and validates responses.

Usage:
    python test_api.py                          # run all tests
    python test_api.py --base-url http://host:8000  # custom URL
    python test_api.py --image path/to/img.jpg  # test with a specific image
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("[ERROR] 'requests' is required. Install it:  pip install requests")


# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
CLASS_NAMES = ["Acne", "Eczema", "Fungal", "Melanoma", "Psoriasis", "Vitiligo"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Test dataset directory (same structure used for training/evaluation)
TEST_DIR = Path(__file__).resolve().parent.parent / "test"


# ── Helpers ──────────────────────────────────────────────────────────────────
def separator(title=""):
    line = "═" * 60
    if title:
        print(f"\n{line}\n  {title}\n{'─' * 60}")
    else:
        print(line)


def print_result(result: dict, label=""):
    pred = result["predicted_class"]
    conf = result["confidence"] * 100
    tta = "ON" if result["tta_enabled"] else "OFF"
    ms = result["inference_time_ms"]
    print(f"  Prediction : {pred}")
    print(f"  Confidence : {conf:.2f}%")
    print(f"  TTA        : {tta}")
    print(f"  Latency    : {ms:.1f} ms")
    if label:
        match = "✓ CORRECT" if pred == label else f"✗ WRONG (expected: {label})"
        print(f"  Ground Truth: {label}  →  {match}")
    print()
    # Probability bar chart
    probs = result["probabilities"]
    print("  Class Probabilities:")
    for cls in CLASS_NAMES:
        p = probs.get(cls, 0) * 100
        bar = "█" * int(p / 3)
        print(f"    {cls:<12} {p:5.1f}%  {bar}")
    print()


# ── Test Functions ───────────────────────────────────────────────────────────
def test_health(base_url: str) -> bool:
    """Test the /health endpoint."""
    separator("TEST: Health Check")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        r.raise_for_status()
        data = r.json()
        print(f"  Status       : {data['status']}")
        print(f"  Model Loaded : {data['model_loaded']}")
        print(f"  Model Path   : {data['model_path']}")
        print(f"  Model Type   : {data['model_type']}")
        print("  ✓ Health check PASSED")
        return True
    except Exception as e:
        print(f"  ✗ Health check FAILED: {e}")
        return False


def test_classes(base_url: str) -> bool:
    """Test the /classes endpoint."""
    separator("TEST: List Classes")
    try:
        r = requests.get(f"{base_url}/classes", timeout=10)
        r.raise_for_status()
        data = r.json()
        print(f"  Classes : {data['classes']}")
        print(f"  Count   : {data['count']}")
        assert data["count"] == 6, f"Expected 6 classes, got {data['count']}"
        assert data["classes"] == CLASS_NAMES, "Class list mismatch"
        print("  ✓ Classes check PASSED")
        return True
    except Exception as e:
        print(f"  ✗ Classes check FAILED: {e}")
        return False


def test_predict_single(base_url: str, image_path: str, label: str = "", tta: bool = True) -> bool:
    """Test the /predict endpoint with a single image."""
    separator(f"TEST: Predict — {Path(image_path).name} (TTA={'ON' if tta else 'OFF'})")
    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            params = {"tta": str(tta).lower()}
            t0 = time.perf_counter()
            r = requests.post(f"{base_url}/predict", files=files, params=params, timeout=120)
            wall_time = (time.perf_counter() - t0) * 1000

        r.raise_for_status()
        data = r.json()
        print(f"  Wall time (incl. network): {wall_time:.1f} ms")
        print_result(data, label)

        # Validate response structure
        assert "predicted_class" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert data["predicted_class"] in CLASS_NAMES
        assert 0.0 <= data["confidence"] <= 1.0
        assert len(data["probabilities"]) == 6

        print("  ✓ Prediction PASSED")
        return True
    except Exception as e:
        print(f"  ✗ Prediction FAILED: {e}")
        return False


def test_invalid_file(base_url: str) -> bool:
    """Test that the API rejects non-image files."""
    separator("TEST: Invalid File Upload")
    try:
        files = {"file": ("test.txt", b"this is not an image", "text/plain")}
        r = requests.post(f"{base_url}/predict", files=files, timeout=30)
        if r.status_code == 400:
            print(f"  Got expected 400 error: {r.json().get('detail', '')}")
            print("  ✓ Invalid file rejection PASSED")
            return True
        else:
            print(f"  ✗ Expected 400 but got {r.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Invalid file test FAILED: {e}")
        return False


def test_no_tta(base_url: str, image_path: str) -> bool:
    """Test prediction with TTA disabled."""
    return test_predict_single(base_url, image_path, tta=False)


def find_test_images(test_dir: Path, max_per_class: int = 1):
    """Find sample images from the test dataset for automated testing."""
    samples = []
    # Folder mapping from training
    folder_map = {
        "Acne": ["Acne and Rosacea Photos"],
        "Eczema": ["Eczema Photos", "Atopic Dermatitis Photos"],
        "Psoriasis": ["Psoriasis pictures Lichen Planus and related diseases"],
        "Fungal": ["Tinea Ringworm Candidiasis and other Fungal Infections"],
        "Melanoma": ["Melanoma Skin Cancer Nevi and Moles"],
        "Vitiligo": ["Light Diseases and Disorders of Pigmentation"],
    }
    for label, folders in folder_map.items():
        for folder_name in folders:
            folder = test_dir / folder_name
            if folder.exists():
                count = 0
                for f in folder.iterdir():
                    if f.suffix.lower() in IMG_EXTS and count < max_per_class:
                        samples.append((str(f), label))
                        count += 1
                break  # one folder per class is enough
    return samples


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test the Skin Disease Classifier API")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument("--image", default=None, help="Path to a specific image to test")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA for tests")
    parser.add_argument("--max-per-class", type=int, default=1, help="Max images per class from test set")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Skin Disease Classifier API — Test Suite            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Target: {args.base_url}")
    print()

    results = []

    # 1. Health check
    results.append(("Health Check", test_health(args.base_url)))

    # 2. List classes
    results.append(("List Classes", test_classes(args.base_url)))

    # 3. Invalid file rejection
    results.append(("Invalid File", test_invalid_file(args.base_url)))

    # 4. Single image prediction(s)
    if args.image:
        # User-specified image
        results.append(("Predict (user image)", test_predict_single(
            args.base_url, args.image, tta=not args.no_tta
        )))
    else:
        # Auto-discover from test dataset
        samples = find_test_images(TEST_DIR, max_per_class=args.max_per_class)
        if samples:
            print(f"\n[INFO] Found {len(samples)} test images from dataset.")
            for img_path, label in samples:
                name = f"Predict ({label})"
                results.append((name, test_predict_single(
                    args.base_url, img_path, label=label, tta=not args.no_tta
                )))
        else:
            print("[WARN] No test images found. Skipping prediction tests.")
            print(f"       Expected test directory: {TEST_DIR}")

    # 5. TTA off test (pick first available image)
    if args.image:
        test_img = args.image
    else:
        samples = find_test_images(TEST_DIR, max_per_class=1)
        test_img = samples[0][0] if samples else None

    if test_img:
        results.append(("Predict (no TTA)", test_no_tta(args.base_url, test_img)))

    # ── Summary ──────────────────────────────────────────────────────────────
    separator("TEST SUMMARY")
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    for name, ok in results:
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {name}")
    print(f"\n  Total: {len(results)} tests  |  Passed: {passed}  |  Failed: {failed}")

    if failed > 0:
        print("\n  ⚠  Some tests failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("\n  🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
