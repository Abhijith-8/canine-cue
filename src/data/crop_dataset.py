"""
crop_dataset.py

Walks through the split dataset (train/val/test x calm/aggressive) and crops
the dog region from each image using YOLOv8 detection (DogDetector).
Saves cropped images to a parallel output directory, preserving folder structure.

If no dog is detected in an image, it's copied as-is (with a warning) so you
don't silently lose data — you can review/remove these later.

NEW: writes a fallback_log.csv in the output_dir listing every image that
fell back to a plain copy, along with the best detection confidence found
(or "none" if YOLO found zero dog boxes at all). Use this to decide whether
you need a lower --conf_thresh or a bigger model (yolov8s.pt / yolov8m.pt).

Usage:
    python crop_dataset.py --input_dir ../../data/processed --output_dir ../../data/processed_cropped
"""

import argparse
import csv
import shutil
from pathlib import Path
from PIL import Image
import sys

# adjust import path to reach src/detection/yolo_detector.py
sys.path.append(str(Path(__file__).resolve().parent.parent))
from detection.yolo_detector import DogDetector

SPLITS = ["train", "val", "test"]
CLASSES = ["calm", "aggressive"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def crop_image(detector, img_path: Path, out_path: Path,
                conf_thresh: float = 0.25, padding: float = 0.10):
    """
    Detect dog(s) in img_path, crop the highest-confidence detection (with a bit
    of padding so we don't cut off ears/tail), and save to out_path.

    Returns (cropped_ok, best_conf_seen):
        cropped_ok      - True if a crop was made, False if fallback copy
        best_conf_seen  - highest confidence YOLO found for *any* dog box,
                           even if below conf_thresh (None if zero dog boxes)
    """
    try:
        detections = detector.detect(str(img_path))
    except Exception as e:
        print(f"  [SKIP] Detection failed on {img_path}: {e}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, out_path)
        return False, None

    # track best confidence seen regardless of threshold, for diagnostics
    best_conf_seen = max((d["confidence"] for d in detections), default=None)

    dog_dets = [d for d in detections if d["confidence"] >= conf_thresh]

    if not dog_dets:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, out_path)
        return False, best_conf_seen

    best_det = max(dog_dets, key=lambda d: d["confidence"])
    x1, y1, x2, y2 = best_det["bbox"]

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  [SKIP] Could not open {img_path}: {e}")
        return False, best_conf_seen

    w, h = img.size

    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = box_w * padding, box_h * padding

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    cropped = img.crop((x1, y1, x2, y2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(out_path)
    return True, best_conf_seen


def main():
    parser = argparse.ArgumentParser(description="Crop dog regions from split dataset using YOLO.")
    parser.add_argument("--input_dir", required=True, help="Path to split dataset root (contains train/val/test)")
    parser.add_argument("--output_dir", required=True, help="Path to save cropped dataset")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Min YOLO confidence to accept a dog detection")
    parser.add_argument("--model_path", default=None, help="Optional path to custom YOLO weights (defaults to yolov8n.pt inside DogDetector)")
    args = parser.parse_args()

    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    if not input_root.exists():
        print(f"ERROR: input_dir does not exist: {input_root}")
        sys.exit(1)

    print(f"Loading YOLO detector...")
    detector_kwargs = {"conf_threshold": args.conf_thresh}
    if args.model_path:
        detector_kwargs["model_path"] = args.model_path
    detector = DogDetector(**detector_kwargs)

    total_images = 0
    total_cropped = 0
    total_fallback = 0

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "fallback_log.csv"
    log_rows = []

    for split in SPLITS:
        for cls in CLASSES:
            src_dir = input_root / split / cls
            if not src_dir.exists():
                print(f"[WARN] Missing folder, skipping: {src_dir}")
                continue

            dst_dir = output_root / split / cls
            images = [p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]

            print(f"\n[{split}/{cls}] Processing {len(images)} images...")

            for img_path in images:
                out_path = dst_dir / img_path.name
                total_images += 1
                cropped_ok, best_conf = crop_image(detector, img_path, out_path, args.conf_thresh)
                if cropped_ok:
                    total_cropped += 1
                else:
                    total_fallback += 1
                    log_rows.append({
                        "split": split,
                        "class": cls,
                        "filename": img_path.name,
                        "best_confidence_found": f"{best_conf:.3f}" if best_conf is not None else "none",
                        "full_path": str(img_path),
                    })

            print(f"[{split}/{cls}] Done.")

    # write fallback log
    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "class", "filename", "best_confidence_found", "full_path"])
            writer.writeheader()
            writer.writerows(log_rows)

    print("\n===== SUMMARY =====")
    print(f"Total images processed : {total_images}")
    print(f"Successfully cropped   : {total_cropped}")
    print(f"Fallback (copied as-is): {total_fallback}  <- see fallback_log.csv for details")
    print(f"Output saved to        : {output_root}")
    if log_rows:
        print(f"Fallback log saved to  : {log_path}")

        # quick breakdown: how many had NO detection at all vs below-threshold detection
        none_count = sum(1 for r in log_rows if r["best_confidence_found"] == "none")
        below_thresh_count = len(log_rows) - none_count
        print(f"  - No dog box detected at all : {none_count}")
        print(f"  - Below-threshold detection  : {below_thresh_count}  <- lowering --conf_thresh may help these")


if __name__ == "__main__":
    main()