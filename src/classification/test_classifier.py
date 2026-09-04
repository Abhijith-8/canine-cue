"""
CanineCue - Standalone inference sanity-check for the trained classifier
Loads the best checkpoint and runs predictions on a folder of images.

Location: src/classification/test_classifier.py

Usage:
    python test_classifier.py --folder path/to/images
    python test_classifier.py --folder data/processed_cropped/test/aggressive
    python test_classifier.py --folder path/to/unseen_photos --threshold 0.6
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Config (must match train_classifier.py)
IMG_SIZE = 224
CANONICAL_MAPPING = {"calm": 0, "aggressive": 1}
IDX_TO_CLASS = {v: k for k, v in CANONICAL_MAPPING.items()}

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "resnet18_calm_aggressive_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_model(checkpoint_path):
    model = build_model()
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model, image_path, threshold=0.5):
    img = Image.open(image_path).convert("RGB")
    tensor = eval_transform(img).unsqueeze(0).to(DEVICE)

    logit = model(tensor)
    prob_aggressive = torch.sigmoid(logit).item()  # P(class == aggressive), since aggressive=1

    pred_idx = 1 if prob_aggressive >= threshold else 0
    pred_class = IDX_TO_CLASS[pred_idx]
    confidence = prob_aggressive if pred_idx == 1 else (1 - prob_aggressive)

    return pred_class, confidence, prob_aggressive


def main():
    parser = argparse.ArgumentParser(description="Run inference on a folder of images.")
    parser.add_argument("--folder", type=str, required=True,
                         help="Folder containing images to test.")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH),
                         help="Path to model checkpoint (.pt).")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Decision threshold on P(aggressive). Default 0.5.")
    args = parser.parse_args()

    folder = Path(args.folder)
    checkpoint_path = Path(args.checkpoint)

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    image_paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    )
    if not image_paths:
        print(f"No images found in {folder}")
        return

    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path)
    print(f"Found {len(image_paths)} image(s) in {folder}\n")

    print(f"{'Filename':<40} {'Prediction':<12} {'Confidence':<12} {'P(aggressive)'}")
    print("-" * 90)

    calm_count = 0
    aggressive_count = 0

    for img_path in image_paths:
        try:
            pred_class, confidence, prob_aggressive = predict_image(model, img_path, args.threshold)
        except Exception as e:
            print(f"{img_path.name:<40} ERROR: {e}")
            continue

        if pred_class == "calm":
            calm_count += 1
        else:
            aggressive_count += 1

        print(f"{img_path.name:<40} {pred_class:<12} {confidence:<12.3f} {prob_aggressive:.3f}")

    print("-" * 90)
    print(f"Summary: {aggressive_count} predicted aggressive, {calm_count} predicted calm "
          f"(out of {len(image_paths)})")


if __name__ == "__main__":
    main()