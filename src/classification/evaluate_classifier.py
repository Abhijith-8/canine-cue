"""
evaluate_model.py

Generates a confusion matrix image + evaluation metrics report
(accuracy, precision, recall, F1, ROC-AUC) for a binary classifier
checkpoint, ready to drop straight into a PPT slide.

Usage:
    python evaluate_model.py \
        --model_path resnet18_calm_aggressive_best.pt \
        --data_dir data/test \
        --output_dir results \
        --class_names calm aggressive

Expects `data_dir` in ImageFolder format:
    data_dir/
        calm/
        aggressive/

Outputs (written to output_dir):
    confusion_matrix.png   -> heatmap, paste directly into a slide
    metrics_bar.png        -> bar chart of accuracy/precision/recall/F1
    metrics_report.json    -> raw numbers if you want to build your own table
    metrics_report.txt     -> sklearn classification_report, human readable
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_args():
    p = argparse.ArgumentParser(description="Evaluate a binary classifier checkpoint")
    p.add_argument("--model_path", required=True, help="Path to .pt checkpoint")
    p.add_argument("--data_dir", required=True, help="ImageFolder-style test dir")
    p.add_argument("--output_dir", default="eval_results", help="Where to save outputs")
    p.add_argument("--class_names", nargs=2, default=["calm", "aggressive"],
                    help="Two class names in the CANONICAL index order used at training "
                         "time, e.g. 'calm aggressive' means calm=0, aggressive=1")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(model_path, device):
    """Loads a full model object or a state_dict checkpoint.
    Adjust the architecture line if your checkpoint is a state_dict only."""
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    else:
        from torchvision.models import resnet18
        model = resnet18(num_classes=1)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_loader(data_dir, img_size, batch_size, canonical_mapping=None):
    """canonical_mapping: dict like {'calm': 0, 'aggressive': 1} matching
    whatever mapping train_classifier.py enforced. Do NOT rely on
    ImageFolder's alphabetical default — that's the inversion bug from v1."""
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=tfm)
    print(f"ImageFolder default mapping: {dataset.class_to_idx}")

    if canonical_mapping is not None:
        if dataset.class_to_idx != canonical_mapping:
            print(f"Overriding to canonical mapping: {canonical_mapping}")
            # remap targets/samples from ImageFolder's order to canonical order
            old_idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
            dataset.samples = [
                (path, canonical_mapping[old_idx_to_class[old_label]])
                for path, old_label in dataset.samples
            ]
            dataset.targets = [s[1] for s in dataset.samples]
            dataset.class_to_idx = canonical_mapping

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    classes_in_idx_order = [c for c, _ in sorted(dataset.class_to_idx.items(), key=lambda x: x[1])]
    return loader, classes_in_idx_order


def run_inference(model, loader, device):
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, annot_kws={"size": 14})
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metrics_bar(metrics_dict, output_path):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color="#4C72B0")
    plt.ylim(0, 1)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                  ha="center", fontsize=11)
    plt.title("Evaluation Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    canonical_mapping = {args.class_names[0]: 0, args.class_names[1]: 1}
    model = load_model(args.model_path, args.device)
    loader, class_names = get_loader(args.data_dir, args.img_size, args.batch_size,
                                      canonical_mapping=canonical_mapping)
    print(f"Using class order (index 0, 1): {class_names}")

    y_true, y_prob = run_inference(model, loader, args.device)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # only one class present in y_true

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
    }

    plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, "confusion_matrix.png"))
    plot_metrics_bar(
        {k: v for k, v in metrics.items() if k != "roc_auc" and v is not None},
        os.path.join(args.output_dir, "metrics_bar.png"),
    )

    report_txt = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(os.path.join(args.output_dir, "metrics_report.txt"), "w") as f:
        f.write(report_txt)
    with open(os.path.join(args.output_dir, "metrics_report.json"), "w") as f:
        json.dump({"metrics": metrics, "confusion_matrix": cm.tolist(),
                    "class_order": class_names}, f, indent=2)

    print("\n=== Summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\nSaved confusion_matrix.png, metrics_bar.png, metrics_report.txt/json to {args.output_dir}")


if __name__ == "__main__":
    main()