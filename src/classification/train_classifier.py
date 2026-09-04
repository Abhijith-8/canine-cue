"""
CanineCue - Binary classifier training (calm vs aggressive)
ResNet18 fine-tuning: frozen backbone phase -> unfrozen fine-tune phase
Location: src/classification/train_classifier.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ---------------------------------------------------------------------------
# Make src/ importable if this script needs sibling modules later
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Config
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed_cropped"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "resnet18_calm_aggressive_best.pt"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

FROZEN_EPOCHS = 8          # phase 1: only classifier head trains
UNFROZEN_EPOCHS = 15       # phase 2: later layers unfrozen, lower LR
FROZEN_LR = 1e-3
UNFROZEN_LR = 1e-5
UNFREEZE_FROM_LAYER = "layer3"  # unfreeze layer3, layer4, and fc onward

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# aggressive = class 1 (positive class for BCEWithLogitsLoss)
POSITIVE_CLASS = "aggressive"

# ---------------------------------------------------------------------------
# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


CANONICAL_MAPPING = {"calm": 0, "aggressive": 1}


def remap_dataset_labels(ds, canonical_mapping):
    """
    Force dataset labels to a fixed canonical mapping, regardless of the
    alphabetical order ImageFolder assigned. Mutates ds.targets and ds.samples
    in place so DataLoader iteration reflects the corrected labels.
    """
    old_to_new = {
        old_idx: canonical_mapping[class_name]
        for class_name, old_idx in ds.class_to_idx.items()
    }

    ds.samples = [(path, old_to_new[old_label]) for path, old_label in ds.samples]
    ds.targets = [old_to_new[old_label] for old_label in ds.targets]
    ds.class_to_idx = canonical_mapping
    ds.classes = sorted(canonical_mapping, key=canonical_mapping.get)

    return ds


def build_dataloaders():
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)

    print("Raw class mapping (train):", train_ds.class_to_idx)

    train_ds = remap_dataset_labels(train_ds, CANONICAL_MAPPING)
    val_ds = remap_dataset_labels(val_ds, CANONICAL_MAPPING)
    test_ds = remap_dataset_labels(test_ds, CANONICAL_MAPPING)

    print("Canonical class mapping (enforced):", train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    return train_ds, train_loader, val_loader, test_loader


def compute_pos_weight(train_ds):
    """pos_weight = n_negative / n_positive, for BCEWithLogitsLoss class imbalance."""
    targets = torch.tensor(train_ds.targets)
    n_pos = (targets == 1).sum().item()
    n_neg = (targets == 0).sum().item()
    pos_weight = n_neg / n_pos
    print(f"Train counts -> calm(0): {n_neg}, aggressive(1): {n_pos}, "
          f"pos_weight: {pos_weight:.4f}")
    return torch.tensor([pos_weight], dtype=torch.float32)


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace final FC for binary logit output (single unit, used with BCEWithLogitsLoss)
    model.fc = nn.Linear(model.fc.in_features, 1)

    # Freeze everything except fc initially
    for name, param in model.named_parameters():
        param.requires_grad = "fc" in name

    return model


def unfreeze_layers(model, from_layer=UNFREEZE_FROM_LAYER):
    """Unfreeze from_layer onward (and everything after it) plus fc."""
    unfreeze = False
    for name, param in model.named_parameters():
        if from_layer in name:
            unfreeze = True
        if unfreeze or "fc" in name:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Unfrozen from '{from_layer}' onward. Trainable params: {trainable:,} / {total:,}")


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        all_preds.extend(preds.cpu().numpy().flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1], zero_division=0
    )

    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "calm_precision": precision[0], "calm_recall": recall[0], "calm_f1": f1[0],
        "aggressive_precision": precision[1], "aggressive_recall": recall[1], "aggressive_f1": f1[1],
    }
    return metrics


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def print_metrics(tag, epoch, metrics):
    print(
        f"[{tag}] Epoch {epoch:02d} | "
        f"loss {metrics['loss']:.4f} | acc {metrics['accuracy']:.4f} | "
        f"aggr P/R/F1 {metrics['aggressive_precision']:.3f}/"
        f"{metrics['aggressive_recall']:.3f}/{metrics['aggressive_f1']:.3f} | "
        f"calm P/R/F1 {metrics['calm_precision']:.3f}/"
        f"{metrics['calm_recall']:.3f}/{metrics['calm_f1']:.3f}"
    )


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds, train_loader, val_loader, test_loader = build_dataloaders()
    pos_weight = compute_pos_weight(train_ds).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = build_model().to(DEVICE)

    best_val_f1 = -1.0
    global_epoch = 0

    # ---------------- Phase 1: frozen backbone ----------------
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FROZEN_LR)
    print("\n=== Phase 1: frozen backbone ===")
    for epoch in range(1, FROZEN_EPOCHS + 1):
        global_epoch += 1
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion)
        print(f"train_loss {train_loss:.4f}  ({time.time()-t0:.1f}s)")
        print_metrics("VAL", global_epoch, val_metrics)

        if val_metrics["aggressive_f1"] > best_val_f1:
            best_val_f1 = val_metrics["aggressive_f1"]
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> new best (aggressive F1 {best_val_f1:.4f}), checkpoint saved")

    # ---------------- Phase 2: unfrozen fine-tune ----------------
    unfreeze_layers(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=UNFROZEN_LR)
    print("\n=== Phase 2: fine-tuning unfrozen layers ===")
    for epoch in range(1, UNFROZEN_EPOCHS + 1):
        global_epoch += 1
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion)
        print(f"train_loss {train_loss:.4f}  ({time.time()-t0:.1f}s)")
        print_metrics("VAL", global_epoch, val_metrics)

        if val_metrics["aggressive_f1"] > best_val_f1:
            best_val_f1 = val_metrics["aggressive_f1"]
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> new best (aggressive F1 {best_val_f1:.4f}), checkpoint saved")

    # ---------------- Final test evaluation ----------------
    print("\n=== Loading best checkpoint for test evaluation ===")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, criterion)
    print_metrics("TEST", global_epoch, test_metrics)
    print(f"\nBest model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()