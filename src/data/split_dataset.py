"""
split_dataset.py
-----------------
Splits flat class folders (calm/, aggressive/) into
train/val/test subfolders with an 70/15/15 split, preserving
class balance in each split.

Expects input like:
    data/raw_labeled/calm/*.jpg
    data/raw_labeled/aggressive/*.jpg

Produces:
    data/processed/train/calm/
    data/processed/train/aggressive/
    data/processed/val/calm/
    data/processed/val/aggressive/
    data/processed/test/calm/
    data/processed/test/aggressive/

Usage:
    python split_dataset.py --input_dir data/raw_labeled --output_dir data/processed
"""

import argparse
import os
import random
import shutil


def split_dataset(input_dir, output_dir, train_ratio=0.70, val_ratio=0.15, seed=42):
    random.seed(seed)
    test_ratio = 1.0 - train_ratio - val_ratio

    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(f"Found classes: {classes}")

    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        images = [f for f in os.listdir(cls_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # remainder goes to test

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, file_list in splits.items():
            split_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            for fname in file_list:
                src = os.path.join(cls_dir, fname)
                dst = os.path.join(split_dir, fname)
                shutil.copy2(src, dst)

        print(f"  {cls}: {n_total} total -> "
              f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    print(f"\nDone. Split data saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Folder containing calm/ and aggressive/ subfolders")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    split_dataset(args.input_dir, args.output_dir, args.train_ratio, args.val_ratio)