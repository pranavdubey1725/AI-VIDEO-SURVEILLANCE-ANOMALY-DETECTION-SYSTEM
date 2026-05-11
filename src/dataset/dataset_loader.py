import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from config import (
    SPLITS_DIR, CLIP_LENGTH, FRAME_SIZE,
    BATCH_SIZE, DEVICE
)


# ── Transforms ────────────────────────────────────────────────────────────────
# Why normalize with ImageNet mean/std?
# Our ResNet50 was pre-trained on ImageNet. Its weights expect inputs
# normalized the same way ImageNet was preprocessed. If we feed it
# differently-normalized images, the feature representations will be wrong.

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),    # augmentation: mirror the scene
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # lighting variation
    transforms.ToTensor(),                      # [H,W,C] uint8 → [C,H,W] float32 in [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],             # ImageNet mean (R, G, B)
        std=[0.229, 0.224, 0.225]               # ImageNet std  (R, G, B)
    ),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class UCFCrimeDataset(Dataset):
    """
    PyTorch Dataset for UCF-Crime clip classification.

    Each sample is:
        frames : Tensor [CLIP_LENGTH, 3, H, W]  — the 16 frames as a stack
        label  : int  — 0 (normal) or 1 (anomalous)

    The Dataset reads one row from the CSV per __getitem__ call.
    PyTorch's DataLoader calls __getitem__ repeatedly to build batches.
    """

    def __init__(self, split: str, transform=None):
        """
        Args:
            split     : "train", "val", or "test"
            transform : torchvision transforms to apply to each frame
        """
        csv_path = SPLITS_DIR / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {csv_path}\n"
                "Run src/dataset/build_splits.py first."
            )

        self.df        = pd.read_csv(csv_path)
        self.transform = transform or EVAL_TRANSFORMS
        self.split     = split

        # Pre-parse frame paths from semicolon-separated string → list
        # We do this once here rather than on every __getitem__ call
        self.frame_path_lists = [
            row.split(";") for row in self.df["frame_paths"]
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Load one clip: read CLIP_LENGTH frames, apply transforms, return tensor.

        Returns:
            frames : FloatTensor [CLIP_LENGTH, 3, H, W]
            label  : int (0 or 1)
        """
        row         = self.df.iloc[idx]
        label       = int(row["label"])
        frame_paths = self.frame_path_lists[idx]

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")   # ensure 3-channel (no RGBA/grayscale)
            img = self.transform(img)              # → Tensor [3, H, W]
            frames.append(img)

        # Stack list of [3,H,W] tensors → [CLIP_LENGTH, 3, H, W]
        frames_tensor = torch.stack(frames, dim=0)

        return frames_tensor, label

    def get_sample_weights(self):
        """
        Returns per-sample weights for WeightedRandomSampler.

        Why we need this:
            76% of training clips are Normal, 24% are Anomaly.
            Without balancing, the model will learn to always predict Normal
            and achieve 76% accuracy without learning anything useful.

        How it works:
            Anomaly clips get weight = 1/count_anomaly
            Normal clips  get weight = 1/count_normal
            → Both classes contribute equally to each batch on average.
        """
        labels       = self.df["label"].values
        class_counts = np.bincount(labels)          # [count_normal, count_anomaly]
        class_weights = 1.0 / class_counts          # [w_normal, w_anomaly]
        sample_weights = class_weights[labels]      # assign weight per sample
        return torch.FloatTensor(sample_weights)


def get_dataloader(split: str, batch_size: int = BATCH_SIZE,
                   num_workers: int = 2) -> DataLoader:
    """
    Creates a DataLoader for the given split.

    Train: uses WeightedRandomSampler to balance normal/anomaly
    Val/Test: sequential, no sampling (we want true distribution for eval)
    """
    is_train  = (split == "train")
    transform = TRAIN_TRANSFORMS if is_train else EVAL_TRANSFORMS
    dataset   = UCFCrimeDataset(split=split, transform=transform)

    if is_train:
        sampler = WeightedRandomSampler(
            weights     = dataset.get_sample_weights(),
            num_samples = len(dataset),
            replacement = True
        )
        loader = DataLoader(
            dataset,
            batch_size  = batch_size,
            sampler     = sampler,       # sampler replaces shuffle
            num_workers = num_workers,
            pin_memory  = True,          # faster GPU transfer
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
        )

    return loader


if __name__ == "__main__":
    # ── Quick verification ─────────────────────────────────────────────────────
    print("Loading train DataLoader...")
    train_loader = get_dataloader("train", batch_size=4)

    frames, labels = next(iter(train_loader))

    print(f"\nBatch shape  : {frames.shape}")
    print(f"Expected     : [4, {CLIP_LENGTH}, 3, {FRAME_SIZE[0]}, {FRAME_SIZE[1]}]")
    print(f"Labels       : {labels.tolist()}")
    print(f"Pixel range  : [{frames.min():.2f}, {frames.max():.2f}]")
    print(f"Dtype        : {frames.dtype}")

    print(f"\nDataset sizes:")
    for split in ["train", "val", "test"]:
        ds = UCFCrimeDataset(split)
        print(f"  {split:<6}: {len(ds):>7,} clips")

    print("\nDataLoader OK")
