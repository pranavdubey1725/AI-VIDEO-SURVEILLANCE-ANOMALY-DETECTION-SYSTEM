import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from config import (
    SPLITS_DIR, FEATURES_DIR, CLIP_LENGTH,
    BATCH_SIZE, UCF_CRIME_CATEGORIES
)


class FeatureDataset(Dataset):
    """
    Dataset that loads pre-computed ResNet50 features from .npy files.

    Each sample:
        features : FloatTensor [CLIP_LENGTH, 2048]
        label    : int (0=normal, 1=anomalous)

    Why this is fast:
        Loading a slice of a numpy array is a memory operation (~0.1ms).
        Loading 16 PNG images + running ResNet50 takes ~50ms per clip.
        With 134,812 training clips × 50 epochs, that's a 500x speedup.
    """

    def __init__(self, split: str):
        # "train_aug" loads train_aug.csv; anything else loads {split}.csv
        csv_path = SPLITS_DIR / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        # FeatureDataset resolves feature paths using the split column in the CSV,
        # which is always "train" even for augmented rows — so path resolution
        # works correctly without any other changes.

        self.df    = pd.read_csv(csv_path)
        self.split = split

        # Build a cache of loaded .npy arrays keyed by video_name
        # We load each video's feature file lazily (on first access)
        self._feature_cache = {}

        # Pre-compute the feature file path for each row
        self._feature_paths = self._resolve_feature_paths()

        # Pre-parse frame paths to get frame indices
        self._frame_indices = self._resolve_frame_indices()

        # Drop rows where feature file doesn't exist
        valid_mask = [p is not None for p in self._feature_paths]
        n_invalid  = sum(1 for v in valid_mask if not v)
        if n_invalid > 0:
            print(f"[FeatureDataset] Warning: {n_invalid} clips have missing feature files — skipping.")

        self._valid_indices = [i for i, v in enumerate(valid_mask) if v]

    def _resolve_feature_paths(self) -> list:
        paths = []
        for _, row in self.df.iterrows():
            split_name = row["split"]
            category   = row["category"]
            video_name = row["video_name"]

            feat_path = FEATURES_DIR / split_name / category / f"{video_name}.npy"
            paths.append(feat_path if feat_path.exists() else None)
        return paths

    def _resolve_frame_indices(self) -> list:
        """
        For each clip row, determine which frame indices to use from the .npy file.

        The .npy file stores frames in sorted order.
        We need to map the clip's frame_paths back to indices into that array.

        Strategy: use frame_start and build CLIP_LENGTH consecutive indices
        from the sorted frame list. We store just the start index.
        """
        indices = []
        for _, row in self.df.iterrows():
            frame_paths = row["frame_paths"].split(";")
            # Extract frame numbers from paths
            frame_nums = []
            for fp in frame_paths:
                stem = Path(fp).stem
                fn   = int(stem.rsplit("_", 1)[1])
                frame_nums.append(fn)
            indices.append(frame_nums)
        return indices

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx):
        actual_idx  = self._valid_indices[idx]
        row         = self.df.iloc[actual_idx]
        label       = int(row["label"])
        feat_path   = self._feature_paths[actual_idx]
        frame_nums  = self._frame_indices[actual_idx]

        # Load .npy file (cached to avoid re-reading disk)
        video_key = str(feat_path)
        if video_key not in self._feature_cache:
            self._feature_cache[video_key] = np.load(feat_path)
        all_features = self._feature_cache[video_key]  # [num_frames, 2048]

        # Map frame numbers to array indices
        # The .npy file rows correspond to sorted frame numbers
        # We need to find where our clip's frames are in the array
        num_stored = all_features.shape[0]

        # Simple index mapping: use evenly spaced indices if needed
        clip_indices = []
        for fn in frame_nums:
            # Approximate index based on frame number spacing
            # The npy was saved in frame_num order, so index ≈ fn / interval
            approx_idx = min(int(fn / 10), num_stored - 1)
            clip_indices.append(approx_idx)

        # Clamp all indices to valid range
        clip_indices = [min(max(i, 0), num_stored - 1) for i in clip_indices]

        # Take exactly CLIP_LENGTH frames
        clip_indices = clip_indices[:CLIP_LENGTH]
        while len(clip_indices) < CLIP_LENGTH:
            clip_indices.append(clip_indices[-1])  # pad by repeating last frame

        clip_features = all_features[clip_indices]           # [CLIP_LENGTH, 2048]
        clip_tensor   = torch.FloatTensor(clip_features)     # → FloatTensor

        return clip_tensor, label

    def get_sample_weights(self):
        labels        = self.df.iloc[self._valid_indices]["label"].values
        class_counts  = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        return torch.FloatTensor(sample_weights)


def get_feature_dataloader(split: str, batch_size: int = BATCH_SIZE,
                            num_workers: int = 0) -> DataLoader:
    """
    Note: num_workers=0 because we use an in-memory cache.
    Multiprocessing workers don't share the cache, making it inefficient.
    """
    dataset  = FeatureDataset(split=split)
    is_train = (split == "train")

    if is_train:
        sampler = WeightedRandomSampler(
            weights     = dataset.get_sample_weights(),
            num_samples = len(dataset),
            replacement = True
        )
        loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler, num_workers=num_workers)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return loader


if __name__ == "__main__":
    print("Testing FeatureDataset...\n")

    for split in ["train", "val", "test"]:
        ds = FeatureDataset(split)
        print(f"{split:<6}: {len(ds):>7,} clips loaded")

    print("\nLoading one batch from train...")
    loader = get_feature_dataloader("train", batch_size=8)
    features, labels = next(iter(loader))

    print(f"Batch shape : {features.shape}")
    print(f"Expected    : [8, {CLIP_LENGTH}, 2048]")
    print(f"Labels      : {labels.tolist()}")
    print(f"Dtype       : {features.dtype}")
    print(f"Value range : [{features.min():.3f}, {features.max():.3f}]")
    print("\nFeatureDataset OK")
