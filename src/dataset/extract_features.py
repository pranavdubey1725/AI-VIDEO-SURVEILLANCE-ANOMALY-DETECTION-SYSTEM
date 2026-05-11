"""
Extracts ResNet50 features for every frame in the dataset and saves to disk.

Output structure:
    data/features/{split}/{category}/{video_name}.npy
    shape: [num_frames, 2048]  — one 2048-dim vector per frame

Why save per-video?
    Clips from the same video share frames (due to sliding window overlap).
    Saving per-video means shared frames are computed only once.
    The LSTM Dataset later indexes into these files to build clips on the fly.

Estimated runtime: ~25-35 minutes on RTX 4060 for the full dataset.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm

from config import (
    RAW_DIR, SPLITS_DIR, FEATURES_DIR,
    FRAME_SIZE, DEVICE, UCF_CRIME_CATEGORIES
)
from src.models.feature_extractor import get_extractor

# Same normalization as training — must match what ResNet50 expects
TRANSFORM = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

BATCH_SIZE  = 64    # frames per GPU batch — adjust down if CUDA OOM
MAX_FRAMES  = 1000  # cap per video — normal videos can have 97k frames which is wasteful
                    # 1000 frames → ~123 clips, more than enough for training


def get_video_frame_map(csv_path: Path) -> dict:
    """
    Reads a split CSV and builds a map:
        {(split, category, video_name): [(frame_number, frame_path), ...]}

    We parse all clips but deduplicate at the frame level — because overlapping
    clips share frames. We only extract each frame's features once.
    """
    df = pd.read_csv(csv_path)
    video_map = defaultdict(dict)  # {(split, category, video): {frame_num: path}}

    for _, row in df.iterrows():
        key = (row["split"], row["category"], row["video_name"])
        paths = row["frame_paths"].split(";")

        for fp in paths:
            p = Path(fp)
            # Frame number is the last part of the stem after final underscore
            frame_num = int(p.stem.rsplit("_", 1)[1])
            video_map[key][frame_num] = fp

    # Convert inner dict to sorted list of (frame_num, path)
    result = {}
    for key, frame_dict in video_map.items():
        result[key] = sorted(frame_dict.items(), key=lambda x: x[0])

    return result


def extract_and_save(video_map: dict, extractor, device: str):
    """
    For each video: load all frames → run through ResNet50 in batches → save .npy
    """
    total_videos = len(video_map)

    for (split, category, video_name), frame_list in tqdm(
        video_map.items(), total=total_videos, desc="Extracting"
    ):
        # Output path
        out_dir  = FEATURES_DIR / split / category
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_name}.npy"

        if out_path.exists():
            continue   # skip already-extracted videos (resumable)

        # Cap to MAX_FRAMES — evenly sampled across the video
        if len(frame_list) > MAX_FRAMES:
            indices    = np.linspace(0, len(frame_list) - 1, MAX_FRAMES, dtype=int)
            frame_list = [frame_list[i] for i in indices]

        frame_nums  = [fn for fn, _ in frame_list]
        frame_paths = [fp for _, fp in frame_list]

        all_features = []

        # Process frames in batches
        for i in range(0, len(frame_paths), BATCH_SIZE):
            batch_paths = frame_paths[i : i + BATCH_SIZE]

            imgs = []
            for fp in batch_paths:
                img = Image.open(fp).convert("RGB")
                img = TRANSFORM(img)
                imgs.append(img)

            batch_tensor = torch.stack(imgs).to(device)  # [B, 3, 224, 224]

            with torch.no_grad():
                features = extractor(batch_tensor)        # [B, 2048]

            all_features.append(features.cpu().numpy())

        # Stack all batches → [num_frames, 2048]
        video_features = np.concatenate(all_features, axis=0)

        # Save: npy file with shape [num_frames, 2048]
        np.save(out_path, video_features)


def main():
    print("=" * 55)
    print("ResNet50 FEATURE EXTRACTION")
    print("=" * 55)
    print(f"Device    : {DEVICE}")
    print(f"Batch size: {BATCH_SIZE} frames")
    print(f"Output    : {FEATURES_DIR}")
    print()

    extractor = get_extractor(DEVICE)

    # Collect all videos across all splits
    all_video_maps = {}
    for split_file in ["train.csv", "val.csv", "test.csv"]:
        csv_path = SPLITS_DIR / split_file
        if csv_path.exists():
            video_map = get_video_frame_map(csv_path)
            all_video_maps.update(video_map)

    print(f"Total unique videos to process: {len(all_video_maps):,}")
    print("Starting extraction (skips already-done videos)...\n")

    extract_and_save(all_video_maps, extractor, DEVICE)

    # Verify output
    npy_files = list(FEATURES_DIR.rglob("*.npy"))
    print(f"\nExtraction complete.")
    print(f"Feature files saved: {len(npy_files):,}")
    print(f"Location: {FEATURES_DIR}")

    # Show one sample
    if npy_files:
        sample = np.load(npy_files[0])
        print(f"\nSample file : {npy_files[0].name}")
        print(f"Shape       : {sample.shape}  ← [num_frames, 2048]")


if __name__ == "__main__":
    main()
