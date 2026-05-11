"""
Builds clip-level train/val/test CSV files from the pre-extracted frames.

What is a clip?
    A clip is a sequence of CLIP_LENGTH consecutive frames from one video.
    e.g. frames [0, 10, 20, ..., 150] from Fighting050_x264 → one clip.

Why clips instead of individual frames?
    Our LSTM needs to see a sequence of frames to understand motion and
    temporal patterns. A single frame can't tell you if someone is running
    or standing still.

Output CSVs (saved to data/splits/):
    train.csv  ← 80% of Train folder videos
    val.csv    ← 20% of Train folder videos
    test.csv   ← all Test folder videos (with frame-level labels)

Each row in the CSV represents one clip:
    video_name, category, label, split, frame_start, frame_paths (semicolon-separated)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from config import (
    RAW_DIR, SPLITS_DIR, UCF_CRIME_CATEGORIES,
    CLIP_LENGTH, CLIP_STRIDE, MIN_FRAMES
)


def get_frame_paths(category_dir: Path, category: str) -> dict:
    """
    Scans a category folder and groups PNG files by video name.
    Returns: {video_name: sorted list of (frame_number, Path) tuples}
    """
    video_frames = defaultdict(list)

    for f in category_dir.glob("*.png"):
        parts = f.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        video_name, frame_num = parts[0], int(parts[1])
        video_frames[video_name].append((frame_num, f))

    # Sort each video's frames by frame number
    for video_name in video_frames:
        video_frames[video_name].sort(key=lambda x: x[0])

    return video_frames


def build_clips_for_video(frame_list: list, video_name: str,
                           category: str, label: int, split: str) -> list:
    """
    Slides a window of CLIP_LENGTH frames across a video with CLIP_STRIDE step.
    Returns a list of clip records (dicts).

    Example with CLIP_LENGTH=16, CLIP_STRIDE=8:
        frames: [f0, f1, f2, ..., f99]
        clip 1: frames[0:16]
        clip 2: frames[8:24]
        clip 3: frames[16:32]  ... and so on
    """
    clips = []
    num_frames = len(frame_list)

    if num_frames < MIN_FRAMES:
        return clips  # skip videos too short for even one clip

    for start in range(0, num_frames - CLIP_LENGTH + 1, CLIP_STRIDE):
        end = start + CLIP_LENGTH
        clip_frames = frame_list[start:end]
        frame_numbers = [fn for fn, _ in clip_frames]
        frame_paths   = [str(p) for _, p in clip_frames]

        clips.append({
            "video_name":  video_name,
            "category":    category,
            "label":       label,
            "split":       split,
            "frame_start": frame_numbers[0],
            "frame_end":   frame_numbers[-1],
            "frame_paths": ";".join(frame_paths),
        })

    return clips


def build_splits():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    all_train_clips = []
    all_test_clips  = []

    print("Building clips from raw frames...")
    print(f"  CLIP_LENGTH={CLIP_LENGTH}, CLIP_STRIDE={CLIP_STRIDE}, MIN_FRAMES={MIN_FRAMES}\n")

    skipped_videos = 0

    for split_name in ["Train", "Test"]:
        split_dir = RAW_DIR / split_name

        for category in UCF_CRIME_CATEGORIES:
            # Handle the NormalVideos folder naming
            if category == "Normal":
                cat_dir = split_dir / "NormalVideos"
            else:
                cat_dir = split_dir / category

            if not cat_dir.exists():
                continue

            label = 0 if category == "Normal" else 1
            video_frames = get_frame_paths(cat_dir, category)

            for video_name, frame_list in video_frames.items():
                clips = build_clips_for_video(
                    frame_list, video_name, category, label, split_name
                )
                if not clips:
                    skipped_videos += 1
                    continue

                if split_name == "Train":
                    all_train_clips.extend(clips)
                else:
                    all_test_clips.extend(clips)

    # ── Train → Train + Val split ─────────────────────────────────────────────
    # We split at the VIDEO level, not clip level.
    # Why? If we split at clip level, clips from the same video could appear
    # in both train and val — that's data leakage. The model would effectively
    # "see" test videos during training.
    train_df = pd.DataFrame(all_train_clips)

    unique_videos = train_df[["video_name", "category", "label"]].drop_duplicates()
    train_videos, val_videos = train_test_split(
        unique_videos,
        test_size=0.2,
        stratify=unique_videos["label"],  # preserve normal/anomaly ratio
        random_state=42
    )

    train_video_names = set(train_videos["video_name"])
    val_video_names   = set(val_videos["video_name"])

    final_train = train_df[train_df["video_name"].isin(train_video_names)].copy()
    final_val   = train_df[train_df["video_name"].isin(val_video_names)].copy()
    final_test  = pd.DataFrame(all_test_clips)

    # Save
    final_train.to_csv(SPLITS_DIR / "train.csv", index=False)
    final_val.to_csv(SPLITS_DIR / "val.csv",   index=False)
    final_test.to_csv(SPLITS_DIR / "test.csv",  index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 55)
    print("SPLIT SUMMARY")
    print("=" * 55)
    print(f"{'Split':<10} {'Videos':>8} {'Clips':>10} {'Normal%':>10} {'Anomaly%':>10}")
    print("-" * 55)

    for name, df in [("Train", final_train), ("Val", final_val), ("Test", final_test)]:
        n_videos = df["video_name"].nunique()
        n_clips  = len(df)
        pct_norm = (df["label"] == 0).mean() * 100
        pct_anom = (df["label"] == 1).mean() * 100
        print(f"{name:<10} {n_videos:>8} {n_clips:>10,} {pct_norm:>9.1f}% {pct_anom:>9.1f}%")

    print("-" * 55)
    print(f"\nSkipped {skipped_videos} videos (fewer than {MIN_FRAMES} frames)")
    print(f"\nSaved CSVs to: {SPLITS_DIR}")
    print("  train.csv / val.csv / test.csv")


if __name__ == "__main__":
    build_splits()
