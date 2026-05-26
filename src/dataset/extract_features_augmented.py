"""
Extracts ResNet50 features from AUGMENTED frames for a subset of training videos.

Runs ONLY on training videos — val and test always use clean features.
Saves augmented feature files as {video_name}_aug.npy alongside originals.

Default (small test): Fighting + Shooting + RoadAccidents + 200 NormalVideos
Full run: pass --all-categories to process every training category.

Usage:
    # Small test (~30-45 min):
    python src/dataset/extract_features_augmented.py

    # Full run (~2 hours):
    python src/dataset/extract_features_augmented.py --all-categories

    # Specific categories:
    python src/dataset/extract_features_augmented.py --categories Fighting Robbery
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm

from config import SPLITS_DIR, FEATURES_DIR, FRAME_SIZE, DEVICE
from src.models.feature_extractor import get_extractor
from src.dataset.augmentations import augment_pil

# Same normalization as original extraction — ResNet50 expects ImageNet stats
TRANSFORM = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

BATCH_SIZE = 64    # frames per GPU batch — same as original extraction
MAX_FRAMES = 1000  # same cap as original

# Default small-test categories (3 worst-performing anomaly + partial Normal)
DEFAULT_ANOMALY_CATEGORIES = ["Fighting", "Shooting", "RoadAccidents"]
NORMAL_CATEGORY             = "Normal"
NORMAL_LIMIT                = 200   # only augment this many Normal videos (out of 800)


def get_video_frame_map(csv_path: Path, categories: list, normal_limit: int) -> dict:
    """
    Build {(category, video_name): [(frame_num, frame_path), ...]} for the
    requested categories only (train split).

    Normal videos are capped at normal_limit to keep extraction time reasonable.
    """
    df = pd.read_csv(csv_path)
    # Only train split — CSV uses "Train" (capital T)
    df = df[df["split"] == "Train"]

    video_map = defaultdict(dict)  # {(category, video): {frame_num: path}}

    normal_videos_seen = set()

    for _, row in df.iterrows():
        cat        = row["category"]
        video_name = row["video_name"]

        if cat not in categories:
            continue

        # Cap Normal category
        if cat == NORMAL_CATEGORY:
            if len(normal_videos_seen) >= normal_limit and video_name not in normal_videos_seen:
                continue
            normal_videos_seen.add(video_name)

        key = (cat, video_name)
        for fp in row["frame_paths"].split(";"):
            p         = Path(fp)
            frame_num = int(p.stem.rsplit("_", 1)[1])
            video_map[key][frame_num] = fp

    # Convert inner dict to sorted list of (frame_num, path)
    result = {}
    for key, frame_dict in video_map.items():
        result[key] = sorted(frame_dict.items(), key=lambda x: x[0])

    return result


def extract_and_save_augmented(video_map: dict, extractor, device: str):
    """
    For each video: load frames → apply augmentation → ResNet50 → save _aug.npy
    """
    skipped = 0
    for (category, video_name), frame_list in tqdm(
        video_map.items(), total=len(video_map), desc="Augmented extraction"
    ):
        # Features directory uses "Train" (capital T) matching the CSV split column
        out_dir  = FEATURES_DIR / "Train" / category
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_name}_aug.npy"

        if out_path.exists():
            skipped += 1
            continue

        # Cap frames
        if len(frame_list) > MAX_FRAMES:
            indices    = np.linspace(0, len(frame_list) - 1, MAX_FRAMES, dtype=int)
            frame_list = [frame_list[i] for i in indices]

        frame_paths = [fp for _, fp in frame_list]
        all_features = []

        for i in range(0, len(frame_paths), BATCH_SIZE):
            batch_paths = frame_paths[i : i + BATCH_SIZE]
            imgs = []
            for fp in batch_paths:
                img = Image.open(fp).convert("RGB")
                img = augment_pil(img)   # ← augmentation applied here
                img = TRANSFORM(img)
                imgs.append(img)

            batch_tensor = torch.stack(imgs).to(device)

            with torch.no_grad():
                features = extractor(batch_tensor)   # [B, 2048]

            all_features.append(features.cpu().numpy())

        video_features = np.concatenate(all_features, axis=0)
        np.save(out_path, video_features)

    return skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-categories", action="store_true",
        help="Augment ALL training categories (full run, ~2 hours)"
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Specific anomaly categories to augment (overrides default)"
    )
    parser.add_argument(
        "--normal-limit", type=int, default=NORMAL_LIMIT,
        help=f"Max Normal videos to augment (default: {NORMAL_LIMIT})"
    )
    args = parser.parse_args()

    from config import UCF_CRIME_CATEGORIES

    if args.all_categories:
        anomaly_cats = [c for c in UCF_CRIME_CATEGORIES if c != "Normal"]
        normal_limit = 800   # all Normal videos
    elif args.categories:
        anomaly_cats = args.categories
        normal_limit = args.normal_limit
    else:
        anomaly_cats = DEFAULT_ANOMALY_CATEGORIES
        normal_limit = args.normal_limit

    categories = anomaly_cats + [NORMAL_CATEGORY]

    print("=" * 60)
    print("AUGMENTED FEATURE EXTRACTION (train split only)")
    print("=" * 60)
    print(f"Device     : {DEVICE}")
    print(f"Categories : {anomaly_cats}")
    print(f"Normal cap : {normal_limit} videos")
    print(f"Output     : {FEATURES_DIR}/Train/{{category}}/{{video}}_aug.npy")
    print()

    csv_path = SPLITS_DIR / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"train.csv not found at {csv_path}")

    video_map = get_video_frame_map(csv_path, categories, normal_limit)
    print(f"Videos to process: {len(video_map):,}")
    print()

    extractor = get_extractor(DEVICE)
    skipped   = extract_and_save_augmented(video_map, extractor, DEVICE)

    npy_files = list(FEATURES_DIR.rglob("*_aug.npy"))
    print(f"\nDone.")
    print(f"  New files created : {len(npy_files) - skipped}")
    print(f"  Already existed   : {skipped}")
    print(f"  Total _aug files  : {len(npy_files)}")

    if npy_files:
        sample = np.load(npy_files[0])
        print(f"\nSample: {npy_files[0].name} -> shape {sample.shape}")


if __name__ == "__main__":
    main()
