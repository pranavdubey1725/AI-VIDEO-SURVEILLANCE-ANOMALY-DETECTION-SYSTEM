"""
Builds train_aug.csv by appending augmented video entries to train.csv.

For every training video whose _aug.npy file exists, this script adds
a duplicate set of clip rows pointing to the augmented features.

Val and test CSVs are NOT touched — evaluation always uses clean features.

Output: data/splits/train_aug.csv

Usage:
    python src/dataset/build_splits_augmented.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
from config import SPLITS_DIR, FEATURES_DIR


def build_augmented_train_csv():
    train_csv = SPLITS_DIR / "train.csv"
    out_csv   = SPLITS_DIR / "train_aug.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")

    df = pd.read_csv(train_csv)

    aug_rows = []
    missing  = 0

    for _, row in df.iterrows():
        category   = row["category"]
        video_name = row["video_name"]
        split_name = row["split"]   # always "train"

        # Check if the _aug.npy file exists for this video
        aug_feat = FEATURES_DIR / split_name / category / f"{video_name}_aug.npy"

        if not aug_feat.exists():
            missing += 1
            continue

        # Clone the row, change video_name to _aug variant
        aug_row = row.copy()
        aug_row["video_name"] = f"{video_name}_aug"
        aug_rows.append(aug_row)

    print(f"Original train clips : {len(df):,}")
    print(f"Augmented clips added: {len(aug_rows):,}  (from {len(aug_rows) // max(1, (len(df) // len(df.groupby('video_name'))))} videos)")
    print(f"Videos without _aug  : {missing:,}  (not yet extracted — run extract_features_augmented.py first)")

    if len(aug_rows) == 0:
        print("\nNo augmented features found. Run extract_features_augmented.py first.")
        return

    aug_df   = pd.DataFrame(aug_rows)
    aug_videos = aug_df["video_name"].str.replace("_aug$", "", regex=True).nunique()
    combined = pd.concat([df, aug_df], ignore_index=True)
    combined.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_csv}")
    print(f"Augmented videos found      : {aug_videos:,}")
    print(f"Total clips in train_aug.csv: {len(combined):,}")

    # Label distribution
    label_counts = combined["label"].value_counts()
    print(f"Normal  clips: {label_counts.get(0, 0):,}")
    print(f"Anomaly clips: {label_counts.get(1, 0):,}")


if __name__ == "__main__":
    build_augmented_train_csv()
