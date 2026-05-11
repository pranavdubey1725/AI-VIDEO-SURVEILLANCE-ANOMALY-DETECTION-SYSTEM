import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
from collections import defaultdict
from config import RAW_DIR, UCF_CRIME_CATEGORIES

def explore_dataset():
    print("=" * 60)
    print("UCF-CRIME DATASET EXPLORATION")
    print("=" * 60)

    records = []

    for split in ["Train", "Test"]:
        split_dir = RAW_DIR / split
        if not split_dir.exists():
            print(f"[WARNING] {split} directory not found")
            continue

        for category in UCF_CRIME_CATEGORIES:
            cat_dir = split_dir / category
            if not cat_dir.exists():
                # try alternate name
                cat_dir = split_dir / "NormalVideos" if category == "Normal" else None
                if cat_dir is None or not cat_dir.exists():
                    continue

            frames = list(cat_dir.glob("*.png"))
            if not frames:
                continue

            # Group frames by video name
            # Filename format: VideoName_frameNumber.png
            # e.g. Fighting050_x264_6800.png → video = Fighting050_x264
            video_frames = defaultdict(list)
            for f in frames:
                # Split from the right on '_' to separate frame number
                parts = f.stem.rsplit("_", 1)
                if len(parts) == 2:
                    video_name, frame_num = parts[0], int(parts[1])
                    video_frames[video_name].append(frame_num)

            for video_name, frame_nums in video_frames.items():
                frame_nums.sort()
                records.append({
                    "split":       split,
                    "category":    category,
                    "video_name":  video_name,
                    "frame_count": len(frame_nums),
                    "min_frame":   min(frame_nums),
                    "max_frame":   max(frame_nums),
                    "label":       0 if category == "Normal" else 1
                })

    df = pd.DataFrame(records)

    if df.empty:
        print("No data found. Check RAW_DIR path in config.py")
        return df

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[1] SPLIT OVERVIEW")
    print(df.groupby(["split", "label"])[["video_name"]].count().rename(
        columns={"video_name": "num_videos"}))

    print("\n[2] VIDEOS PER CATEGORY")
    cat_summary = df.groupby(["split", "category"]).agg(
        videos=("video_name", "count"),
        total_frames=("frame_count", "sum"),
        avg_frames=("frame_count", "mean")
    ).round(1)
    print(cat_summary.to_string())

    print("\n[3] FRAME COUNT STATS (per video)")
    print(df["frame_count"].describe().round(1))

    print("\n[4] SAMPLE VIDEOS")
    print(df.sample(min(5, len(df)))[
        ["split", "category", "video_name", "frame_count"]
    ].to_string(index=False))

    print("\n[5] TOTAL FRAMES IN DATASET")
    print(f"  Train: {df[df.split=='Train']['frame_count'].sum():,}")
    print(f"  Test:  {df[df.split=='Test']['frame_count'].sum():,}")
    print(f"  Total: {df['frame_count'].sum():,}")

    # Save for reuse in later scripts
    out_path = Path(__file__).parent.parent.parent / "data" / "splits" / "dataset_index.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] Dataset index → {out_path}")

    return df


if __name__ == "__main__":
    df = explore_dataset()
