"""
Surveillance-specific image augmentation pipeline.

These augmentations simulate real-world CCTV degradation that UCF-Crime
doesn't capture well: low light, sensor noise, lens distortion, codec artifacts,
and night-vision/IR cameras.

Applied BEFORE ResNet50 feature extraction — pixel-space operations only.
NOT applied at val/test time (evaluation uses clean features only).

albumentations 2.0.x API is used throughout (parameter names differ from 1.x).
"""

import numpy as np
from PIL import Image
import albumentations as A


# ── Augmentation pipeline ──────────────────────────────────────────────────────
# Each transform has an individual probability — so every frame gets a different
# random combination of degradations, not the same fixed set every time.
#
# Probabilities are deliberately moderate (0.3-0.5):
#   too high → every frame is destroyed, features become uninformative
#   too low  → almost never fires, no real benefit

SURVEILLANCE_AUG = A.Compose([

    # --- Sensor noise (CCTV cameras pick up random electrical noise) ---
    # std_range is in [0,1] normalized range for uint8 images
    # (0.03, 0.12) ≈ 7-30 pixel-value noise on 0-255 scale — realistic CCTV grain
    A.GaussNoise(std_range=(0.03, 0.12), p=0.45),

    # --- Motion blur (camera shake, fast movement, pan) ---
    # blur_limit=9 means max kernel 9x9 — noticeable but not extreme
    A.MotionBlur(blur_limit=9, p=0.40),

    # --- Low-light / darkness (night corridors, poorly lit areas) ---
    # brightness_limit biased towards negative (darker), slight contrast variation
    A.RandomBrightnessContrast(
        brightness_limit=(-0.45, 0.05),
        contrast_limit=(-0.25, 0.15),
        p=0.50,
    ),

    # --- Heavy JPEG compression (low-bitrate CCTV encoders) ---
    # quality_range=(10, 50): 10 is very heavy, 50 is moderate compression
    # Default (99, 100) is essentially lossless — we want the blocky artifacts
    A.ImageCompression(quality_range=(10, 50), p=0.40),

    # --- Lens/fisheye distortion ---
    # distort_limit=(-0.35, 0.35): negative = barrel (fisheye), positive = pincushion
    # mode='fisheye' applies radial distortion matching a wide-angle lens
    A.OpticalDistortion(
        distort_limit=(-0.35, 0.35),
        mode="fisheye",
        p=0.30,
    ),

    # --- Night-vision / IR simulation ---
    # Real IR cameras produce grayscale output; converted back to 3-channel RGB
    # (R=G=B) so ResNet50 still receives a 3-channel tensor
    A.ToGray(p=0.20),

])


def augment_pil(pil_image: Image.Image) -> Image.Image:
    """
    Apply surveillance augmentation to a PIL Image, return a PIL Image.

    The augmentation is random — each call may apply a different combination
    of transforms based on each transform's individual probability.

    Args:
        pil_image: RGB PIL Image

    Returns:
        Augmented RGB PIL Image (same size, dtype uint8)
    """
    img_np = np.array(pil_image)          # HWC uint8, [0, 255]
    result = SURVEILLANCE_AUG(image=img_np)
    aug_np = result["image"]               # HWC uint8, [0, 255]
    return Image.fromarray(aug_np)


if __name__ == "__main__":
    # ── Visual verification ────────────────────────────────────────────────────
    # Run this directly to see augmented samples:
    #   python src/dataset/augmentations.py
    #
    # Saves a comparison grid: outputs/augmentation_samples.png
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from config import OUTPUTS_DIR, SPLITS_DIR
    import pandas as pd
    import random

    print("Loading a sample frame from train split...")
    csv_path = SPLITS_DIR / "train.csv"
    df = pd.read_csv(csv_path)

    # Pick one frame from a Fighting clip (worst category)
    fighting_rows = df[df["category"] == "Fighting"]
    if len(fighting_rows) == 0:
        fighting_rows = df
    sample_row = fighting_rows.iloc[0]
    frame_path = sample_row["frame_paths"].split(";")[0]

    try:
        original = Image.open(frame_path).convert("RGB")
    except FileNotFoundError:
        print(f"Frame not found: {frame_path}")
        print("Generating synthetic test frame instead.")
        # Create a synthetic frame for testing when raw data isn't available
        synthetic = np.random.randint(50, 180, (224, 224, 3), dtype=np.uint8)
        original = Image.fromarray(synthetic)

    print(f"Original frame: {frame_path}")
    print("Generating 8 augmented variants...")

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original", fontsize=10)
    axes[0, 0].axis("off")

    for i in range(8):
        row, col = divmod(i + 1, 3)
        aug = augment_pil(original)
        axes[row, col].imshow(aug)
        axes[row, col].set_title(f"Augmented #{i+1}", fontsize=10)
        axes[row, col].axis("off")

    plt.suptitle("Surveillance Augmentation Samples\n"
                 "(noise / blur / dark / compression / fisheye / IR)", fontsize=12)
    plt.tight_layout()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "augmentation_samples.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    print("Open outputs/augmentation_samples.png to verify the augmentations look correct.")
