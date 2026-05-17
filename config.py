from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
FRAMES_DIR      = DATA_DIR / "frames"
SPLITS_DIR      = DATA_DIR / "splits"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
OUTPUTS_DIR     = ROOT_DIR / "outputs"
FEATURES_DIR    = DATA_DIR / "features"

# ── Dataset ────────────────────────────────────────────────────────────────────
UCF_CRIME_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "RoadAccidents",
    "Burglary", "Explosion", "Fighting", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism", "Normal"
]

TRAIN_SPLIT = 0.8
VAL_SPLIT   = 0.1
TEST_SPLIT  = 0.1

# ── Video / Frame settings ─────────────────────────────────────────────────────
# Frames are pre-extracted from videos every 10 original frames
FRAME_INTERVAL  = 10         # gap between consecutive extracted frames
FRAME_SIZE      = (224, 224) # height x width fed into ResNet50
CLIP_LENGTH     = 16         # number of frames per clip fed into LSTM
CLIP_STRIDE     = 8          # how many frames to slide between consecutive clips
MIN_FRAMES      = 16         # discard videos with fewer frames than this

# ── Model ──────────────────────────────────────────────────────────────────────
FEATURE_DIM     = 2048       # ResNet50 output dimension
LSTM_HIDDEN     = 256        # LSTM hidden state size
LSTM_LAYERS     = 2          # stacked LSTM layers
DROPOUT         = 0.5

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-4
NUM_EPOCHS      = 50
import os
DEVICE          = os.environ.get("DEVICE", "cuda")  # override with DEVICE=cpu for CPU-only

# ── Inference ──────────────────────────────────────────────────────────────────
ANOMALY_THRESHOLD = 0.5      # score above this → flagged as anomalous
