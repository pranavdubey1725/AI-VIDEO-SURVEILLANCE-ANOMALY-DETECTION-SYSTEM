# Project History & Development Log
## AI Video Surveillance — Anomaly Detection System

> This document captures every decision, discovery, failure, design choice, and engineering reasoning made throughout the development of this project. It is a living document — updated at the end of every phase.

---

## Project Identity

| Field | Detail |
|---|---|
| Project | AI Video Surveillance — Anomaly Detection System |
| Purpose | Resume/portfolio project demonstrating end-to-end ML engineering |
| Developer | Pranav (VIT Bhopal) |
| Started | May 2026 |
| Stack | PyTorch, YOLOv8, OpenCV, FastAPI, Vanilla JS |
| Dataset | UCF-Crime (Kaggle: odins0n/ucf-crime-dataset) |

---

## The Problem Statement

Build a system that:
1. Accepts a surveillance video as input
2. Watches the footage frame by frame
3. Detects anomalous events (fighting, robbery, assault, etc.)
4. Explains WHY it flagged something
5. Returns timestamped results through a web interface

---

## Architecture Decision (Final)

```
Input Video
    │
    ▼
[Frame Extraction] ← OpenCV
    │
    ├──────────────────────────┐
    ▼                          ▼
[YOLOv8 Detection]     [ResNet50 Feature Extractor]
 "What's in frame"      "Spatial features per frame"
    │                          │
    │                    [LSTM Sequence Model]
    │                    "Is this clip anomalous?"
    │                          │
    └──────────┬───────────────┘
               ▼
     [Anomaly Score + Explanation Engine]
     "Fighting detected — confidence 91%"
               │
               ▼
     [FastAPI Backend] → [Streamlit Frontend]
```

**Why two streams?**
- YOLO alone detects objects but cannot judge behavior
- LSTM alone can flag anomalies but can't explain what it saw
- Combined: LSTM scores the clip, YOLO explains what's in it

---

## Key Architectural Decisions & Reasoning

### Why PyTorch over TensorFlow/Keras?
- YOLOv8 (Ultralytics) is built entirely in PyTorch — no workaround
- PyTorch is the dominant framework in research (most papers use it)
- User has prior PyTorch experience
- Decision: Use PyTorch throughout for consistency

### Why ResNet50 as feature extractor?
- Pre-trained on ImageNet (1.2M images) — rich visual understanding already baked in
- Transfer learning: borrow spatial feature knowledge instead of training from scratch
- 2048-dim output is a rich representation of each frame
- Well-documented, stable, and available directly in torchvision
- Alternative considered: EfficientNet (slightly better accuracy, more complex), rejected for simplicity

### Why LSTM over 3D CNN?
- 3D CNN processes space+time together in one pass (e.g. C3D, SlowFast)
- LSTM processes a sequence of frame features, one frame at a time
- For this project: LSTM is more interpretable and simpler to implement correctly
- 3D CNNs need significantly more memory and are harder to debug
- Trade-off accepted: LSTM is slightly weaker on purely motion-based tasks, but much easier to understand and explain in interviews

### Why UCF-Crime dataset?
- Real CCTV surveillance footage — not synthetic, not clean
- 13 real crime categories recognized by interviewers and researchers
- 1,900 videos, 128 hours — serious scale
- Widely cited in papers — gives interview credibility
- Alternatives rejected: UCSD (too simple), CUHK Avenue (too small)

### Why YOLOv8 for object detection?
- State of the art real-time object detection as of 2024-2025
- Maintained by Ultralytics, excellent documentation
- PyTorch-native — fits our stack
- Alternatives: YOLOv3 via OpenCV DNN (considered, rejected — YOLOv8 is strictly better)

---

## Tech Stack (Final)

| Component | Library | Version |
|---|---|---|
| ML Framework | PyTorch | 2.6.0+cu124 |
| Object Detection | Ultralytics YOLOv8 | 8.4.47 |
| Feature Extraction | torchvision ResNet50 | 0.21.0+cu124 |
| Video/Image Processing | OpenCV | 4.13.0 |
| Data Analysis | Pandas | 2.3.3 |
| Data Analysis | NumPy | 2.0.2 |
| Visualization | Seaborn + Matplotlib | 0.13.2 / 3.9.4 |
| ML Utilities | Scikit-learn | 1.6.1 |
| API Backend | FastAPI | 0.128.8 |
| API Server | Uvicorn | 0.39.0 |
| Frontend UI | Streamlit | 1.50.0 |

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8GB |
| CUDA Driver | 581.57 (supports up to CUDA 13.0) |
| PyTorch CUDA Build | cu124 (CUDA 12.4) |
| OS | Windows 11 |

**Why CUDA 12.4 build on a driver that supports CUDA 13.0?**
The `nvidia-smi` CUDA version shows the *maximum* CUDA the driver supports, not what is installed. PyTorch ships its own bundled CUDA runtime. We picked cu124 because it was the latest stable PyTorch build available and fully supported by the driver.

---

## Phase 0 — Environment Setup

**Date completed:** May 2026
**Status:** ✅ Complete

### What We Did
1. Verified GPU via `nvidia-smi` — RTX 4060 Laptop, 8GB VRAM, Driver 581.57
2. Created Python virtual environment (`python -m venv venv`)
3. Installed PyTorch 2.6.0 with CUDA 12.4:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
4. Installed all remaining dependencies:
   ```bash
   pip install ultralytics opencv-python pandas seaborn matplotlib scikit-learn fastapi uvicorn streamlit ipykernel
   ```
5. Created full project folder structure
6. Wrote `config.py` with all settings centralized
7. Verified environment with `gpu_check.py`

### Problem Encountered: Windows Long Path Error
- **What happened:** First install attempt failed with OSError on a JupyterLab file with an extremely long path
- **Root cause:** Windows has a 260-character path limit by default. Our project folder path is long, and JupyterLab installs deeply nested files that breach this limit.
- **Solution:** Skipped JupyterLab entirely. VS Code has built-in Jupyter notebook support via the Jupyter extension — same functionality, no path issue.
- **Lesson:** Always be aware of Windows path length limits when working with deeply nested packages.

### Project Folder Structure Created
```
surveillance-system/
├── config.py
├── gpu_check.py
├── data/
│   ├── raw/          ← dataset lives here
│   ├── frames/       ← (not used — frames pre-extracted in dataset)
│   └── splits/       ← train.csv / val.csv / test.csv
├── src/
│   ├── dataset/      ← data loading, preprocessing
│   ├── models/       ← ResNet50, LSTM, YOLOv8 wrappers
│   ├── training/     ← training loop, evaluation
│   ├── explainability/ ← Grad-CAM
│   └── inference/    ← end-to-end pipeline
├── api/              ← FastAPI app
├── ui/               ← Streamlit app
├── notebooks/        ← EDA
├── checkpoints/      ← saved model weights
└── outputs/          ← inference results
```

### gpu_check.py Output (Verified)
```
PyTorch:     2.6.0+cu124
CUDA:        Available
GPU:         NVIDIA GeForce RTX 4060 Laptop GPU
VRAM:        8.0 GB
GPU compute: OK — matrix multiply on GPU succeeded
OpenCV:      4.13.0
Ultralytics: 8.4.47
NumPy:       2.0.2
Pandas:      2.3.3
Scikit-learn:1.6.1
FastAPI:     0.128.8
Streamlit:   1.50.0
ALL CHECKS PASSED
```

---

## Phase 1 — Dataset

**Date completed:** May 2026
**Status:** ✅ Complete

### Dataset Selection Process
- Searched Kaggle for "UCF Crime"
- Selected `odins0n/ucf-crime-dataset` (11.8GB, 38,233 downloads, 0.875 usability)
- Rejected alternatives:
  - `alirakhmaev/ucf-crime-full` (26GB, low usability rating 0.3125)
  - `minhajuddinmeraj/anomalydetectiondatasetucf` (41GB, too large)
  - `shashiprakash204/ucfcrimeminidataset` (940MB, too small for production)

### Download Command
```bash
kaggle datasets download -d odins0n/ucf-crime-dataset -p data/raw --unzip
```

### Critical Discovery: Frames Pre-Extracted
**Original plan:** Download raw videos → write frame extraction script → extract at 5 FPS
**Reality:** The Kaggle dataset already contains pre-extracted PNG frames, not raw videos.

**Frame naming convention:**
```
{Category}{VideoNumber}_x264_{FrameNumber}.png
e.g. Fighting050_x264_6800.png
     → Video: Fighting050_x264
     → Frame number: 6800 (from original video)
```
Frames are sampled every 10 original frames (interval = 10).

**Impact:** Skipped frame extraction entirely. Adjusted `config.py` — replaced `FPS` with `FRAME_INTERVAL = 10`.

### Dataset Structure (After Download)
```
data/raw/
├── Train/
│   ├── Abuse/         (48 videos)
│   ├── Arrest/        (45 videos)
│   ├── Arson/         (41 videos)
│   ├── Assault/       (47 videos)
│   ├── Burglary/      (87 videos)
│   ├── Explosion/     (29 videos)
│   ├── Fighting/      (45 videos)
│   ├── NormalVideos/  (800 videos)
│   ├── RoadAccidents/ (127 videos)
│   ├── Robbery/       (145 videos)
│   ├── Shooting/      (27 videos)
│   ├── Shoplifting/   (29 videos)
│   ├── Stealing/      (95 videos)
│   └── Vandalism/     (45 videos)
└── Test/
    └── (same categories, fewer videos)
```

### Dataset Statistics (from explore.py)
```
Split   Normal    Anomaly   Total Videos
Train   800       810       1,610
Test    150       140         290
Total   950       950       1,900

Total frames: 1,377,653
  Train: 1,266,345
  Test:    111,308

Frame count per video:
  Min:    11 frames
  Median: 214 frames
  Max:    97,651 frames
  Std:    3,289 (high variance)
```

### Key Dataset Observations

**1. Category imbalance in anomaly videos:**
```
Robbery:       145 training videos  ← most
RoadAccidents: 127 training videos
Shooting:       27 training videos  ← least
Explosion:      29 training videos
```
Some categories have 5x more data than others. Handled via weighted sampling in training.

**2. Frame count variance is extreme:**
Min=11, Max=97,651 — videos shorter than 16 frames are discarded (3 videos skipped).

**3. Normal class dominates frames:**
Normal: 947,768 training frames (75%). Will cause model to predict "normal" for everything if not handled. Solution: weighted loss or balanced sampler.

**4. Weak supervision:**
Training data has only video-level labels ("this video contains fighting") — NOT frame-level labels. Frame-level labels only exist in Test set for evaluation. This is the standard UCF-Crime benchmark setup.

### Clip Building Strategy

**What is a clip?**
A sequence of 16 consecutive frames from one video, fed as a unit to the LSTM.

**Why 16 frames?**
A standard in video understanding literature. At our sampling rate (every 10 original frames at ~30fps = ~3fps effective), 16 frames ≈ 5 seconds of footage. Enough to capture a short anomalous event.

**Sliding window with stride:**
```
Frames: [f0, f1, f2, ... f99]
Clip 1: f0  → f15   (frames 0-15)
Clip 2: f8  → f23   (stride=8, 50% overlap)
Clip 3: f16 → f31
```
50% overlap ensures anomalies that start mid-clip are still captured.

**Why video-level split (not clip-level)?**
If we split by clip, clips from the same video appear in both train and val. The model "sees" validation videos during training — data leakage. Splits are always done at the video level.

### Split Results (from build_splits.py)
```
Split    Videos    Clips     Normal%   Anomaly%
Train    1,285    134,812     76.2%      23.8%
Val        322     21,174     68.8%      31.2%
Test       290     13,494     58.5%      41.5%
```

### Files Written in Phase 1

**`src/dataset/explore.py`**
- Scans raw data folders
- Groups frames by video name
- Produces dataset statistics (video counts, frame counts, distributions)
- Saves `data/splits/dataset_index.csv`

**`src/dataset/build_splits.py`**
- Builds 16-frame sliding window clips from all videos
- Performs video-level train/val split (stratified by label)
- Saves `data/splits/train.csv`, `val.csv`, `test.csv`
- Each CSV row = one clip (video_name, category, label, frame_paths)

### config.py (Current State)
```python
# Paths
ROOT_DIR, DATA_DIR, RAW_DIR, FRAMES_DIR, SPLITS_DIR
CHECKPOINTS_DIR, OUTPUTS_DIR

# Dataset
UCF_CRIME_CATEGORIES = [13 anomaly + Normal]
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.8, 0.1, 0.1

# Frame settings
FRAME_INTERVAL = 10      # pre-extracted every 10 original frames
FRAME_SIZE     = (224, 224)
CLIP_LENGTH    = 16
CLIP_STRIDE    = 8
MIN_FRAMES     = 16

# Model
FEATURE_DIM  = 2048      # ResNet50 output
LSTM_HIDDEN  = 256
LSTM_LAYERS  = 2
DROPOUT      = 0.5

# Training
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-4
NUM_EPOCHS     = 50
DEVICE         = "cuda"

# Inference
ANOMALY_THRESHOLD = 0.5
```

---

## Phase 2 — PyTorch Dataset & DataLoader

**Date completed:** May 2026
**Status:** ✅ Complete

### What We Did

This phase had two parallel tracks:
1. **Image DataLoader** (`dataset_loader.py`) — loads raw PNG frames on the fly
2. **Feature DataLoader** (`feature_dataset.py`) — loads pre-computed .npy features (the one actually used for training)

### Track 1: Image Dataset (dataset_loader.py) — Built, then superseded

Written first, before we realized how slow training-from-images would be.

**`UCFCrimeDataset`** — a standard PyTorch Dataset:
- Reads a split CSV
- For each clip row: loads CLIP_LENGTH PNG images from disk
- Applies transforms (resize → tensor → normalize)
- Returns `[CLIP_LENGTH, 3, 224, 224]` FloatTensor + label

**Transforms written:**
```python
TRAIN_TRANSFORMS = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(),        # augmentation — mirror frame horizontally
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

EVAL_TRANSFORMS = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

Why ImageNet normalization? ResNet50 was trained on ImageNet with these exact statistics. Using the same normalization ensures our input matches the distribution the pretrained weights expect. Any deviation degrades the quality of extracted features.

Why no augmentation on eval? We want deterministic predictions at evaluation time. Augmentation is purely a training-time regularization trick.

**Why this DataLoader was not used for training:**
- Loading 16 PNGs per clip + running ResNet50 forward pass ≈ 50ms per clip
- 134,812 clips × 50 epochs × 50ms = ~94 hours of I/O + compute just for data loading
- Not viable. Moved to pre-computed features instead.

### Track 2: Feature Dataset (feature_dataset.py) — The one used for training

**Core insight:** ResNet50's weights never change during training (frozen). So running the same frame through ResNet50 produces the exact same output every time. Running it 50 times (once per epoch) is pure waste.

**Solution:** Run ResNet50 once per frame, save the 2048-dim output to disk as `.npy` files. Training then just loads numpy arrays — a memory operation, not a GPU operation.

**`FeatureDataset` class:**
- Reads split CSV
- For each clip row: looks up the corresponding `.npy` file for that video
- Loads `.npy` file into an in-memory cache on first access (cache keyed by file path)
- Extracts CLIP_LENGTH feature vectors using frame number → array index mapping
- Returns `[CLIP_LENGTH, 2048]` FloatTensor + label

**Frame index mapping strategy:**
Frames in `.npy` files are stored in sorted frame-number order. To find where our clip's frames are in the array:
```python
approx_idx = min(int(frame_number / 10), num_stored - 1)
```
Frame numbers are multiples of 10 (sampled every 10 original frames), so dividing by 10 gives the approximate array index.

**Why num_workers=0?**
The in-memory cache lives in the main process. Multiprocessing workers get a copy-on-fork of the parent process (on Linux) or a fresh process (Windows). Workers don't share the cache, so each worker loads the same .npy files independently — defeating the entire purpose. num_workers=0 keeps everything in one process with a shared cache.

### Class Imbalance Handling

Training split: 76.2% Normal, 23.8% Anomaly. If we sample uniformly, the model sees 3x more normal clips and learns to predict "normal" for everything.

**Solution:** `WeightedRandomSampler`
```python
class_counts  = np.bincount(labels)          # [normal_count, anomaly_count]
class_weights = 1.0 / class_counts           # anomaly clips get higher weight
sample_weights = class_weights[labels]        # per-sample weights
sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
```

Effect: the sampler draws anomaly clips more often, balancing the training distribution without discarding any data.

### Verification Results

```
train : 134,812 clips loaded
val   :  21,174 clips loaded
test  :  13,494 clips loaded

Batch shape : torch.Size([8, 16, 2048])
Expected    : [8, 16, 2048]
Labels      : [1, 0, 1, 0, 0, 1, 1, 0]
Dtype       : torch.float32
Value range : [0.000, 8.002]
```

Value range [0, 8] is expected — these are post-ReLU activations from ResNet50's penultimate layer, which are always ≥ 0 and vary in magnitude based on how strongly each feature fires.

### Files Written in Phase 2
- `src/dataset/dataset_loader.py` — image-based DataLoader (built for learning, superseded)
- `src/dataset/feature_dataset.py` — feature-based DataLoader (used for training)

---

## Phase 3 — Feature Extraction (ResNet50)

**Date completed:** May 2026
**Status:** ✅ Complete

### What We Did

Ran ResNet50 over every frame in the dataset and saved the resulting 2048-dim vectors to disk as `.npy` files.

### Architecture of ResNet50Extractor

```python
class ResNet50Extractor(nn.Module):
    def __init__(self):
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):  # [B, 3, 224, 224]
        out = self.features(x)    # [B, 2048, 1, 1]
        return out.squeeze(-1).squeeze(-1)  # [B, 2048]
```

`backbone.children()` gives: `[conv, bn, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]`. We take everything except the final `fc` layer. The `avgpool` layer converts `[B, 2048, 7, 7]` → `[B, 2048, 1, 1]`, which we then squeeze to `[B, 2048]`.

All weights frozen — ResNet50 is used purely as a feature map, not trained.

### extract_features.py Logic

- **Per-video processing:** For each video, collect all unique frame paths across all clips. Extract features once per frame (not once per clip).
- **Deduplication:** Overlapping clips share frames. We deduplicate at the frame level first — huge savings.
- **MAX_FRAMES = 1000 cap:** Some Normal videos have up to 97,651 frames. Processing all of them is wasteful (each video contributes ~123 clips max). We evenly sample 1000 frames using `np.linspace`.
- **BATCH_SIZE = 64 frames per GPU batch:** Maximizes GPU utilization without running out of 8GB VRAM.
- **Resumable:** Skips `.npy` files that already exist. If the process crashes, re-running picks up where it left off.

### Problem Encountered: Extreme Speed Variation

**What happened:** First few videos processed in ~5 seconds each. Then at ~43% completion, speed dropped to 102-703 seconds per video.

**Root cause:** The dataset contains Normal videos with 97,651 frames (vs ~200 frames for anomaly videos). The normal video folder came later in the iteration order. Processing 97k frames through ResNet50 in batches of 64 takes a long time.

**Solution already in place:** MAX_FRAMES=1000 cap was designed to handle this — but the cap is applied *after* reading the frame list from CSV. The slow videos were the ones that had tens of thousands of unique frames to load from disk. The real bottleneck was Windows disk I/O, not GPU compute.

**Lesson:** Windows reading thousands of scattered small PNG files is dramatically slower than Linux (no page cache warming, different filesystem behavior). On Linux the same extraction would take ~30 minutes. On Windows it took 2h 12min.

### Extraction Results

```
Total unique videos processed : 1,897
Feature files saved           : 1,897
Output directory              : data/features/{split}/{category}/{video_name}.npy

Sample file shape             : [num_frames, 2048]
  (Anomaly videos: ~50-300 frames)
  (Normal videos: capped at 1000 frames)

Total runtime                 : 2 hours 12 minutes
```

### Files Written in Phase 3
- `src/models/feature_extractor.py` — ResNet50 wrapper, frozen weights
- `src/dataset/extract_features.py` — batch extraction script with progress bar

---

## Phase 4 — LSTM Anomaly Detection Model

**Date completed:** May 2026
**Status:** ✅ Complete

### Model Architecture: AnomalyLSTM

```
Input: [batch, 16, 2048]   ← sequence of 16 frame features per clip

    LSTM (input=2048, hidden=256, layers=2, batch_first=True)
         ↓
    last hidden state: hidden[-1] = [batch, 256]
         ↓
    Dropout(0.5)
         ↓
    Linear(256 → 64)
         ↓
    ReLU()
         ↓
    Linear(64 → 1)
         ↓
    Sigmoid()
         ↓
Output: [batch, 1]   ← anomaly probability in [0, 1]
```

**Why stacked LSTM (2 layers)?**
Single LSTM captures simple temporal patterns. The second layer takes the output sequence of the first as input — it learns higher-order temporal abstractions. Two layers is a good balance between expressiveness and training stability; three or more layers often leads to vanishing gradients in video tasks.

**Why unidirectional LSTM?**
A bidirectional LSTM would read the sequence forward and backward, using future context to inform past predictions. In real-time surveillance inference, we don't have future frames. Training bidirectional but deploying unidirectional creates a train/test mismatch. We use unidirectional throughout.

**Why take the last hidden state (hidden[-1]) instead of using all timesteps?**
We want a single anomaly score per clip, not a score per frame. The last hidden state has "seen" all 16 frames and compresses the entire sequence into 256 numbers. Using all timesteps (lstm_out) would require additional pooling. Taking hidden[-1] is simpler and commonly used in weakly supervised settings.

**Classifier head design:**
```
Linear(256 → 64) → ReLU → Linear(64 → 1) → Sigmoid
```
Two linear layers with ReLU adds one non-linearity, which is enough to learn decision boundaries more complex than a single hyperplane. Sigmoid ensures output is always in [0, 1], which is needed for both the ranking loss and threshold-based inference.

**Total parameters: 2,904,193** (all trainable — LSTM + classifier only, ResNet50 is separate and frozen)

### Loss Function: RankingLoss

Based on: Sultani et al., "Real-world Anomaly Detection in Surveillance Videos" (CVPR 2018) — the landmark paper that introduced the UCF-Crime benchmark.

**Core insight:** We don't know which frames within an anomalous video are actually anomalous. But we know:
- Clips from anomalous videos should score higher than clips from normal videos

**The loss enforces this ordering:**
```python
diff = anom_scores.unsqueeze(1) - normal_scores.unsqueeze(0)
# diff shape: [n_anom, n_normal] — every anomaly vs every normal
loss = torch.clamp(margin - diff, min=0.0).mean()
```

- If `anom_score - normal_score ≥ 1.0` → loss = 0 (constraint satisfied)
- If `anom_score - normal_score < 1.0` → loss = `1 - (anom_score - normal_score)` (penalize)

Broadcasting gives us all `n_anom × n_normal` pairwise comparisons in one vectorized operation — more gradient signal than just taking one pair.

**What happens when a batch has only normals or only anomalies?**
The ranking loss cannot compute. We return `torch.tensor(0.0, requires_grad=True)` to avoid crashing while preserving the computational graph for `loss.backward()`.

### Training Setup

| Setting | Value | Reason |
|---|---|---|
| Optimizer | Adam | Adaptive LR per parameter, standard for LSTM training |
| Learning rate | 1e-4 | Safe starting point, conservative enough to not diverge |
| LR scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=5) | Halve LR when AUC stops improving for 5 epochs |
| Gradient clipping | max_norm=1.0 | Prevent exploding gradients in LSTM, common in sequence models |
| Epochs | 50 (started from epoch 10 after bug fix) | |
| Checkpoint | Saved whenever val AUC improves (best_model.pt) | |
| Periodic save | Every 5 epochs | Recovery points |

**Why gradient clipping?** LSTMs are vulnerable to exploding gradients during backpropagation through time. When gradients get very large, the optimizer takes a huge step and destabilizes training. Clipping to norm=1.0 caps the step size while preserving the gradient direction.

### Critical Bug: Checkpoint Saving Never Updated After Epoch 2

**What went wrong (first run):** The training code saved `best_model.pt` using the condition `val_loss < best_val_loss`. But the ranking loss drops to 0.0000 by epoch 2 and stays there — the model consistently ranks anomaly clips above normal clips by the full margin. Once val_loss is 0.0000, no future epoch can improve on it. So `best_model.pt` was permanently stuck at epoch 2's weights (AUC 0.7818).

**Why val_loss saturates:** Ranking loss = `max(0, margin - (anom_score - normal_score))`. When the model always gives anomaly clips scores ≥ 1.0 above normal scores, every pairwise comparison is satisfied and the loss is exactly 0.0000. AUC continued to improve (the score distribution was getting better) but the checkpoint gate never fired again.

**The fix:**
```python
# WRONG — val_loss is 0.0000 from epoch 2 onward
if val_loss < best_val_loss:
    torch.save(...)

# CORRECT — AUC keeps improving even after loss saturates
if val_auc > best_auc:
    torch.save(...)

# Also fixed the LR scheduler
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
# mode="min" (default) would never fire since loss=0 always; mode="max" fires on AUC
```

**Impact:** Had to rerun training from epoch 10 checkpoint (not epoch 2). The final best model is from epoch 29.

### Other Bug Encountered: PyTorch 2.6 weights_only Default Changed

`torch.load(checkpoint)` raised `UnpicklingError` because PyTorch 2.6 changed the default from `weights_only=False` to `weights_only=True`. Fix: add `weights_only=False` to all `torch.load()` calls in `train.py` and `evaluate.py`.

### Training Was Killed by head -30 Pipe (Epoch 12)

First training run was launched as `python train.py 2>&1 | head -30`. The `head` command exits after 30 lines, which sends SIGPIPE to the Python process, killing training at epoch 12. Fix: redirect to log file instead (`> training.log 2>&1`) and monitor with `tail -f`.

### Final Training Results

Training ran for 50 epochs total (resumed from epoch 10 with the corrected checkpoint logic).

**LR schedule progression:**

| LR | Triggered Epoch | Best Val AUC at this LR |
|---|---|---|
| 1e-4 | start | 0.8646 |
| 5e-5 | epoch ~15 | — |
| 2.5e-5 | epoch ~20 | 0.8862 |
| 1.25e-5 | epoch ~25 | **0.8881** |
| 6.25e-6 | epoch ~35 | — |
| 3.13e-6 | epoch ~40 | — |
| 1.56e-6 | epoch ~45 | — |

Each LR halving unlocked a better AUC plateau by allowing finer weight adjustments.

**Best checkpoint:** `checkpoints/best_model.pt` — epoch 29, val AUC = **0.8881**

**Final test set results (from evaluate.py):**

| Metric | Value |
|---|---|
| Test AUC-ROC | **0.8030** |
| Accuracy (threshold=0.5) | 80% |
| Anomaly recall | 69% |
| Sultani et al. 2018 baseline | 0.7510 |

**Per-category AUC (each category vs Normal):**

| Category | AUC | Category | AUC |
|---|---|---|---|
| Burglary | 0.9298 | Robbery | 0.8176 |
| Vandalism | 0.9187 | Stealing | 0.7481 |
| Assault | 0.8831 | Abuse | 0.7362 |
| Arson | 0.8737 | Arrest | 0.7340 |
| Shoplifting | 0.8636 | Fighting | 0.6576 |
| Explosion | 0.8257 | RoadAccidents | 0.6481 |
| — | — | Shooting | 0.6426 |

**Why does Fighting score lower than Burglary?** Burglary and Vandalism have visually distinctive spatial cues (broken glass, forced entry). Fighting involves human body motion that looks similar to normal activity at the frame level — the temporal pattern is harder to capture with a 16-frame clip. RoadAccidents and Shooting have fewer training videos, which limits per-category accuracy.

### Bug in evaluate.py: Per-Category AUC All NaN

**What happened:** The first version of `per_category_auc()` computed AUC within each category in isolation. Each anomaly category in the test set contains only anomalous clips — so `roc_auc_score` received single-class labels → ValueError / NaN.

**Fix:** Changed to compare each anomaly category against all Normal test clips combined:
```python
# WRONG — single class
scores_for_category = df[df["category"] == cat]["score"]
auc = roc_auc_score(labels_for_category, scores)  # all labels=1, fails

# CORRECT — each anomaly category vs Normal clips
normal_scores = df[df["label"] == 0]["score"].values
for cat in anomaly_categories:
    cat_scores = df[df["category"] == cat]["score"].values
    combined = np.concatenate([normal_scores, cat_scores])
    combined_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(cat_scores))])
    cat_auc = roc_auc_score(combined_labels, combined)
```

### Bug: Unicode Arrow on Windows

Used `→` in print statements in evaluate.py. Windows terminal (cp1252) cannot encode this character. Fix: replaced all `→` with `->`.

**Val loss dropping to 0.0000:** Expected and correct. Once the ranking constraint is fully satisfied, val_loss = 0. AUC-ROC is the meaningful metric from that point.

### Files Written in Phase 4
- `src/models/lstm_model.py` — AnomalyLSTM model + RankingLoss
- `src/training/train.py` — training loop with checkpointing, LR scheduling, history logging
- `src/training/evaluate.py` — evaluation on test set with AUC, confusion matrix, per-category breakdown, plots

---

## Phase 5 — YOLOv8 Integration

**Date completed:** May 2026
**Status:** ✅ Complete

### What We Did

Wrapped YOLOv8n as a per-frame object detector that runs on the representative frame of every flagged clip. This gives the system its "explainability" — instead of just saying "this clip is anomalous", it now says "this clip is anomalous — detected: person (0.94), knife (0.71)".

### Why YOLOv8 on top of LSTM, not instead of it

- LSTM answers: "Is this clip temporally anomalous?" (sequence understanding)
- YOLO answers: "What objects are in this frame?" (spatial understanding)
- Combined: LSTM flags the clip → YOLO explains what it saw

Running YOLO on every frame would be redundant — it's only meaningful once the LSTM has already identified a segment worth inspecting.

### detector.py Design

```python
class YOLOv8Detector:
    def detect(self, frame) -> List[Detection]
    def detect_batch(self, frames) -> List[List[Detection]]
    def summarize(self, detections) -> str   # "Detected: person (0.94), knife (0.71)"
```

- Model: `yolov8n.pt` (nano — fastest, ~6MB, sufficient for surveillance objects)
- Confidence threshold: 0.3 (low enough to catch partial occlusions)
- Returns structured `Detection` dataclass: class_name, confidence, bbox, bbox_norm
- Results sorted by confidence descending

### Decision: YOLOv8n over larger variants

YOLOv8 comes in n/s/m/l/x sizes. We chose `n` (nano) because:
- In the pipeline, YOLO only runs on flagged clips — a small fraction of all clips
- Speed matters for a responsive UI
- For "what's in this surveillance frame" the nano model is accurate enough — people, vehicles, and common objects are easily detected
- Configurable: the pipeline accepts `yolo_size` parameter so any size can be swapped in

### Files Written
- `src/models/detector.py` — YOLOv8 wrapper with single and batch detection
- `src/inference/pipeline.py` — end-to-end inference pipeline (video → report)

---

## Phase 6 — Explainability (Grad-CAM)

**Date completed:** May 2026
**Status:** ✅ Complete

### What Grad-CAM Does

Produces a heatmap showing which spatial regions of a frame contributed most to the ResNet50 2048-dim feature vector that the LSTM used to score the clip. Overlaid on the original frame with a jet colormap (blue=low attention, red=high attention).

### Why it works with a frozen network

Grad-CAM only needs gradients for a single backward pass — it does NOT update weights. We:
1. Temporarily re-enable `requires_grad=True` on ResNet50 parameters
2. Run forward + backward pass to get gradients at `layer4`
3. Restore `requires_grad=False` immediately after
4. Weights are never changed

### Algorithm

```
target_layer = ResNet50.layer4   # last conv block, [B, 2048, 7, 7]

Forward:  frame → layer4 activations [2048, 7, 7] → features [2048]
Backward: gradient of L2_norm(features) w.r.t. layer4 activations

weights[c]  = global_avg_pool(gradients[c])         # [2048]
cam         = ReLU( Σ weights[c] * activations[c] ) # [7, 7]
heatmap     = resize(cam / cam.max(), original_size) # [H, W] ∈ [0,1]
overlay     = frame * 0.6 + jet_colormap(heatmap) * 0.4
```

**Why L2 norm as the target scalar?**
We don't have a single class logit to differentiate w.r.t. (no classification head). Using the feature vector norm as a proxy captures which spatial regions produce the strongest activations — i.e., which regions are most "active" and therefore most influential on the LSTM's score.

### Integration into Pipeline

Every anomalous clip automatically gets:
1. YOLO detections (what's in the frame)
2. Grad-CAM heatmap (where the model was looking)
3. Both returned through the API as JPEG endpoints

### Files Written
- `src/explainability/gradcam.py` — Grad-CAM on ResNet50 layer4

---

## Phase 7 — FastAPI Backend

**Date completed:** May 2026
**Status:** ✅ Complete

### Architecture: Async Job Queue

Video analysis takes 20-120 seconds. A synchronous HTTP endpoint would time out in browsers and reverse proxies. Solution: async job pattern.

```
POST /analyze  →  { job_id }         (returns immediately)
GET  /jobs/{id}  →  { status }       (client polls every 1.5s)
GET  /jobs/{id}/results  →  full JSON (when status == "done")
GET  /jobs/{id}/clips/{idx}/frame    (original frame JPEG)
GET  /jobs/{id}/clips/{idx}/heatmap  (Grad-CAM overlay JPEG)
DELETE /jobs/{id}                    (free memory)
```

### Implementation Details

- **FastAPI BackgroundTasks**: `run_analysis()` runs in a thread pool, not blocking the async event loop. The pipeline (PyTorch + YOLO) is CPU/GPU bound, not I/O bound, so it correctly goes to a thread.
- **In-memory job store**: `jobs: Dict[str, Dict]` — simple for a demo, would be Redis in production
- **Pipeline singleton**: loaded once at `@app.on_event("startup")` — not per-request, because ResNet50 + YOLO take ~15s to initialise
- **Image endpoints**: frames and heatmaps served as JPEG `StreamingResponse` — Streamlit can embed them directly via `st.image()`
- **CORS**: enabled for all origins so Streamlit on :8501 can call FastAPI on :8000

### Problem Encountered: python-multipart not installed

FastAPI's `Form()` and `File()` (multipart form data) require `python-multipart`. The server crashed at startup with `RuntimeError: Form data requires python-multipart`. Fixed with `pip install python-multipart`.

### End-to-End Test Results

```
POST /analyze   → job_id returned in <1s
GET  /jobs/{id} → status=done after 2.5s (8s video, 9 clips)
GET  /results   → clips=9, anomalous=9, max_score=1.0
GET  /frame     → 200 OK, 1502 bytes
GET  /heatmap   → 200 OK, 1501 bytes
DELETE          → job cleaned up
```

### Files Written
- `api/main.py` — complete FastAPI application with all endpoints

---

## Phase 8 — Streamlit Frontend

**Date completed:** May 2026
**Status:** ✅ Complete

### What the UI Shows

1. **Upload panel** — drag & drop any MP4/AVI/MOV/MKV
2. **Threshold slider** — adjustable anomaly threshold (default 0.5)
3. **Progress bar** — polls `/jobs/{id}` every 1.5s, fake-increments for UX
4. **Metrics row** — duration, clips analyzed, anomalous clips, peak score
5. **Anomaly timeline chart** — Plotly line chart of score vs time, red shading on flagged regions, threshold dashed line
6. **Flagged clip cards** — sorted by score (highest first), each shows:
   - Left: original frame (fetched from `/clips/{idx}/frame`)
   - Middle: Grad-CAM overlay (fetched from `/clips/{idx}/heatmap`)
   - Right: YOLO detection bars with confidence
7. **Auto-expand top 3** — most suspicious clips open by default
8. **API health widget** — sidebar shows connection status in real time

### Architecture Decision: Streamlit calls FastAPI, not pipeline directly

Original version called the pipeline directly in Streamlit. This works but couples the UI to the ML code. The production version has:

- **FastAPI**: owns the pipeline, job lifecycle, model loading
- **Streamlit**: pure presentation layer, only makes HTTP calls

Benefits:
- Can swap the UI for anything (React, mobile app, curl)
- FastAPI can serve multiple concurrent jobs
- API docs (`/docs`) are free — great for portfolio/interviews

### Files Written
- `ui/app.py` — Streamlit frontend
- `run.py` — single-command launcher for both servers

---

## Phase 9 — Integration & Portfolio Polish

**Date completed:** May 2026
**Status:** ✅ Complete

### End-to-End Integration Test

Full automated test hitting every API endpoint in sequence:
```
Health check   → OK
Submit video   → job_id returned
Poll status    → done in 2s
Fetch results  → 9 clips, all anomalous (Fighting video, score=1.0)
Fetch frame    → 200, JPEG returned
Fetch heatmap  → 200, JPEG returned
Delete job     → cleaned up
```
All endpoints pass.

### README Written

`surveillance-system/README.md` contains:
- Architecture diagram (ASCII)
- Results table (AUC, per-category breakdown)
- Dataset statistics
- Project structure with file descriptions
- Quick start instructions
- API reference table
- Key design decisions table
- Training details and LR schedule
- References to Sultani 2018, Grad-CAM, ResNet50 papers

### What's Production-Ready and What Isn't

**Production-ready:**
- Model training, checkpointing, evaluation
- Full inference pipeline
- API design (endpoints, job pattern, image serving)
- Streamlit UI

**Demo-quality (good enough for portfolio, not for real deployment):**
- In-memory job store (use Redis/database in production)
- No authentication on API
- No rate limiting
- Frames kept in memory per job (use disk storage for large videos)
- Single-worker uvicorn (use gunicorn + workers in production)

---

## Phase 10 — Error Handling & Automated Test Suite

**Date completed:** May 2026
**Status:** ✅ Complete

### Motivation

The system worked end-to-end, but a real portfolio project should be robust: bad inputs should produce clear error messages, not stack traces. And the claim "all endpoints pass" should be backed by reproducible automated tests, not manual curl calls.

The structure mirrors production-grade web apps: layered validation (client → server → pipeline) and a test suite that runs against the live server.

---

### Layer 1 — Streamlit UI (ui/app.py)

Client-side file size validation before the request ever leaves the browser:

| Condition | Behaviour |
|---|---|
| File > 500 MB | `st.error()` → `st.stop()` — blocked before submit |
| File > 200 MB | `st.warning()` — proceeds but user is informed |
| File ≤ 200 MB | Normal flow |

```python
file_mb = uploaded.size / 1024 / 1024
if file_mb > 500:
    st.error("File exceeds the 500 MB limit.")
    st.stop()
if file_mb > 200:
    st.warning(f"Large file ({file_mb:.0f} MB) — analysis may take several minutes.")
```

---

### Layer 2 — FastAPI Input Validation (/analyze endpoint)

Four checks run in order before any work is done:

```
1. Pipeline loaded?           → 503 if not ready yet
2. Threshold ∈ [0.0, 1.0]?  → 400 "threshold must be between..."
3. File extension allowed?   → 400 "Unsupported file type..."
4. File content non-empty?  → 400 "Uploaded file is empty"
   File not too large?      → 413 "File too large (X MB). Maximum: 500 MB"
```

Threshold is validated before reading file bytes (cheap check first). Extension is validated before reading content (avoids reading a multi-GB file just to reject it).

```python
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

if not (0.0 <= threshold <= 1.0):
    raise HTTPException(400, f"threshold must be between 0.0 and 1.0, got {threshold}")

suffix = Path(file.filename or "").suffix.lower()
if suffix not in ALLOWED_EXTENSIONS:
    raise HTTPException(400, f'Unsupported file type "{suffix}". Accepted: mp4, avi, mov, mkv')

content = await file.read()
if len(content) == 0:
    raise HTTPException(400, "Uploaded file is empty")
if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
    raise HTTPException(413, f"File too large ({len(content)/1024/1024:.0f} MB).")
```

---

### Layer 3 — Pipeline Error Classification (run_analysis)

Previously, any exception in the analysis worker just set `job["error"] = str(e)` — cryptic for the user. Now a classifier maps internal exceptions to user-facing messages:

```python
def _classify_pipeline_error(e: Exception) -> str:
    msg = str(e)
    if "Cannot open video" in msg:
        return "Video file could not be opened — it may be corrupt or in an unsupported codec."
    if "too short" in msg.lower():
        return msg   # already user-friendly from pipeline.py
    if "out of memory" in msg.lower() or ("cuda" in msg.lower() and "memory" in msg.lower()):
        return "GPU ran out of memory. Try a shorter video or reduce the file size."
    if "no such file" in msg.lower():
        return "Internal error: temporary file was removed before processing completed."
    return f"Analysis failed: {msg}"
```

Polled from `GET /jobs/{id}` as `status["error"]` — Streamlit displays it in a red error box.

---

### Test Suite — tests/test_api.py

14 automated tests against the running FastAPI server. Run with:
```bash
cd surveillance-system
pytest tests/test_api.py -v
```

**Coverage:**

| Group | # Tests | What's Tested |
|---|---|---|
| Health | 1 | GET /health → 200, pipeline=true, val_auc and test_auc present |
| 404 on unknown job | 5 | status, results, frame, heatmap, delete — all 404 |
| Input validation | 6 | no file (422), wrong extension (400), empty file (400), threshold > 1 (400), threshold < 0 (400), boundary values 0.0/1.0 (valid) |
| Too-short video | 1 | Synthetic 10-frame MP4 submitted → job eventually status=failed, error contains "short" |
| Full job lifecycle | 1 | Submit → results-before-done (400) → poll to done → results shape → frame JPEG → bad clip idx (404) → delete → confirm gone |

**Key implementation details:**
- `tiny_video` fixture: builds a real 10-frame MP4 using `cv2.VideoWriter` — valid enough for OpenCV to open, too short for the 16-frame pipeline minimum
- `_poll()` helper: polls `/jobs/{id}` every 1 second until `status ∈ {done, failed}` or timeout
- `test_full_job_lifecycle` uses the existing `data/test_video.mp4` and is skipped with `pytest.mark.skipif` if the file doesn't exist
- All 14 tests pass (verified against live server)

**Test result:**
```
14 passed in 54.47s
```

---

### Phase 0 — Environment Setup
- [x] Verify GPU via nvidia-smi
- [x] Verify PyTorch CUDA access
- [x] Create virtual environment
- [x] Install PyTorch (cu124)
- [x] Install remaining dependencies
- [x] Create project folder structure
- [x] Write config.py
- [x] Write and run gpu_check.py

### Phase 1 — Dataset
- [x] Download UCF-Crime via Kaggle API
- [x] Audit dataset structure (explore.py)
- [x] Understand frame naming convention
- [x] Build clip CSV files (build_splits.py)
- [x] Verify train/val/test split counts

### Phase 2 — PyTorch Dataset & DataLoader
- [x] Write image-based UCFCrimeDataset (dataset_loader.py)
- [x] Write feature-based FeatureDataset (feature_dataset.py)
- [x] Implement WeightedRandomSampler for class imbalance
- [x] Verify batch shape [8, 16, 2048], dtype float32

### Phase 3 — Feature Extraction (ResNet50)
- [x] Load ResNet50, remove classification head (feature_extractor.py)
- [x] Freeze all weights (no training through backbone)
- [x] Write batch extraction script (extract_features.py)
- [x] Add MAX_FRAMES=1000 cap for large Normal videos
- [x] Extract features for all 1,897 videos (2h 12min on Windows)
- [x] Save per-video .npy files [num_frames, 2048]
- [ ] t-SNE visualization (optional, deferred)

### Phase 4 — LSTM Model & Training
- [x] Build AnomalyLSTM (lstm_model.py)
- [x] Implement RankingLoss (Sultani et al. CVPR 2018)
- [x] Verify model shapes and loss computation
- [x] Write training loop with LR scheduling (train.py)
- [x] Write evaluation script with AUC + plots (evaluate.py)
- [x] Fix checkpoint bug (save by val_auc, not val_loss)
- [x] Fix LR scheduler (mode="max" on AUC)
- [x] Fix PyTorch 2.6 weights_only=False in torch.load
- [x] Fix per-category AUC NaN bug (compare each category vs Normal)
- [x] Fix Unicode arrow in evaluate.py (Windows cp1252)
- [x] Training complete — 50 epochs, best at epoch 29
- [x] Run evaluate.py on test set → Test AUC = 0.8030
- [x] Analyze per-category AUC breakdown (13 categories)

### Phase 5 — YOLOv8 Integration
- [x] Write YOLOv8Detector wrapper (detector.py)
- [x] Single-frame and batch detection
- [x] Summarize detections as human-readable string
- [x] Wire into inference pipeline

### Phase 6 — Explainability (Grad-CAM)
- [x] Implement Grad-CAM on ResNet50 layer4 (gradcam.py)
- [x] Jet colormap heatmap overlay
- [x] Integrated into pipeline — runs on all flagged clips

### Phase 7 — FastAPI Backend
- [x] POST /analyze — async job submission
- [x] GET /jobs/{id} — status polling
- [x] GET /jobs/{id}/results — full JSON results
- [x] GET /jobs/{id}/clips/{idx}/frame — original frame JPEG
- [x] GET /jobs/{id}/clips/{idx}/heatmap — Grad-CAM JPEG
- [x] DELETE /jobs/{id} — memory cleanup
- [x] GET /health — liveness + model info

### Phase 8 — Streamlit Frontend
- [x] Upload panel with file size display
- [x] Threshold slider (configurable)
- [x] Progress bar with API polling
- [x] Metrics row (duration, clips, anomalous, peak score)
- [x] Plotly anomaly timeline chart with red shading
- [x] Flagged clip cards (frame + heatmap + YOLO detections)
- [x] API health check in sidebar
- [x] Single-command launcher (run.py)

### Phase 9 — Integration & Polish
- [x] End-to-end automated API test (all endpoints verified)
- [x] README.md with architecture, results, quick start, API reference
- [x] requirements.txt with exact package versions
- [x] api/__init__.py and ui/__init__.py created
- [x] Test video created from UCF-Crime frames
- [x] Project History updated for all phases

### Phase 10 — Error Handling & Test Suite
- [x] Layer 1: Streamlit file size validation (>500MB blocked, >200MB warned)
- [x] Layer 2: FastAPI input validation (threshold bounds, extension, empty/large file)
- [x] Layer 3: Pipeline error classification (corrupt video, too short, CUDA OOM)
- [x] tests/__init__.py and tests/test_api.py created
- [x] 14 automated tests — all passing (54s runtime)
- [x] pytest installed in venv

### Phase 11 — Pre-Demo Bug Fixes & Docker CPU Path
- [x] Fix Streamlit crash (`width="stretch"` → `use_container_width=True`)
- [x] Fix CUDA fallback in config.py (`torch.cuda.is_available()` check)
- [x] Fix memory leak (DELETE /jobs/{id} after results are rendered)
- [x] Fix Docker startup ordering (healthcheck + `condition: service_healthy`)
- [x] Fix test race condition (conditional results-before-done assertion)
- [x] Remove `--reload` from run.py
- [x] Fix requirements.txt for CPU machines (`>=` bounds + install instructions)
- [x] Add docker-compose.cpu.yml for GPU-free Docker path
- [x] Delete dead code (`new files/` directory)
- [x] All 14 tests passing (verified: `14 passed in 47.50s`)

---

## Decisions Log (Quick Reference)

| Decision | Choice | Reason |
|---|---|---|
| Framework | PyTorch | YOLOv8 requires it; industry standard |
| Dataset | UCF-Crime (odins0n Kaggle) | Most downloaded, validated, 13 real categories |
| Feature extractor | ResNet50 pretrained | Transfer learning, 2048-dim features, stable |
| Temporal model | LSTM | Simpler than 3D CNN, more explainable |
| Object detector | YOLOv8 | State of art, PyTorch-native |
| Explainability | Grad-CAM | Standard visual explanation for CNNs |
| API | FastAPI | Async, production-grade, easy to learn |
| UI | Streamlit | Fast to build, clean for demos |
| Clip length | 16 frames | Standard in video understanding |
| Clip stride | 8 (50% overlap) | Ensures boundary anomalies are captured |
| Split strategy | Video-level | Prevents data leakage |
| Jupyter | Skipped (use VS Code) | Windows Long Path error with JupyterLab |
| Training data pipeline | Pre-computed .npy features | 500x speedup vs loading images each epoch |
| Checkpoint saving | Best AUC (not val loss) | Val loss saturates at 0 after epoch 2; AUC stays informative |
| LR scheduler mode | mode="max" on AUC | val_loss=0 always, so mode="min" would never fire; AUC drives halving |
| YOLO model size | YOLOv8n (nano) | Only runs on flagged clips, not every frame; speed > accuracy here |
| Grad-CAM target | ResNet50 layer4 | Last conv block before avgpool; most semantically rich spatial features |
| Grad-CAM scalar | L2 norm of features | No classification logit available; norm captures overall feature magnitude |
| API pattern | Async job queue | Analysis takes 20-120s; synchronous endpoint would time out |
| Job store | In-memory dict | Sufficient for demo; production would use Redis |
| Streamlit↔FastAPI | HTTP calls only | Decouples UI from ML code; API can serve any client |
| python-multipart | Required for FastAPI Form/File | Not installed by default; needed for video upload endpoint |
| Checkpoint metric | val_auc not val_loss | val_loss=0.0000 from epoch 2 onward — checkpoint gate would never fire again |
| LR scheduler mode | mode="max" on AUC | mode="min" (default) would never halve LR since loss stays at 0 |
| Error classification | _classify_pipeline_error() | Maps raw exceptions to user-facing messages; hides internal paths and stack traces |
| Input validation order | threshold → extension → content | Cheap checks first; avoids reading multi-GB files just to reject bad threshold |
| Test framework | pytest + requests | Standard Python test tooling; tests run against live server, not mocked |
| Tiny video fixture | cv2.VideoWriter, 10 frames | Valid MP4 (OpenCV can open) but below the 16-frame minimum — tests "too short" path |
| MAX_FRAMES cap | 1000 per video | Normal videos up to 97k frames; cap at 1000 = enough clips, fast extraction |
| Feature cache | In-memory dict, num_workers=0 | Workers don't share memory on Windows; single process shares one cache |
| Class imbalance | WeightedRandomSampler | Oversamples anomaly clips (23.8%) without discarding normal data |
| Loss function | RankingLoss margin=1.0 | Sultani et al. CVPR 2018; enforces clip-level ordering under weak supervision |
| LSTM direction | Unidirectional | Bidirectional would use future frames — invalid for real-time inference |
| Hidden state used | last hidden (hidden[-1]) | Single clip-level score needed; last state encodes full sequence |
| Gradient clipping | max_norm=1.0 | Prevents exploding gradients in backprop through time |
| LR scheduler | ReduceLROnPlateau (patience=5) | Automatically adapts if training stagnates |

---

## Interview Talking Points (Running List)

**On the dataset:**
- "UCF-Crime is a weakly supervised dataset — training labels are video-level only, not frame-level. This reflects real-world conditions where nobody annotates every frame."
- "We split train/val at the video level, not clip level, to prevent data leakage — clips from the same video cannot appear in both splits."
- "The dataset has 76% normal clips vs 24% anomaly clips. We used WeightedRandomSampler to oversample the minority class — otherwise the model learns to predict 'normal' for everything and gets 76% accuracy while being useless."

**On the architecture:**
- "ResNet50 gives us 2048-dimensional spatial features per frame. The LSTM then processes a sequence of these features to understand temporal patterns — what's changing over time."
- "We freeze ResNet50 entirely and only train the LSTM + classifier head. This is transfer learning: we borrow spatial knowledge from ImageNet and add temporal reasoning on top."
- "The LSTM is unidirectional — not bidirectional — because in real-time surveillance we don't have access to future frames. Training with future context but deploying without it creates a train/test mismatch."

**On training efficiency:**
- "We pre-computed ResNet50 features once to disk and trained the LSTM on numpy arrays. This is a ~500x speedup — loading a numpy slice takes 0.1ms vs loading 16 images + running ResNet50 takes ~50ms. With 134,812 training clips × 50 epochs, the savings are massive."
- "The first epoch was slow because our in-memory cache was cold. Every .npy file had to be read from disk. From epoch 2 onward everything is cached in RAM and training is ~40x faster."

**On the loss function:**
- "We used a ranking loss from Sultani et al. (CVPR 2018) — the paper that introduced the UCF-Crime benchmark. The key insight: we don't know which frames are anomalous, but we know anomalous clips should score higher than normal clips. The loss enforces this ordering using pairwise comparisons across all anomaly-normal pairs in the batch."
- "The ranking loss is zero when all anomaly scores beat all normal scores by at least the margin (1.0). Once the model learns basic ordering, the ranking loss saturates — AUC-ROC then becomes the meaningful metric to watch."

**On evaluation:**
- "The primary metric is AUC-ROC, not accuracy. With 76% normal clips, a model predicting 'normal' for everything gets 76% accuracy — that's useless. AUC-ROC measures how well the model separates the two classes regardless of any threshold."
- "The standard benchmark metric for UCF-Crime is frame-level AUC-ROC on the test set."
- "Grad-CAM shows which region of the frame the model was paying attention to — this makes the system explainable, not just a black box."

---

## Phase 12 — Vanilla JS Frontend, Production QA Audit & 6 Critical Bug Fixes

**Date:** May 2026

### What Changed

The Streamlit frontend was replaced entirely with a static vanilla HTML/CSS/JS app served by FastAPI itself. An 8-phase engineering audit identified 6 critical bugs, all fixed, with 48 automated tests written to verify each fix.

### Stack Change

| Before | After |
|---|---|
| FastAPI (API) + Streamlit (UI) | FastAPI serving both API and static UI |
| Two processes (`run.py` spawned both) | Single process |
| Two ports (:8000 API, :8501 UI) | Single port (:8000) |
| `streamlit==1.50.0` in requirements | Removed — no pip package needed |
| Two Docker services | One Docker service |

### New Frontend Files

| File | Purpose |
|---|---|
| `ui/index.html` | Single-page shell, Chart.js CDN, semantic sections |
| `ui/style.css` | Dark surveillance theme, CSS variables, responsive |
| `ui/app.js` | Drag-drop upload, job polling, Chart.js timeline, clip accordion, cleanup |

### Key Design Decisions

**Why not React/Vue?**
No build step, no node_modules, no bundler. The UI is a single-page client talking to one API. Vanilla JS is sufficient and keeps the repository clean.

**Why `const API = ""`?**
Relative URL prefix means the frontend works on localhost:8000, Docker :8000, and HuggingFace Spaces :7860 without any configuration change.

**Why StaticFiles mount last?**
FastAPI route matching is first-match. The StaticFiles mount must be registered after all API routes so `/health`, `/analyze`, `/jobs/{id}` etc. are matched by their actual handlers — not intercepted by the file server.

### Bug Fixes

**Fix 1 — showSection() invisible sections**
CSS `display:none` + JS `element.style.display = ""` = section stays hidden forever. CSS stylesheet rule wins when inline style is cleared. Fix: set explicit `"flex"`.

**Fix 2a — sendBeacon DELETE**
`navigator.sendBeacon()` only sends POST. Cleanup calls to `DELETE /jobs/{id}` always got 405. Fix: `fetch(..., { method: "DELETE", keepalive: true })`.

**Fix 2b — infinite memory growth**
Jobs accumulated in-memory dict with no eviction. Added `JOB_TTL_SECONDS = 1800`, `_evict_expired_jobs()` async background task, `created_at` timestamp on every job, and `logging.basicConfig`.

**Fix 3 — dead Docker container**
`docker-compose.yml` had a `ui:` service pointing at a deleted file (`ui/app.py`). Would crash-loop on `docker compose up`. Removed entire `ui:` service.

**Fix 4 — wrong README**
README still described Streamlit architecture (`:8501`, `[Streamlit Frontend]`, `ui/app.py`). Updated to reflect vanilla JS frontend.

**Fix 5 — dead dependency**
`streamlit==1.50.0` in requirements.txt installed ~40 packages for a UI that did not exist. Removed.

**Fix 6 — log files not ignored**
`api/api.log` not in `.gitignore`. Added `*.log` pattern.

### Test Coverage Added

```
tests/
  test_api.py              14 tests  — core endpoint lifecycle
  test_fixes_static.py     33 tests  — source-code audit, no server needed
  test_fixes_runtime.py    15 tests  — live API + TTL logic
  README.md                — documents how to run all test groups
```

All 33 static tests pass without a running server. Static tests use `re.search` with `\s*` to tolerate alignment whitespace in source code.

### Phase 12 Checklist

- [x] Streamlit removed from requirements.txt
- [x] ui/index.html, ui/style.css, ui/app.js created
- [x] FastAPI StaticFiles mount (last route)
- [x] run.py simplified, startup.sh updated
- [x] Fix 1: showSection "flex" fix
- [x] Fix 2a: sendBeacon → keepalive fetch
- [x] Fix 2b: TTL eviction + logging + created_at
- [x] Fix 3: docker-compose Streamlit service removed
- [x] Fix 4: README updated
- [x] Fix 5: requirements.txt cleaned
- [x] Fix 6: *.log in .gitignore
- [x] 48 tests written, all passing
- [x] Committed and pushed to GitHub

---
