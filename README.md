# AI Video Surveillance — Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-orange?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red?logo=streamlit)
![Tests](https://img.shields.io/badge/tests-14%20passed-brightgreen)
![AUC](https://img.shields.io/badge/Test%20AUC--ROC-0.8030-blue)

An end-to-end deep learning system that detects anomalous events in surveillance videos. Upload a video, get back timestamped anomaly scores, object detections, and Grad-CAM heatmaps showing exactly what triggered each alert.

**Test AUC-ROC: 0.8030** on the UCF-Crime benchmark — beats the original Sultani et al. 2018 paper (0.7510) by ~5 percentage points.

---

## Architecture

```
Input Video (.mp4 / .avi / .mov / .mkv)
        │
        ▼
[Frame Extraction]  — OpenCV
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
[ResNet50 Extractor]            [YOLOv8n Detector]
 2048-dim per frame              Bounding boxes
 (frozen, ImageNet)              on flagged frames
        │                                 │
        ▼                                 │
[Sliding Window Clips]                    │
 16 frames, stride 8                      │
        │                                 │
        ▼                                 │
[AnomalyLSTM]                   [Grad-CAM]
 Score ∈ [0, 1]                  Heatmap overlay
 per clip                        on flagged frames
        │                                 │
        └──────────────┬──────────────────┘
                       ▼
           [FastAPI Backend]  :8000
                       │
                       ▼
           [Streamlit Frontend]  :8501
```

---

## Results

| Metric | Value |
|---|---|
| Val AUC-ROC | 0.8881 |
| **Test AUC-ROC** | **0.8030** |
| Accuracy (threshold=0.5) | 80% |
| Anomaly recall | 69% |
| Sultani et al. 2018 (baseline) | 0.7510 |

### Per-Category AUC (test set, each vs Normal)

| Category | AUC | Category | AUC |
|---|---|---|---|
| Burglary | 0.9298 | Robbery | 0.8176 |
| Vandalism | 0.9187 | Stealing | 0.7481 |
| Assault | 0.8831 | Abuse | 0.7362 |
| Arson | 0.8737 | Arrest | 0.7340 |
| Shoplifting | 0.8636 | Fighting | 0.6576 |
| Explosion | 0.8257 | RoadAccidents | 0.6481 |
| — | — | Shooting | 0.6426 |

ROC curve and score distribution plots are in [`outputs/`](outputs/).

---

## Dataset

**UCF-Crime** (`odins0n/ucf-crime-dataset` on Kaggle)
- 1,900 surveillance videos, 128 hours
- 13 anomaly categories + Normal
- Pre-extracted PNG frames (every 10th original frame)
- Video-level weak labels (frame-level labels only in test set)

| Split | Videos | Clips | Normal% |
|---|---|---|---|
| Train | 1,285 | 134,812 | 76.2% |
| Val | 322 | 21,174 | 68.8% |
| Test | 290 | 13,494 | 58.5% |

---

## Project Structure

```
surveillance-system/
├── config.py                    # all settings centralised
├── run.py                       # launch both servers with one command
├── requirements.txt
│
├── api/
│   └── main.py                  # FastAPI backend (async job queue)
│
├── ui/
│   └── app.py                   # Streamlit frontend
│
├── src/
│   ├── dataset/
│   │   ├── explore.py           # dataset statistics
│   │   ├── build_splits.py      # sliding-window clip CSVs
│   │   ├── dataset_loader.py    # image-based DataLoader
│   │   ├── feature_dataset.py   # .npy feature DataLoader (used for training)
│   │   └── extract_features.py  # ResNet50 batch extraction
│   │
│   ├── models/
│   │   ├── feature_extractor.py # frozen ResNet50 wrapper
│   │   ├── lstm_model.py        # AnomalyLSTM + RankingLoss
│   │   └── detector.py          # YOLOv8n wrapper
│   │
│   ├── training/
│   │   ├── train.py             # training loop with checkpointing
│   │   └── evaluate.py          # AUC, confusion matrix, plots
│   │
│   ├── explainability/
│   │   └── gradcam.py           # Grad-CAM on ResNet50 layer4
│   │
│   └── inference/
│       └── pipeline.py          # end-to-end: video -> report
│
├── tests/
│   └── test_api.py              # 14 automated tests (pytest)
│
├── checkpoints/
│   └── best_model.pt            # epoch 29, val AUC 0.8881
│
└── outputs/
    ├── roc_curve.png
    ├── score_distribution.png
    └── evaluation_results.json
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/pranavdubey1725/ai-video-surveillance-anomaly-detection.git
cd ai-video-surveillance-anomaly-detection

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# PyTorch with CUDA 12.4 (adjust index URL for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

### 2. Start both servers

```bash
python run.py
```

Open **http://localhost:8501** in your browser. The `best_model.pt` checkpoint is included — no retraining required.

### 3. Analyze a video

- Upload any `.mp4`, `.avi`, `.mov`, or `.mkv` file
- Adjust the anomaly threshold (default 0.5)
- Click **Analyze Video**
- View the anomaly timeline, flagged clips, Grad-CAM heatmaps, and YOLO detections

---

## Retraining from Scratch

```bash
# 1. Download dataset (requires Kaggle API key)
kaggle datasets download -d odins0n/ucf-crime-dataset -p data/raw --unzip

# 2. Build clip CSVs
python src/dataset/build_splits.py

# 3. Pre-compute ResNet50 features (~2h on Windows, ~30min on Linux)
python src/dataset/extract_features.py

# 4. Train — 50 epochs, ~90 min on RTX 4060
python src/training/train.py

# 5. Evaluate on test set
python src/training/evaluate.py
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check + model info |
| POST | `/analyze` | Upload video, start background job |
| GET | `/jobs/{id}` | Poll job status |
| GET | `/jobs/{id}/results` | Full results JSON |
| GET | `/jobs/{id}/clips/{idx}/frame` | Original frame image (JPEG) |
| GET | `/jobs/{id}/clips/{idx}/heatmap` | Grad-CAM overlay image (JPEG) |
| DELETE | `/jobs/{id}` | Free memory for completed job |

Interactive docs: **http://localhost:8000/docs**

### Error responses

| Status | Meaning |
|---|---|
| 400 | Bad input — invalid threshold, unsupported file type, or empty file |
| 413 | File too large (>500 MB) |
| 422 | Missing required field |
| 503 | Pipeline not loaded yet — retry in a few seconds |

---

## Test Suite

```bash
# Requires both servers to be running (python run.py)
pytest tests/test_api.py -v
```

14 automated tests covering:
- Health check
- 404 handling for unknown jobs (status, results, frame, heatmap, delete)
- Input validation (no file, wrong extension, empty file, threshold out of range)
- Too-short video handling (job fails gracefully with readable error)
- Full job lifecycle (submit → poll → results → frame image → delete → confirm gone)

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Feature extractor | ResNet50 (frozen) | Transfer learning; 500x faster than on-the-fly extraction |
| Temporal model | LSTM (2 layers, hidden=256) | Simpler than 3D CNN, interpretable, interview-friendly |
| Loss function | Ranking loss (Sultani 2018) | Weak supervision: video-level labels only, no frame annotations |
| Class imbalance | WeightedRandomSampler | 76% normal without balancing makes model predict "normal" always |
| Metric | AUC-ROC (not accuracy) | Imbalanced dataset: 76% accuracy achievable by predicting all-normal |
| Split strategy | Video-level | Clip-level split causes data leakage — clips from same video in train+val |
| Checkpoint saving | Best AUC (not val loss) | Val loss saturates at 0 after epoch 2; AUC remains informative |
| LR scheduler mode | mode="max" | Triggers on AUC improvement, not val loss (which is 0 from epoch 2) |
| YOLO | Only on flagged clips | Running YOLO on every frame is redundant and slow |
| Grad-CAM target | L2 norm of features | No classification logit available; norm captures which regions fire most |
| API pattern | Async job queue | Analysis takes 20-120s; synchronous endpoint would time out |
| Streamlit | Calls FastAPI only | Decouples UI from ML; API can serve any client |

---

## Training Details

| Setting | Value |
|---|---|
| Optimizer | Adam, lr=1e-4 |
| LR schedule | ReduceLROnPlateau (mode=max, factor=0.5, patience=5) |
| Gradient clipping | max_norm=1.0 |
| Epochs | 50 (best checkpoint at epoch 29) |
| Batch size | 32 |
| Training time | ~90 min on RTX 4060 Laptop |

LR halving progression and AUC gains:

| LR | Best Val AUC |
|---|---|
| 1e-4 | 0.8646 |
| 2.5e-5 | 0.8862 |
| 1.25e-5 | **0.8881** |

---

## Hardware

- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM)
- CUDA: 12.4
- OS: Windows 11
- PyTorch: 2.6.0+cu124

---

## Development Log

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for the full project history — every decision, bug, fix, and reasoning captured throughout the build.

---

## References

- Sultani, W., Chen, C., & Shah, M. (2018). **Real-world anomaly detection in surveillance videos**. CVPR 2018. *(Introduced UCF-Crime dataset and ranking loss)*
- Selvaraju, R. R., et al. (2017). **Grad-CAM: Visual explanations from deep networks**. ICCV 2017.
- He, K., et al. (2016). **Deep residual learning for image recognition**. CVPR 2016.
