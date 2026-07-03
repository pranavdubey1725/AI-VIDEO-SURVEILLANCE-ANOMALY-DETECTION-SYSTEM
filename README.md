# AI Video Surveillance: Anomaly Detection

An end-to-end deep learning system for detecting anomalous events in surveillance footage. Users upload a video and receive timestamped anomaly scores, object detections, and Grad-CAM heatmaps that highlight the regions responsible for each alert.

The model is trained on the UCF-Crime benchmark and achieves a **test AUC-ROC of 0.8030**, an improvement of roughly five percentage points over the original Sultani et al. (2018) baseline of 0.7510.

---

## Results

| Metric | Value |
|---|---|
| Validation AUC-ROC | 0.8881 |
| **Test AUC-ROC** | **0.8030** |
| Accuracy at threshold 0.5 | 80% |
| Anomaly recall | 69% |
| Sultani et al. (2018) baseline | 0.7510 |

### Per-Category Test AUC (each category vs. Normal)

| Category | AUC | Category | AUC |
|---|---|---|---|
| Burglary | 0.9298 | Robbery | 0.8176 |
| Vandalism | 0.9187 | Stealing | 0.7481 |
| Assault | 0.8831 | Abuse | 0.7362 |
| Arson | 0.8737 | Arrest | 0.7340 |
| Shoplifting | 0.8636 | Fighting | 0.6576 |
| Explosion | 0.8257 | Road Accidents | 0.6481 |
| Robbery | 0.8176 | Shooting | 0.6426 |

ROC curve and score distribution plots are available in `outputs/`.

---

## Architecture

```
Input Video (.mp4 / .avi / .mov / .mkv)
        |
        v
[Frame Extraction]           OpenCV reads every frame
        |
        +----------------------------------+
        v                                  v
[ResNet50 Feature Extractor]      [YOLOv8n Object Detector]
 2048-dim vector per frame          Bounding boxes on flagged frames only
 Frozen ImageNet weights
        |
        v
[Sliding Window Clips]
 16 frames per clip, stride 8
        |
        v
[AnomalyLSTM]                [Grad-CAM]
 Score in [0, 1] per clip     Heatmap overlay on flagged frames
        |                              |
        +----------------+-------------+
                         v
             [FastAPI Backend]  :8000
              (serves UI + API)
                         |
                         v
        [Vanilla JS Frontend — browser]
```

---

## Component Overview

### Feature Extractor — ResNet50 (Frozen)

Each frame is resized to 224×224 and passed through a pretrained ResNet50 to produce a 2048-dimensional feature vector. The weights remain frozen throughout — no fine-tuning is performed. This follows the standard transfer-learning approach: ImageNet features capture texture, edges, and object shapes with enough fidelity to distinguish normal from anomalous activity without retraining on surveillance-specific data.

Pre-computing features for the full dataset takes approximately two hours on a GPU. At inference time, features are computed on the fly for each uploaded video.

### Temporal Model — AnomalyLSTM

```
Input:  [batch, 16 frames, 2048 features]
LSTM:   2 layers, hidden size 256, dropout 0.5
Output: [batch, 1]  — anomaly score in [0, 1]
```

Each 16-frame clip is drawn from a sliding window over the video (stride 8, so consecutive clips overlap by half). The LSTM processes the sequence of frame features and outputs a single score per clip. The final hidden state of the top LSTM layer passes through a compact fully connected head (256 → 64 → 1) with a sigmoid activation.

**Why LSTM rather than a 3D CNN?** LSTMs are simpler to train, require less memory, and are easier to reason about and explain. A 3D CNN would likely deliver higher recall on fast-motion anomalies such as fights or explosions, but at the cost of a harder-to-debug model and slower iteration.

### Loss Function — Ranking Loss (Sultani et al., 2018)

UCF-Crime provides only video-level labels — each video is marked "anomalous" or "normal," with no frame-level annotations in the training set. This constitutes weak supervision.

The ranking loss is designed for exactly this setting: rather than predicting exact scores, it only requires that anomalous clips score higher than normal clips by a fixed margin.

```
loss = mean( max(0, 1 - score_anomalous + score_normal) )
```

Every anomalous score in a batch is compared against every normal score. If an anomalous score already exceeds a normal score by at least 1, that pair contributes zero loss. This is the central idea introduced in the paper that established the UCF-Crime dataset.

### Object Detector — YOLOv8n

YOLOv8n runs only on the representative frame of clips the LSTM has flagged as anomalous. Running it on every frame would be redundant, since the LSTM already handles temporal anomaly detection — YOLO's role is to add interpretability by identifying which objects (people, vehicles, and any labeled weapons) were visible in a flagged clip.

### Explainability — Grad-CAM

Grad-CAM identifies which spatial regions of an input frame the ResNet50 responded to most strongly. It computes gradients of the L2 norm of the feature map (the layer4 output) with respect to the feature map activations, then uses those gradients to weight the activation maps and produce a heatmap.

Note that the standard Grad-CAM target is a class logit. Since this ResNet50 has been stripped of its classification head to produce the 2048-dimensional feature vector, the L2 norm of the feature map is used as a proxy target instead. This yields visually plausible heatmaps, though it is less theoretically grounded than class-discriminative Grad-CAM.

### API — FastAPI with an Asynchronous Job Queue

Video analysis takes between 20 and 120 seconds depending on length — too long for a synchronous HTTP request, which risks timing out in browsers and proxies. The system instead uses a job-queue pattern:

1. `POST /analyze` saves the uploaded video to a temporary file, starts a background thread, and immediately returns a `job_id`.
2. The client polls `GET /jobs/{id}` until `status == "done"`.
3. `GET /jobs/{id}/results` returns the full results as JSON.
4. Frame and heatmap images are served as JPEG streams through dedicated endpoints.
5. `DELETE /jobs/{id}` releases the in-memory frame data once a job is no longer needed.

Jobs are held in a plain Python dictionary in memory, so a server restart clears all job state. This is acceptable for local development; a production deployment would use Redis or a database instead.

### Frontend — Vanilla HTML/CSS/JS

The frontend is a single-page application served as static files directly by FastAPI (via `StaticFiles`). It communicates with the backend exclusively through the REST API and never touches the ML models directly. This decoupling means the API can serve any client — mobile, CLI, or another service — without any changes to the ML code.

---

## Dataset

**UCF-Crime** (`odins0n/ucf-crime-dataset` on Kaggle)

- 1,900 surveillance videos totaling 128 hours
- 13 anomaly categories: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Road Accidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism
- Plus a Normal class
- Training labels are video-level only (no frame annotations)
- The test set includes frame-level binary labels, used solely for evaluation

| Split | Videos | Clips | Normal fraction |
|---|---|---|---|
| Train | 1,285 | 134,812 | 76.2% |
| Validation | 322 | 21,174 | 68.8% |
| Test | 290 | 13,494 | 58.5% |

Splits are constructed at the video level rather than the clip level. Clip-level splitting would allow clips from the same video to appear in both training and validation sets, causing data leakage and inflating validation AUC.

---

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam, lr = 1e-4 |
| LR schedule | ReduceLROnPlateau (mode=max, patience=5, factor=0.5) |
| Gradient clipping | max_norm = 1.0 |
| Epochs | 50 (best checkpoint saved at epoch 29) |
| Batch size | 32 |
| Class imbalance handling | WeightedRandomSampler (76% of clips are Normal) |
| Training time | ~90 minutes on RTX 4060 Laptop (8 GB VRAM) |

**Learning rate schedule progression:**

| Learning rate | Best validation AUC |
|---|---|
| 1e-4 | 0.8646 |
| 2.5e-5 | 0.8862 |
| 1.25e-5 | **0.8881** (saved as `best_model.pt`) |

The scheduler operates in `mode="max"` to monitor AUC rather than loss. Validation loss collapses to 0.0 after epoch 2 — a known artifact of the ranking loss under class imbalance — so monitoring loss instead would prevent the learning rate from ever decaying.

---

## Project Structure

```
surveillance-system/
|-- config.py                    all hyperparameters and paths in one place
|-- run.py                       launch both servers with one command
|-- requirements.txt
|
|-- api/
|   +-- main.py                  FastAPI backend — async job queue, 7 endpoints
|
|-- ui/
|   |-- index.html               Web UI — upload, progress, results dashboard
|   |-- style.css                Dark minimal theme (Inter + JetBrains Mono)
|   +-- app.js                   Vanilla JS — polling, Chart.js chart, clip cards
|
|-- src/
|   |-- dataset/
|   |   |-- explore.py           dataset statistics and sanity checks
|   |   |-- build_splits.py      video-level train/val/test split + clip CSV export
|   |   |-- dataset_loader.py    image-based DataLoader (used for exploration)
|   |   |-- feature_dataset.py   .npy feature DataLoader (used during training)
|   |   +-- extract_features.py  ResNet50 batch feature extraction to disk
|   |
|   |-- models/
|   |   |-- feature_extractor.py frozen ResNet50 wrapper
|   |   |-- lstm_model.py        AnomalyLSTM + RankingLoss
|   |   +-- detector.py          YOLOv8n wrapper
|   |
|   |-- training/
|   |   |-- train.py             training loop with checkpointing
|   |   +-- evaluate.py          AUC-ROC, confusion matrix, per-category breakdown
|   |
|   |-- explainability/
|   |   +-- gradcam.py           Grad-CAM on ResNet50 layer4
|   |
|   +-- inference/
|       +-- pipeline.py          end-to-end: video -> VideoResult dataclass
|
|-- tests/
|   +-- test_api.py              14 automated pytest tests
|
|-- checkpoints/
|   +-- best_model.pt            epoch 29, val AUC 0.8881
|
|-- outputs/
|   |-- roc_curve.png
|   |-- score_distribution.png
|   +-- evaluation_results.json
|
|-- Dockerfile
|-- docker-compose.yml
+-- .dockerignore
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- An NVIDIA GPU with CUDA 12.4 is recommended (CPU is supported, but inference is slower)

### 1. Clone and Install

```bash
git clone https://github.com/pranavdubey1725/AI-VIDEO-SURVEILLANCE-ANOMALY-DETECTION-SYSTEM.git
cd AI-VIDEO-SURVEILLANCE-ANOMALY-DETECTION-SYSTEM

python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# PyTorch with CUDA 12.4 (adjust the index URL for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Remaining dependencies
pip install -r requirements.txt
```

For CPU-only environments, install the default PyTorch build instead:

```bash
pip install torch torchvision torchaudio
```

Then open `config.py` and change `DEVICE = "cuda"` to `DEVICE = "cpu"`.

### 2. Start the Server

```bash
python run.py
```

This launches FastAPI on port 8000, serving both the REST API and the web UI from a single process. Open `http://localhost:8000` in your browser. The included `best_model.pt` checkpoint means no retraining is required to get started.

### 3. Analyze a Video

- Upload a `.mp4`, `.avi`, `.mov`, or `.mkv` file
- Adjust the anomaly threshold slider (default: 0.5)
- Click **Analyze Video**
- Review the anomaly score timeline, flagged clips, Grad-CAM heatmaps, and YOLO detections

---

## Running with Docker

Docker runs the API and UI as two separate containers on a shared network. The UI container reaches the API through the Docker service name (`http://api:8000`) rather than `localhost`; this is handled automatically via an environment variable, so no code changes are required for local development.

### Prerequisites

- Docker and Docker Compose (V2)
- For GPU acceleration: the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Build and Start

```bash
docker compose up --build
```

The first build takes several minutes, since PyTorch (~2.5 GB) is downloaded and installed inside the image. Subsequent builds reuse the cached layer.

Open `http://localhost:8000` in your browser.

### CPU-Only (No GPU)

A ready-made CPU variant is available and requires no edits:

```bash
docker compose -f docker-compose.cpu.yml up --build
```

CPU inference works but is significantly slower — roughly 8–12 minutes per 60-second video, compared to under 30 seconds on a GPU.

### Stop

```bash
docker compose down
```

### A Note on Image Size

The Docker image is large (approximately 6–8 GB) due to the PyTorch CUDA wheels, which is typical for GPU-enabled deep learning containers. The `.dockerignore` file ensures the 11.8 GB dataset, pre-computed features, and intermediate checkpoints are never copied into the image.

---

## API Reference

Interactive documentation is available at `http://localhost:8000/docs` while the server is running.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check and model metadata |
| POST | `/analyze` | Upload a video and start a background analysis job |
| GET | `/jobs/{id}` | Poll job status: queued / processing / done / failed |
| GET | `/jobs/{id}/results` | Full results as JSON (available once status is done) |
| GET | `/jobs/{id}/clips/{idx}/frame` | Representative frame as JPEG |
| GET | `/jobs/{id}/clips/{idx}/heatmap` | Grad-CAM overlay as JPEG |
| DELETE | `/jobs/{id}` | Free memory for a completed job |

### Error Codes

| Status | Meaning |
|---|---|
| 400 | Bad input — invalid threshold, unsupported file type, or empty file |
| 413 | File too large (above 500 MB) |
| 422 | Missing required field |
| 503 | Pipeline not yet loaded — retry after a few seconds |

---

## Tests

```bash
# Requires both servers running (python run.py)
cd surveillance-system
pytest tests/test_api.py -v
```

The suite includes 14 automated tests covering:

- Correct model metadata returned by the health endpoint
- 404 responses for unknown job IDs (status, results, frame, heatmap, delete)
- Input validation: missing file, wrong extension, empty file, threshold out of range
- Graceful failure with a readable error message for videos shorter than 16 frames
- The full job lifecycle: submit, poll, fetch results, fetch frame image, delete, confirm removal

Tests generate a synthetic 10-frame video with OpenCV rather than requiring a real surveillance clip. The full lifecycle test (`test_full_job_lifecycle`) is skipped unless a file is present at `data/test_video.mp4`.

---

## Retraining from Scratch

```bash
# 1. Download the dataset (requires Kaggle API credentials)
kaggle datasets download -d odins0n/ucf-crime-dataset -p data/raw --unzip

# 2. Build clip CSVs (video-level splits)
python src/dataset/build_splits.py

# 3. Pre-compute ResNet50 features and save as .npy files
#    (~2 hours on Windows with a GPU, ~30 minutes on Linux)
python src/dataset/extract_features.py

# 4. Train — 50 epochs, ~90 minutes on an RTX 4060 Laptop
python src/training/train.py

# 5. Evaluate on the test set and generate output plots
python src/training/evaluate.py
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Feature extractor | Frozen ResNet50 | ~500x faster than training a CNN from scratch; ImageNet features transfer well |
| Temporal model | LSTM, 2 layers, hidden size 256 | Simpler than a 3D CNN; fewer parameters; interpretable hidden state |
| Loss function | Ranking loss (Sultani et al., 2018) | Only video-level labels are available; ranking loss works without frame annotations |
| Class imbalance | WeightedRandomSampler | Without balancing, 76% normal clips push the model toward always predicting "normal" |
| Primary metric | AUC-ROC, not accuracy | At 76% normal, always predicting "normal" yields 76% accuracy — AUC is more informative |
| Split strategy | Video-level splits | Clip-level splits cause data leakage when clips from the same video appear in both train and val |
| Checkpoint criterion | Best validation AUC | Validation loss collapses to 0 after epoch 2; AUC remains meaningful throughout training |
| LR scheduler mode | `mode="max"` | Scheduler tracks AUC improvement rather than validation loss |
| YOLO placement | Flagged clips only | Running YOLO on every frame is slow and redundant once the LSTM has flagged anomalies |
| Grad-CAM target | L2 norm of the feature map | No classification logit is available; the norm captures the most active spatial regions |
| API pattern | Asynchronous job queue | Analysis takes 20–120 seconds; a synchronous endpoint would risk timing out |
| Frontend coupling | Vanilla JS, API-only | Keeps the UI fully decoupled from the ML code; any client can consume the API |

---

## Limitations

**Recall on fast-motion anomalies is limited.** Categories such as Fighting (0.6576), Shooting (0.6426), and Road Accidents (0.6481) score close to random chance. These events unfold over rapid motion across a small number of frames — a 16-frame LSTM clip at 30 fps spans only 0.5 seconds, often too little context to capture the event. A 3D CNN or transformer with a longer temporal window would likely perform better here.

**No frame-level labels during training.** The ranking loss works with video-level labels alone, but as a result the model does not learn precisely which frames are anomalous. At inference time it scores 16-frame clips rather than individual frames, and the reported timestamp corresponds to the start of the highest-scoring clip rather than the exact onset of the anomaly.

**In-memory job store.** Completed jobs — including all decoded video frames — remain in a Python dictionary until explicitly deleted. For long videos this can consume several gigabytes of RAM. A production system would stream frames to disk or object storage and evict stale jobs automatically.

**Single-GPU training assumption.** Both the training script and the feature extraction script assume a single GPU; multi-GPU training is not implemented.

**Validation–test gap.** Validation AUC (0.8881) exceeds test AUC (0.8030), suggesting some overfitting to the validation set introduced through the LR schedule — the scheduler reduces the learning rate when validation AUC plateaus, which implicitly fits to the validation signal. A held-out test set entirely excluded from training decisions would provide a cleaner estimate of generalization.

**YOLO labels are generic.** YOLOv8n is an 80-class COCO detector — it can identify "person" or "car," but has no crime-specific categories. Detections provide useful scene context but are not crime-aware.

**CPU inference is slow.** Processing a 60-second video takes roughly 8–12 minutes on CPU alone; a GPU reduces this to under 30 seconds.

---

## Hardware Used

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop (8 GB VRAM) |
| CUDA | 12.4 |
| OS | Windows 11 |
| PyTorch | 2.6.0+cu124 |

---

## References

- Sultani, W., Chen, C., & Shah, M. (2018). *Real-World Anomaly Detection in Surveillance Videos.* CVPR 2018. Introduced the UCF-Crime dataset and the ranking loss for weak supervision.
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV 2017.
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
