# AI Video Surveillance — Anomaly Detection

An end-to-end deep learning system that detects anomalous events in surveillance videos.
Upload a video and get back timestamped anomaly scores, object detections, and Grad-CAM
heatmaps showing exactly which regions triggered each alert.

Trained on the UCF-Crime benchmark. Test AUC-ROC: **0.8030** — beats the original
Sultani et al. 2018 paper (0.7510) by approximately 5 percentage points.

---

## Results

| Metric | Value |
|---|---|
| Validation AUC-ROC | 0.8881 |
| **Test AUC-ROC** | **0.8030** |
| Accuracy at threshold 0.5 | 80% |
| Anomaly recall | 69% |
| Sultani et al. 2018 baseline | 0.7510 |

### Per-Category Test AUC (each category vs Normal)

| Category | AUC | Category | AUC |
|---|---|---|---|
| Burglary | 0.9298 | Robbery | 0.8176 |
| Vandalism | 0.9187 | Stealing | 0.7481 |
| Assault | 0.8831 | Abuse | 0.7362 |
| Arson | 0.8737 | Arrest | 0.7340 |
| Shoplifting | 0.8636 | Fighting | 0.6576 |
| Explosion | 0.8257 | Road Accidents | 0.6481 |
| Robbery | 0.8176 | Shooting | 0.6426 |

ROC curve and score distribution plots are in `outputs/`.

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
                         |
                         v
             [Streamlit Frontend]  :8501
```

---

## How Each Component Works

### Feature Extractor — ResNet50 (frozen)

Each video frame is resized to 224x224 and passed through a pretrained ResNet50,
producing a 2048-dimensional feature vector. The weights are frozen — no fine-tuning.
This is the standard transfer-learning approach: ImageNet features capture texture,
edges, and object shapes well enough to distinguish normal from anomalous activity
without retraining on surveillance data.

Pre-computing features for the entire dataset takes about 2 hours on a GPU. At inference
time features are computed on-the-fly per video.

### Temporal Model — AnomalyLSTM

```
Input:  [batch, 16 frames, 2048 features]
LSTM:   2 layers, hidden size 256, dropout 0.5
Output: [batch, 1]  — anomaly score in [0, 1]
```

The 16-frame clip is a sliding window over the video (stride 8, so consecutive clips
overlap by half). The LSTM reads the sequence of frame features and produces one score
per clip. The final hidden state of the top LSTM layer is passed through a small
fully-connected head (256 -> 64 -> 1) with sigmoid activation.

Why LSTM and not a 3D CNN? LSTMs are simpler to train, require less memory, and are
easier to explain in an interview. A 3D CNN would likely give higher recall on fast
motion anomalies (fights, explosions) but would be harder to debug and slower to
iterate on.

### Loss Function — Ranking Loss (Sultani 2018)

The UCF-Crime dataset only provides video-level labels: a video is either "anomalous"
or "normal" — there are no frame-level annotations in the training set. This is called
weak supervision.

The ranking loss exploits this: instead of predicting exact scores, it only requires
that anomalous clips score higher than normal clips by a margin of 1.

```
loss = mean( max(0, 1 - score_anomalous + score_normal) )
```

Every anomalous score in a batch is compared against every normal score. If the
anomalous score already exceeds the normal score by at least 1, the loss is zero for
that pair. This is the core idea from the paper that introduced UCF-Crime.

### Object Detector — YOLOv8n

YOLOv8n runs only on the representative frame of clips that the LSTM flagged as
anomalous. Running it on every frame would be redundant and slow — the LSTM already
handles temporal anomaly detection. YOLO adds interpretability: it tells you what
objects were visible in a flagged clip (persons, vehicles, weapons if labeled).

### Explainability — Grad-CAM

Grad-CAM computes which spatial regions of the input frame the ResNet50 responded to
most strongly. It does this by computing gradients of the L2 norm of the feature map
(layer4 output) with respect to the feature map activations, then weighting the
activation maps by those gradients and producing a heatmap.

Note: the standard Grad-CAM target is a class logit. Since our ResNet50 has no
classification head (we stripped it to get the 2048-dim feature vector), we use the L2
norm of the feature map as a proxy. This produces visually plausible heatmaps but is
not as theoretically grounded as class-discriminative Grad-CAM.

### API — FastAPI with async job queue

Video analysis takes 20–120 seconds depending on length. A synchronous HTTP endpoint
would time out in browsers and proxies. Instead:

1. `POST /analyze` saves the video to a temp file, starts a background thread, and
   immediately returns a `job_id`.
2. The client polls `GET /jobs/{id}` until `status == "done"`.
3. `GET /jobs/{id}/results` returns the full JSON report.
4. Frame and heatmap images are served as JPEG streams via dedicated endpoints.
5. `DELETE /jobs/{id}` frees the in-memory frames.

Jobs are stored in a plain Python dict (in-memory). If the server restarts, all jobs
are lost. This is acceptable for a local development system; a production deployment
would use Redis or a database.

### Frontend — Streamlit

The Streamlit UI only talks to the FastAPI backend. It never touches the ML models
directly. This decoupling means the API can serve any client (mobile, CLI, another
service) without changing the ML code.

---

## Dataset

**UCF-Crime** (`odins0n/ucf-crime-dataset` on Kaggle)

- 1,900 surveillance videos, 128 hours total
- 13 anomaly categories: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting,
  Road Accidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism
- Plus a Normal class
- Training labels are video-level only (no frame annotations)
- Test set has frame-level binary labels, used only for evaluation

| Split | Videos | Clips | Normal fraction |
|---|---|---|---|
| Train | 1,285 | 134,812 | 76.2% |
| Validation | 322 | 21,174 | 68.8% |
| Test | 290 | 13,494 | 58.5% |

Splits are video-level (not clip-level). Splitting at the clip level would cause data
leakage: clips from the same video would appear in both train and validation, inflating
validation AUC by several points.

---

## Training Details

| Setting | Value |
|---|---|
| Optimizer | Adam, lr=1e-4 |
| LR schedule | ReduceLROnPlateau (mode=max, patience=5, factor=0.5) |
| Gradient clipping | max_norm=1.0 |
| Epochs | 50 (best checkpoint saved at epoch 29) |
| Batch size | 32 |
| Class imbalance | WeightedRandomSampler (76% of clips are Normal) |
| Training time | ~90 minutes on RTX 4060 Laptop (8GB VRAM) |

LR schedule progression:

| Learning rate | Best validation AUC |
|---|---|
| 1e-4 | 0.8646 |
| 2.5e-5 | 0.8862 |
| 1.25e-5 | **0.8881** (saved as best_model.pt) |

The scheduler is set to `mode="max"` to monitor AUC rather than loss. Validation loss
collapses to 0.0 after epoch 2 (a known artifact of the ranking loss with class
imbalance), so monitoring it would stop the LR from decaying at all.

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
|   +-- app.py                   Streamlit frontend — polls API, renders results
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
- NVIDIA GPU with CUDA 12.4 recommended (CPU works but inference is slow)

### 1. Clone and install

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

For CPU-only (no GPU), install the default PyTorch:

```bash
pip install torch torchvision torchaudio
```

Then open `config.py` and change `DEVICE = "cuda"` to `DEVICE = "cpu"`.

### 2. Start both servers

```bash
python run.py
```

This launches the FastAPI backend on port 8000 and the Streamlit UI on port 8501.
Open `http://localhost:8501` in your browser. The `best_model.pt` checkpoint is
included — no retraining required.

### 3. Analyze a video

- Upload any `.mp4`, `.avi`, `.mov`, or `.mkv` file
- Adjust the anomaly threshold slider (default 0.5)
- Click **Analyze Video**
- View the anomaly score timeline, flagged clips, Grad-CAM heatmaps, and YOLO detections

---

## Running with Docker

Docker runs the API and UI as two separate containers on a shared network. The UI
container reaches the API via the Docker service name (`http://api:8000`) rather than
localhost. This is handled automatically through an environment variable — no code
changes are needed for local development.

### Prerequisites

- Docker and Docker Compose (V2)
- For GPU acceleration: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Build and start

```bash
docker compose up --build
```

First build takes several minutes because PyTorch (~2.5 GB) is downloaded and installed
inside the image. Subsequent starts reuse the cached layer.

Open `http://localhost:8501` in your browser.

### CPU-only (no GPU)

Remove the `deploy` block from the `api` service in `docker-compose.yml` and change the
`DEVICE` environment variable to `cpu`:

```yaml
environment:
  - DEVICE=cpu
```

CPU inference works but is significantly slower (8–12 minutes per 60-second video vs
under 30 seconds with a GPU).

### Stop

```bash
docker compose down
```

### Image size note

The Docker image is large (~6–8 GB) because of the PyTorch CUDA wheels. This is
expected for GPU-enabled deep learning containers. The `.dockerignore` file ensures the
11.8 GB dataset, pre-computed features, and intermediate checkpoints are never copied
into the image.

---

## API Reference

Interactive docs are available at `http://localhost:8000/docs` while the server is
running.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check and model metadata |
| POST | `/analyze` | Upload video, start background analysis job |
| GET | `/jobs/{id}` | Poll job status: queued / processing / done / failed |
| GET | `/jobs/{id}/results` | Full results JSON (only when status is done) |
| GET | `/jobs/{id}/clips/{idx}/frame` | Representative frame as JPEG |
| GET | `/jobs/{id}/clips/{idx}/heatmap` | Grad-CAM overlay as JPEG |
| DELETE | `/jobs/{id}` | Free memory for a completed job |

### Error codes

| Status | Meaning |
|---|---|
| 400 | Bad input — invalid threshold, unsupported file type, or empty file |
| 413 | File too large (above 500 MB) |
| 422 | Missing required field |
| 503 | Pipeline not loaded yet — retry in a few seconds |

---

## Tests

```bash
# Requires both servers running (python run.py)
cd surveillance-system
pytest tests/test_api.py -v
```

14 automated tests covering:

- Health endpoint returns correct model metadata
- 404 responses for unknown job IDs (status, results, frame, heatmap, delete)
- Input validation: missing file, wrong extension, empty file, threshold out of range
- Too-short video (under 16 frames) fails gracefully with a readable error message
- Full job lifecycle: submit, poll, fetch results, fetch frame image, delete, confirm gone

The tests generate a synthetic 10-frame video using OpenCV rather than requiring a real
surveillance video. The full lifecycle test (`test_full_job_lifecycle`) is skipped unless
a file at `data/test_video.mp4` is present.

---

## Retraining from Scratch

```bash
# 1. Download dataset (requires Kaggle API credentials)
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

| Decision | Choice | Reason |
|---|---|---|
| Feature extractor | ResNet50 frozen | 500x faster than on-the-fly CNN training; ImageNet features transfer well |
| Temporal model | LSTM, 2 layers, hidden 256 | Simpler than 3D CNN; fewer parameters; interpretable hidden state |
| Loss function | Ranking loss (Sultani 2018) | Only video-level labels available; ranking loss works without frame annotations |
| Class imbalance | WeightedRandomSampler | Without balancing, 76% normal clips cause the model to predict "normal" always |
| Primary metric | AUC-ROC, not accuracy | At 76% normal, always predicting normal gives 76% accuracy — AUC is informative |
| Split strategy | Video-level splits | Clip-level splits cause data leakage when clips from the same video appear in train and val |
| Checkpoint criterion | Best validation AUC | Validation loss collapses to 0 after epoch 2; AUC remains meaningful throughout |
| LR scheduler mode | mode="max" | Scheduler monitors AUC improvement, not validation loss |
| YOLO placement | Only on flagged clips | Running YOLO on every frame is slow and redundant when LSTM already flags anomalies |
| Grad-CAM target | L2 norm of feature map | No classification logit available; norm captures which spatial regions activate most |
| API pattern | Async job queue | Analysis takes 20–120 seconds; synchronous endpoints would time out |
| Frontend coupling | Streamlit calls API only | Keeps UI completely decoupled from ML; any client can use the API |

---

## Shortcomings and Limitations

**Recall on fast-motion anomalies is low.** Categories like Fighting (0.6576), Shooting
(0.6426), and Road Accidents (0.6481) have AUC scores close to random. These events
involve rapid motion across a small number of frames. A 16-frame LSTM clip at 30 fps
covers only 0.5 seconds, which is often not enough context. A 3D CNN or transformer
with longer temporal context would likely perform better here.

**No frame-level labels during training.** The ranking loss works with video-level labels
but it does not learn exactly which frames are anomalous. At inference time the model
scores clips (16-frame windows), not individual frames. The reported timestamps are the
start time of the highest-scoring clip, not the precise moment the anomaly begins.

**In-memory job store.** Completed jobs (including all decoded video frames) are held in
a Python dict until explicitly deleted. For long videos this can use several GB of RAM.
A production system would stream frames to disk or object storage and evict old jobs
automatically.

**Single-GPU training assumption.** The training script and feature extraction script
assume one GPU. Multi-GPU training is not implemented.

**Val-test gap.** Validation AUC (0.8881) is higher than test AUC (0.8030). The gap
suggests some overfitting to the validation set through the LR schedule (the scheduler
reduces LR when validation AUC stops improving, which implicitly fits to validation
signal). A held-out test set that was never used during training decisions would give a
cleaner estimate.

**YOLO labels are generic.** YOLOv8n is an 80-class COCO detector. It can detect
"person" and "car" but has no crime-specific categories. Detections provide basic
scene context but are not crime-aware.

**CPU inference is slow.** Processing a 60-second video takes roughly 8–12 minutes on
CPU only. A GPU reduces this to under 30 seconds.

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

- Sultani, W., Chen, C., & Shah, M. (2018). **Real-world anomaly detection in surveillance
  videos**. CVPR 2018. Introduced the UCF-Crime dataset and the ranking loss for weak
  supervision.
- Selvaraju, R. R., et al. (2017). **Grad-CAM: Visual explanations from deep networks via
  gradient-based localization**. ICCV 2017.
- He, K., et al. (2016). **Deep residual learning for image recognition**. CVPR 2016.
