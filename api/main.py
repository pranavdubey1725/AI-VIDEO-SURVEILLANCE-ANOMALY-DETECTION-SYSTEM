"""
FastAPI backend for the AI Video Surveillance System.

Architecture:
    POST /analyze          — upload video, kick off background job
    GET  /jobs/{id}        — poll job status (queued/processing/done/failed)
    GET  /jobs/{id}/results — full analysis results as JSON
    GET  /jobs/{id}/clips/{idx}/frame   — original frame image for a clip
    GET  /jobs/{id}/clips/{idx}/heatmap — Grad-CAM overlay image for a clip
    GET  /health           — liveness check + model info

Why async + background tasks?
    Video analysis takes 20-120 seconds depending on length.
    A synchronous endpoint would time out in browsers and proxies.
    We start the job immediately, return a job_id, and let the client
    poll /jobs/{id} until status == "done".
"""

import sys
import uuid
import asyncio
import tempfile
import os
import io
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import asdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

JOB_TTL_SECONDS = 1800  # auto-expire jobs after 30 minutes

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image

from config import ANOMALY_THRESHOLD, CHECKPOINTS_DIR
from src.inference.pipeline import SurveillancePipeline, ClipResult, VideoResult

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_FILE_SIZE_MB   = 200   # HuggingFace Spaces free tier caps at ~200 MB


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Video Surveillance API",
    description="Anomaly detection in surveillance videos using ResNet50 + LSTM + YOLOv8",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Streamlit on localhost
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pipeline (singleton, loaded once at startup) ───────────────────────────────
pipeline: Optional[SurveillancePipeline] = None

@app.on_event("startup")
async def load_pipeline():
    global pipeline
    logger.info("Loading SurveillancePipeline...")
    pipeline = SurveillancePipeline()
    logger.info("Pipeline ready.")
    asyncio.create_task(_evict_expired_jobs())


async def _evict_expired_jobs():
    """Background loop: delete jobs older than JOB_TTL_SECONDS every 5 minutes."""
    while True:
        await asyncio.sleep(300)
        cutoff = time.time() - JOB_TTL_SECONDS
        expired = [jid for jid, j in list(jobs.items()) if j["created_at"] < cutoff]
        for jid in expired:
            jobs.pop(jid, None)
            logger.info("Evicted expired job %s", jid)
        if expired:
            logger.info("Evicted %d expired job(s)", len(expired))


# ── Job store (in-memory) ─────────────────────────────────────────────────────
class JobStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


jobs: Dict[str, Dict[str, Any]] = {}
# Structure per job:
# {
#   "status":   JobStatus,
#   "error":    str | None,
#   "result":   VideoResult | None,
#   "frames":   list[PIL.Image] | None,   # original frames
#   "filename": str,
#   "threshold": float,
# }


# ── Background worker ─────────────────────────────────────────────────────────
def _classify_pipeline_error(e: Exception) -> str:
    """Map raw exceptions to user-friendly error messages."""
    msg = str(e)
    if "Cannot open video" in msg:
        return "Video file could not be opened — it may be corrupt or in an unsupported codec."
    if "too short" in msg.lower():
        return msg
    if "out of memory" in msg.lower() or ("cuda" in msg.lower() and "memory" in msg.lower()):
        return "GPU ran out of memory. Try a shorter video or reduce the file size."
    if "no such file" in msg.lower():
        return "Internal error: temporary file was removed before processing completed."
    return f"Analysis failed: {msg}"


def run_analysis(job_id: str, video_path: str, threshold: float):
    """Runs in a thread pool via BackgroundTasks."""
    jobs[job_id]["status"] = JobStatus.PROCESSING

    try:
        pipeline.threshold = threshold

        # Extract frames (keep them for image endpoints)
        frames, fps, total_frames = pipeline._extract_frames(video_path)

        if len(frames) < 16:
            raise ValueError(f"Video too short: {len(frames)} frames (need 16+)")

        features    = pipeline._frames_to_features(frames)
        clip_scores = pipeline._score_clips(features)

        from src.inference.pipeline import ClipResult, VideoResult
        import time
        t0 = time.time()

        results   = []
        anomalous = []

        for idx, (start, end, score) in enumerate(clip_scores):
            r = ClipResult(
                clip_idx      = idx,
                start_frame   = start,
                end_frame     = end,
                timestamp_sec = start / fps,
                anomaly_score = score,
                is_anomalous  = score >= threshold,
            )
            results.append(r)
            if r.is_anomalous:
                anomalous.append(r)

        # Run YOLO + Grad-CAM only on top 5 clips by score — each Grad-CAM
        # backward pass takes ~2-5s on CPU; running it on all clips is too slow.
        top_anomalous = sorted(anomalous, key=lambda r: r.anomaly_score, reverse=True)[:5]
        for result in top_anomalous:
            mid   = min((result.start_frame + result.end_frame) // 2, len(frames) - 1)
            frame = frames[mid]
            result.detections    = pipeline.detector.detect(frame)
            result.explanation   = pipeline.detector.summarize(result.detections)
            hm, ov               = pipeline.gradcam.compute_and_overlay(frame)
            result.heatmap       = hm
            result.overlay_image = ov

        video_result = VideoResult(
            video_path      = video_path,
            fps             = fps,
            total_frames    = len(frames),
            duration_sec    = len(frames) / fps,
            clips_analyzed  = len(clip_scores),
            anomalous_clips = len(anomalous),
            max_score       = max((s for _, _, s in clip_scores), default=0.0),
            results         = results,
            processing_time = time.time() - t0,
        )

        jobs[job_id]["result"] = video_result
        jobs[job_id]["frames"] = frames
        jobs[job_id]["status"] = JobStatus.DONE

    except Exception as e:
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"]  = _classify_pipeline_error(e)
        logger.exception("Job %s failed: %s", job_id, e)
    finally:
        # Clean up temp file
        if os.path.exists(video_path):
            os.unlink(video_path)


# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


def result_to_dict(result: VideoResult) -> dict:
    """Serialise VideoResult to JSON-safe dict."""
    clips = []
    for r in result.results:
        clips.append({
            "clip_idx":      r.clip_idx,
            "start_frame":   r.start_frame,
            "end_frame":     r.end_frame,
            "timestamp_sec": round(r.timestamp_sec, 3),
            "anomaly_score": round(r.anomaly_score, 4),
            "is_anomalous":  r.is_anomalous,
            "explanation":   r.explanation,
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 3),
                    "bbox":       list(d.bbox),
                }
                for d in r.detections
            ],
        })

    return {
        "video_path":       result.video_path,
        "fps":              result.fps,
        "total_frames":     result.total_frames,
        "duration_sec":     round(result.duration_sec, 2),
        "clips_analyzed":   result.clips_analyzed,
        "anomalous_clips":  result.anomalous_clips,
        "max_score":        round(result.max_score, 4),
        "processing_time":  round(result.processing_time, 2),
        "clips":            clips,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":    "ok",
        "pipeline":  pipeline is not None,
        "model":     "ResNet50 + LSTM + YOLOv8n",
        "val_auc":   0.8881,
        "test_auc":  0.8030,
    }


@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    file:      UploadFile = File(...),
    threshold: float      = Form(default=ANOMALY_THRESHOLD),
):
    """Upload a video and start background analysis. Returns a job_id to poll."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not loaded yet — try again in a few seconds")

    # Threshold bounds
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(400, f"threshold must be between 0.0 and 1.0, got {threshold}")

    # File extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f'Unsupported file type "{suffix}". Accepted: mp4, avi, mov, mkv',
        )

    # Read and validate content
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Uploaded file is empty")
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            413,
            f"File too large ({len(content) / 1024 / 1024:.0f} MB). Maximum: {MAX_FILE_SIZE_MB} MB",
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status":     JobStatus.QUEUED,
        "error":      None,
        "result":     None,
        "frames":     None,
        "filename":   file.filename,
        "threshold":  threshold,
        "created_at": time.time(),
    }
    logger.info("Job %s queued — file=%s threshold=%.2f", job_id, file.filename, threshold)

    background_tasks.add_task(run_analysis, job_id, tmp.name, threshold)

    return {"job_id": job_id, "status": JobStatus.QUEUED, "filename": file.filename}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    """Poll job status. When status == 'done', fetch results from /jobs/{id}/results."""
    if job_id not in jobs:
        raise HTTPException(404, f"Job {job_id} not found")

    job = jobs[job_id]
    response = {
        "job_id":    job_id,
        "status":    job["status"],
        "filename":  job["filename"],
        "threshold": job["threshold"],
    }
    if job["status"] == JobStatus.FAILED:
        response["error"] = job["error"]
    if job["status"] == JobStatus.DONE and job["result"]:
        r = job["result"]
        response["summary"] = {
            "duration_sec":    round(r.duration_sec, 2),
            "clips_analyzed":  r.clips_analyzed,
            "anomalous_clips": r.anomalous_clips,
            "max_score":       round(r.max_score, 4),
            "processing_time": round(r.processing_time, 2),
        }
    return response


@app.get("/jobs/{job_id}/results")
def job_results(job_id: str):
    """Full analysis results as JSON — only available when status == 'done'."""
    if job_id not in jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    job = jobs[job_id]
    if job["status"] != JobStatus.DONE:
        raise HTTPException(400, f"Job not done yet (status={job['status']})")
    return result_to_dict(job["result"])


@app.get("/jobs/{job_id}/clips/{clip_idx}/frame")
def clip_frame(job_id: str, clip_idx: int):
    """Return the representative frame (JPEG) for a specific clip."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != JobStatus.DONE:
        raise HTTPException(400, "Job not done yet")

    result  = job["result"]
    frames  = job["frames"]
    clips   = [r for r in result.results if r.clip_idx == clip_idx]
    if not clips:
        raise HTTPException(404, f"Clip {clip_idx} not found")

    clip    = clips[0]
    mid_idx = min((clip.start_frame + clip.end_frame) // 2, len(frames) - 1)
    img     = frames[mid_idx]

    return StreamingResponse(io.BytesIO(pil_to_bytes(img)), media_type="image/jpeg")


@app.get("/jobs/{job_id}/clips/{clip_idx}/heatmap")
def clip_heatmap(job_id: str, clip_idx: int):
    """Return the Grad-CAM overlay (JPEG) for an anomalous clip."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != JobStatus.DONE:
        raise HTTPException(400, "Job not done yet")

    result = job["result"]
    clips  = [r for r in result.results if r.clip_idx == clip_idx]
    if not clips:
        raise HTTPException(404, f"Clip {clip_idx} not found")

    clip = clips[0]
    if clip.overlay_image is None:
        raise HTTPException(404, "No heatmap for this clip (not anomalous or not computed)")

    return StreamingResponse(
        io.BytesIO(pil_to_bytes(clip.overlay_image)), media_type="image/jpeg"
    )


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Free memory for a completed job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    del jobs[job_id]
    return {"deleted": job_id}


# ── Static UI (must be last — catches all unmatched routes) ───────────────────
from fastapi.staticfiles import StaticFiles
_ui_dir = Path(__file__).parent.parent / "ui"
if _ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(_ui_dir), html=True), name="static")
