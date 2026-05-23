"""
Automated test suite for the AI Video Surveillance API.

Requires the FastAPI server to be running:
    python run.py           (or)
    uvicorn api.main:app --reload

Run:
    cd surveillance-system
    pytest tests/test_api.py -v
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests

BASE = "http://localhost:8000"
TEST_VIDEO = Path(__file__).parent.parent / "data" / "test_video.mp4"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory):
    """Create a valid but too-short MP4 (10 frames — below the 16-frame minimum)."""
    path = str(tmp_path_factory.mktemp("videos") / "tiny.mp4")
    out  = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (224, 224))
    for _ in range(10):
        out.write(np.zeros((224, 224, 3), dtype=np.uint8))
    out.release()
    return path


def _poll(job_id: str, timeout: int = 60):
    """Poll /jobs/{id} until status is 'done' or 'failed'. Returns final status dict."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{BASE}/jobs/{job_id}", timeout=5)
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("done", "failed"):
            return data
        time.sleep(1)
    pytest.fail(f"Job {job_id} did not finish within {timeout}s")


# ── Health (1 test) ───────────────────────────────────────────────────────────

def test_health():
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["pipeline"] is True
    assert "val_auc" in d
    assert "test_auc" in d


# ── 404 on unknown job (5 tests) ──────────────────────────────────────────────

def test_job_status_not_found():
    assert requests.get(f"{BASE}/jobs/no-such-job").status_code == 404


def test_job_results_not_found():
    assert requests.get(f"{BASE}/jobs/no-such-job/results").status_code == 404


def test_clip_frame_not_found():
    assert requests.get(f"{BASE}/jobs/no-such-job/clips/0/frame").status_code == 404


def test_clip_heatmap_not_found():
    assert requests.get(f"{BASE}/jobs/no-such-job/clips/0/heatmap").status_code == 404


def test_delete_not_found():
    assert requests.delete(f"{BASE}/jobs/no-such-job").status_code == 404


# ── Input validation — /analyze (6 tests) ────────────────────────────────────

def test_analyze_no_file():
    # FastAPI rejects missing required File() with 422
    assert requests.post(f"{BASE}/analyze").status_code == 422


def test_analyze_wrong_extension():
    r = requests.post(
        f"{BASE}/analyze",
        files={"file": ("notes.txt", b"not a video", "text/plain")},
    )
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["detail"]


def test_analyze_empty_file():
    r = requests.post(
        f"{BASE}/analyze",
        files={"file": ("clip.mp4", b"", "video/mp4")},
    )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_analyze_threshold_above_one():
    r = requests.post(
        f"{BASE}/analyze",
        files={"file": ("clip.mp4", b"x", "video/mp4")},
        data={"threshold": "1.5"},
    )
    # Threshold check fires before extension/content check
    assert r.status_code == 400
    assert "threshold" in r.json()["detail"].lower()


def test_analyze_threshold_negative():
    r = requests.post(
        f"{BASE}/analyze",
        files={"file": ("clip.mp4", b"x", "video/mp4")},
        data={"threshold": "-0.1"},
    )
    assert r.status_code == 400
    assert "threshold" in r.json()["detail"].lower()


def test_analyze_threshold_boundary_values():
    # Exact boundaries 0.0 and 1.0 are valid — should not return 400 for threshold
    # (may fail later for other reasons, but not threshold validation)
    for t in ("0.0", "1.0"):
        r = requests.post(
            f"{BASE}/analyze",
            files={"file": ("clip.mp4", b"", "video/mp4")},
            data={"threshold": t},
        )
        # Empty file → 400 "empty", NOT a threshold error
        assert r.status_code == 400
        assert "threshold" not in r.json()["detail"].lower()


# ── Too-short video → job fails gracefully (1 test) ──────────────────────────

def test_analyze_too_short(tiny_video):
    with open(tiny_video, "rb") as f:
        r = requests.post(
            f"{BASE}/analyze",
            files={"file": ("tiny.mp4", f, "video/mp4")},
        )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    status = _poll(job_id, timeout=30)
    assert status["status"] == "failed"
    assert "short" in status["error"].lower()


# ── Full job lifecycle (1 test, covers 5 sub-assertions) ─────────────────────

@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_video.mp4 not found")
def test_full_job_lifecycle():
    # 1. Submit
    with open(TEST_VIDEO, "rb") as f:
        r = requests.post(
            f"{BASE}/analyze",
            files={"file": ("test.mp4", f, "video/mp4")},
            data={"threshold": "0.5"},
            timeout=30,
        )
    assert r.status_code == 200
    body   = r.json()
    job_id = body["job_id"]
    assert body["status"] == "queued"

    # 2. Results-before-done → 400 (only if job hasn't already finished — tiny test
    #    videos can complete before this line runs, which is fine)
    if requests.get(f"{BASE}/jobs/{job_id}").json()["status"] != "done":
        assert requests.get(f"{BASE}/jobs/{job_id}/results").status_code == 400

    # 3. Poll to completion
    status = _poll(job_id, timeout=120)
    assert status["status"] == "done", f"Job failed: {status.get('error')}"
    summary = status["summary"]
    assert summary["clips_analyzed"] > 0
    assert 0.0 <= summary["max_score"] <= 1.0

    # 4. Results shape
    results = requests.get(f"{BASE}/jobs/{job_id}/results", timeout=10).json()
    assert "clips" in results
    clips = results["clips"]
    assert len(clips) == summary["clips_analyzed"]
    first = clips[0]
    assert 0.0 <= first["anomaly_score"] <= 1.0
    assert isinstance(first["is_anomalous"], bool)
    assert "timestamp_sec" in first
    assert "detections" in first

    # 5. Frame image for clip 0
    r = requests.get(f"{BASE}/jobs/{job_id}/clips/0/frame", timeout=10)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert len(r.content) > 500

    # 6. Invalid clip index → 404
    assert requests.get(f"{BASE}/jobs/{job_id}/clips/99999/frame").status_code == 404

    # 7. Delete
    r = requests.delete(f"{BASE}/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["deleted"] == job_id

    # 8. Confirm gone
    assert requests.get(f"{BASE}/jobs/{job_id}").status_code == 404
