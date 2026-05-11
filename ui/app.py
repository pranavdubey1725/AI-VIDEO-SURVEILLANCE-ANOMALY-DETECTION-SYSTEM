"""
Streamlit frontend for the AI Video Surveillance System.
Communicates with the FastAPI backend at http://localhost:8000.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import requests
import time
import pandas as pd
from PIL import Image
import io

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Surveillance — Anomaly Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎥 AI Surveillance")
    st.markdown("**Anomaly Detection System**")
    st.divider()

    # API health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API connected")
        st.markdown(f"**Val AUC:** `{health.get('val_auc', '—')}`")
        st.markdown(f"**Test AUC:** `{health.get('test_auc', '—')}`")
    except Exception:
        st.error("API not reachable — start the FastAPI server first:\n\n"
                 "`uvicorn api.main:app --reload`")

    st.divider()
    st.markdown("### Settings")
    threshold = st.slider(
        "Anomaly threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Clips scoring above this are flagged as anomalous"
    )

    st.divider()
    st.markdown("### Model")
    st.markdown("""
    - **Feature extractor**: ResNet50
    - **Temporal model**: LSTM (2×256)
    - **Object detector**: YOLOv8n
    - **Explainability**: Grad-CAM
    """)

    st.divider()
    st.markdown("### Dataset")
    st.markdown("""
    **UCF-Crime**
    - 1,900 surveillance videos
    - 13 anomaly categories
    - 128 hours of footage
    """)


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("AI Video Surveillance — Anomaly Detection")
st.markdown(
    "Upload a surveillance video. The system analyzes it clip by clip, "
    "scores each segment, and shows you exactly what it found — "
    "with Grad-CAM heatmaps and YOLO object detections."
)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded is None:
    st.info("Upload a video to begin.")
    st.markdown("---")
    st.markdown("#### Detectable anomaly categories")
    cats = [
        ("🥊", "Fighting"),      ("🔫", "Shooting"),
        ("🔪", "Assault"),       ("🏃", "Robbery"),
        ("🔥", "Arson"),         ("💥", "Explosion"),
        ("🛒", "Shoplifting"),   ("🚗", "Road Accidents"),
        ("🏠", "Burglary"),      ("📦", "Stealing"),
        ("🚔", "Arrest"),        ("😤", "Abuse"),
        ("💢", "Vandalism"),
    ]
    cols = st.columns(4)
    for i, (icon, name) in enumerate(cats):
        cols[i % 4].markdown(f"{icon} **{name}**")
    st.stop()

file_mb = uploaded.size / 1024 / 1024
st.markdown(f"**File:** `{uploaded.name}`  |  **Size:** {file_mb:.1f} MB")

if file_mb > 500:
    st.error("File exceeds the 500 MB limit. Please upload a shorter video.")
    st.stop()
if file_mb > 200:
    st.warning(f"Large file ({file_mb:.0f} MB) — analysis may take several minutes.")

if not st.button("Analyze Video", type="primary", use_container_width=True):
    st.stop()

# ── Submit job ────────────────────────────────────────────────────────────────
with st.spinner("Submitting video to API..."):
    try:
        resp = requests.post(
            f"{API_URL}/analyze",
            files={"file": (uploaded.name, uploaded.getvalue(), "video/mp4")},
            data={"threshold": threshold},
            timeout=30,
        )
        resp.raise_for_status()
        job = resp.json()
        job_id = job["job_id"]
    except Exception as e:
        st.error(f"Failed to submit video: {e}")
        st.stop()

st.info(f"Job submitted — ID: `{job_id}`")

# ── Poll until done ───────────────────────────────────────────────────────────
progress = st.progress(0, text="Queued...")
status_box = st.empty()

fake_progress = 0
while True:
    try:
        status_resp = requests.get(f"{API_URL}/jobs/{job_id}", timeout=5).json()
    except Exception:
        time.sleep(2)
        continue

    s = status_resp["status"]

    if s == "queued":
        fake_progress = min(fake_progress + 2, 10)
        progress.progress(fake_progress, text="Queued — waiting for pipeline...")

    elif s == "processing":
        fake_progress = min(fake_progress + 3, 85)
        progress.progress(fake_progress, text="Processing — analyzing clips...")

    elif s == "done":
        progress.progress(100, text="Done!")
        summary = status_resp.get("summary", {})
        break

    elif s == "failed":
        st.error(f"Analysis failed: {status_resp.get('error', 'unknown error')}")
        st.stop()

    time.sleep(1.5)

# ── Fetch full results ────────────────────────────────────────────────────────
results_resp = requests.get(f"{API_URL}/jobs/{job_id}/results", timeout=10).json()
clips        = results_resp["clips"]

# ── Metrics ───────────────────────────────────────────────────────────────────
st.success(f"Analysis complete in {summary.get('processing_time', 0):.1f}s")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Duration",        f"{summary.get('duration_sec', 0):.1f}s")
c2.metric("Clips analyzed",  f"{summary.get('clips_analyzed', 0):,}")
c3.metric("Anomalous clips", f"{summary.get('anomalous_clips', 0)}",
          delta_color="inverse",
          delta=f"{summary.get('anomalous_clips',0)/max(summary.get('clips_analyzed',1),1)*100:.0f}%")
c4.metric("Peak score",      f"{summary.get('max_score', 0):.3f}")

st.divider()

# ── Anomaly timeline chart ────────────────────────────────────────────────────
st.subheader("Anomaly Score Timeline")

import plotly.graph_objects as go

fig = go.Figure()
times  = [c["timestamp_sec"] for c in clips]
scores = [c["anomaly_score"]  for c in clips]
flagged = [c["is_anomalous"]  for c in clips]

clip_duration = (clips[1]["timestamp_sec"] - clips[0]["timestamp_sec"]) if len(clips) > 1 else 1.0

for c in clips:
    if c["is_anomalous"]:
        fig.add_vrect(
            x0=c["timestamp_sec"], x1=c["timestamp_sec"] + clip_duration,
            fillcolor="red", opacity=0.12, line_width=0,
        )

fig.add_trace(go.Scatter(
    x=times, y=scores, mode="lines",
    line=dict(color="royalblue", width=2),
    fill="tozeroy", fillcolor="rgba(65,105,225,0.15)",
    name="Anomaly Score",
))
fig.add_hline(y=threshold, line_dash="dash", line_color="red",
              annotation_text=f"Threshold ({threshold})",
              annotation_position="top right")
fig.update_layout(
    xaxis_title="Time (seconds)", yaxis_title="Anomaly Score",
    yaxis=dict(range=[0, 1.05]), height=300,
    margin=dict(l=0, r=0, t=20, b=0), showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Flagged clips ─────────────────────────────────────────────────────────────
anomalous_clips = sorted(
    [c for c in clips if c["is_anomalous"]],
    key=lambda c: c["anomaly_score"], reverse=True
)

if not anomalous_clips:
    st.success("No anomalies detected above the threshold.")
else:
    st.subheader(f"Flagged Clips  ({len(anomalous_clips)} found)")
    st.caption(
        "Sorted by anomaly score (highest first). "
        "Left: original frame. Middle: Grad-CAM heatmap (red = model focus). "
        "Right: YOLO detections."
    )

    for i, clip in enumerate(anomalous_clips):
        idx  = clip["clip_idx"]
        label = (
            f"⚠️  {clip['timestamp_sec']:.1f}s  |  "
            f"Score: {clip['anomaly_score']:.4f}  |  "
            f"{clip['explanation'] or 'No objects detected'}"
        )

        with st.expander(label, expanded=(i < 3)):
            col1, col2, col3 = st.columns(3)

            # Original frame from API
            with col1:
                st.markdown("**Original Frame**")
                frame_resp = requests.get(
                    f"{API_URL}/jobs/{job_id}/clips/{idx}/frame", timeout=10
                )
                if frame_resp.status_code == 200:
                    img = Image.open(io.BytesIO(frame_resp.content))
                    st.image(img, use_container_width=True)

            # Grad-CAM heatmap from API
            with col2:
                st.markdown("**Grad-CAM Heatmap**")
                hm_resp = requests.get(
                    f"{API_URL}/jobs/{job_id}/clips/{idx}/heatmap", timeout=10
                )
                if hm_resp.status_code == 200:
                    hm_img = Image.open(io.BytesIO(hm_resp.content))
                    st.image(hm_img, use_container_width=True)
                else:
                    st.info("Heatmap not available")

            # Detection details
            with col3:
                st.markdown("**Detection Details**")
                st.markdown(f"**Timestamp:** `{clip['timestamp_sec']:.2f}s`")
                st.markdown(f"**Score:** `{clip['anomaly_score']:.4f}`")
                st.markdown(f"**Frames:** `{clip['start_frame']} – {clip['end_frame']}`")
                st.divider()
                dets = clip.get("detections", [])
                if dets:
                    st.markdown("**YOLO Detections:**")
                    for d in dets[:6]:
                        bar = "█" * int(d["confidence"] * 20) + "░" * (20 - int(d["confidence"] * 20))
                        st.markdown(f"`{d['class_name']:<12}` {bar} `{d['confidence']:.2f}`")
                else:
                    st.info("No objects detected by YOLO")

st.divider()
st.caption(
    "AI Video Surveillance System — "
    "ResNet50 + LSTM (UCF-Crime, Test AUC 0.8030) + YOLOv8n + Grad-CAM | "
    "FastAPI + Streamlit"
)
