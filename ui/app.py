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

import os
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Surveillance — Anomaly Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Apply global font */
html, body, [class*="css"]  {
    font-family: 'Outfit', sans-serif;
}

/* Main background */
.stApp {
    background-color: #0b0f19;
    background-image: radial-gradient(circle at 50% 0%, #1e293b 0%, #0b0f19 70%);
    color: #e2e8f0;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.7) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Typography overrides */
h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}
p, span, div {
    color: #cbd5e1;
}

/* Cards for metrics */
div[data-testid="metric-container"] {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    border: 1px solid rgba(99, 102, 241, 0.4);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(99, 102, 241, 0.2);
}
div[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
}

/* Button styling */
div.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    color: white !important;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39);
    transition: all 0.3s ease;
    width: 100%;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
    transform: translateY(-2px);
    color: white !important;
}

/* File uploader styling */
[data-testid="stFileUploadDropzone"] {
    background-color: rgba(30, 41, 59, 0.5);
    border: 2px dashed rgba(99, 102, 241, 0.4);
    border-radius: 16px;
    padding: 40px;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #6366f1;
    background-color: rgba(30, 41, 59, 0.8);
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.1);
}

/* Dividers */
hr {
    border-color: rgba(255, 255, 255, 0.1) !important;
}

/* Expanders (flagged clips) */
[data-testid="stExpander"] {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    overflow: hidden;
}
[data-testid="stExpander"] > summary {
    background: rgba(30, 41, 59, 0.8);
    padding: 15px 20px;
    font-weight: 600;
}
[data-testid="stExpander"] > summary:hover {
    color: #6366f1 !important;
}

/* Progress Bar */
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #6366f1, #a855f7, #ec4899);
}

/* Expander inner content formatting */
div[data-testid="stExpanderDetails"] {
    background: transparent;
    padding: 20px;
}

/* Image borders */
img {
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Status Alerts */
[data-testid="stAlert"] {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #cbd5e1 !important;
}
</style>
""", unsafe_allow_html=True)

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
st.markdown("""
<div style="text-align: center; padding: 3rem 0 1rem 0;">
    <h1 style="font-size: 3.5rem; font-weight: 700; margin-bottom: 0.5rem; background: -webkit-linear-gradient(45deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        AI Video Surveillance
    </h1>
    <p style="font-size: 1.25rem; color: #94a3b8; max-width: 800px; margin: 0 auto; line-height: 1.6;">
        Next-Generation Anomaly Detection System. Upload a surveillance video to analyze it clip by clip, score each segment, and review results with Grad-CAM heatmaps and YOLO object detections.
    </p>
</div>
""", unsafe_allow_html=True)

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
        r = requests.get(f"{API_URL}/jobs/{job_id}", timeout=5)
        if r.status_code == 404:
            st.error("Job was lost — the API server may have restarted. Please upload the video again.")
            st.stop()
        status_resp = r.json()
    except Exception:
        time.sleep(2)
        continue

    s = status_resp.get("status")
    if s is None:
        time.sleep(2)
        continue

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
st.markdown("""
<div style="margin-top: 2rem; margin-bottom: 1rem;">
    <h2 style="font-size: 1.8rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.5rem;">Anomaly Score Timeline</h2>
    <p style="color: #94a3b8;">Temporal analysis of anomaly probability throughout the video.</p>
</div>
""", unsafe_allow_html=True)

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
            fillcolor="#ef4444", opacity=0.15, line_width=0,
        )

# Create gradient fill
fig.add_trace(go.Scatter(
    x=times, y=scores, mode="lines",
    line=dict(color="#6366f1", width=3, shape="spline", smoothing=0.3),
    fill="tozeroy", fillcolor="rgba(99, 102, 241, 0.2)",
    name="Anomaly Score",
))
fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444", line_width=2,
              annotation_text=f"Threshold ({threshold})",
              annotation_position="top right",
              annotation_font_color="#ef4444")
fig.update_layout(
    xaxis_title="Time (seconds)", yaxis_title="Anomaly Probability",
    yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    height=350,
    margin=dict(l=0, r=0, t=20, b=0), showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", color="#94a3b8")
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
    st.markdown(f"""
    <div style="margin-top: 2rem; margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.8rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.5rem;">
            Flagged Events <span style="background: #ef4444; color: white; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 1rem; margin-left: 0.5rem; vertical-align: middle;">{len(anomalous_clips)} found</span>
        </h2>
        <p style="color: #94a3b8;">
            Sorted by anomaly severity. Left: original frame. Middle: Grad-CAM heatmap (red = AI focus). Right: YOLO detections.
        </p>
    </div>
    """, unsafe_allow_html=True)

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

    # Free server memory now that all clips have been rendered
    try:
        requests.delete(f"{API_URL}/jobs/{job_id}", timeout=5)
    except Exception:
        pass

st.divider()
st.caption(
    "AI Video Surveillance System — "
    "ResNet50 + LSTM (UCF-Crime, Test AUC 0.8030) + YOLOv8n + Grad-CAM | "
    "FastAPI + Streamlit"
)
