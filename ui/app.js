// Empty string = relative to current origin — works locally (localhost:8000)
// and on HuggingFace Spaces (port 7860) without any change.
const API = "";

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile   = null;
let currentJobId   = null;
let pollTimer      = null;
let progressTimer  = null;
let fakeProgress   = 0;
let timelineChart  = null;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const uploadSection     = document.getElementById("uploadSection");
const processingSection = document.getElementById("processingSection");
const resultsSection    = document.getElementById("resultsSection");
const uploadZone        = document.getElementById("uploadZone");
const fileInput         = document.getElementById("fileInput");
const fileInfo          = document.getElementById("fileInfo");
const fileName          = document.getElementById("fileName");
const fileSize          = document.getElementById("fileSize");
const fileRemove        = document.getElementById("fileRemove");
const analyzeBtn        = document.getElementById("analyzeBtn");
const thresholdSlider   = document.getElementById("threshold");
const thresholdValue    = document.getElementById("thresholdValue");
const errorBox          = document.getElementById("errorBox");
const errorText         = document.getElementById("errorText");
const navStatus         = document.getElementById("navStatus");
const processingLabel   = document.getElementById("processingLabel");
const processingJob     = document.getElementById("processingJob");
const progressFill      = document.getElementById("progressFill");
const statsGrid         = document.getElementById("statsGrid");
const resultsTitle      = document.getElementById("resultsTitle");
const resultsSub        = document.getElementById("resultsSub");
const flaggedSection    = document.getElementById("flaggedSection");
const newAnalysisBtn    = document.getElementById("newAnalysisBtn");

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
        if (r.ok) {
            const d = await r.json();
            setStatus(d.pipeline ? "online" : "loading", d.pipeline ? "System ready" : "Loading model…");
        } else {
            setStatus("offline", "API error");
        }
    } catch {
        setStatus("offline", "Offline");
    }
}

function setStatus(state, text) {
    const dot  = navStatus.querySelector(".status-dot");
    const span = navStatus.querySelector(".status-text");
    dot.className  = `status-dot status-${state}`;
    span.textContent = text;
}

// ── Sections ──────────────────────────────────────────────────────────────────
function showSection(name) {
    uploadSection.style.display     = name === "upload"     ? "" : "none";
    processingSection.style.display = name === "processing" ? "" : "none";
    resultsSection.style.display    = name === "results"    ? "" : "none";
}

// ── File handling ─────────────────────────────────────────────────────────────
const ALLOWED_EXT  = [".mp4", ".avi", ".mov", ".mkv"];
const MAX_BYTES    = 200 * 1024 * 1024;

function formatBytes(b) {
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
    return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

function ext(name) {
    const i = name.lastIndexOf(".");
    return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function setFile(file) {
    if (!ALLOWED_EXT.includes(ext(file.name))) {
        showError(`Unsupported format "${ext(file.name)}". Use MP4, AVI, MOV, or MKV.`);
        return;
    }
    if (file.size > MAX_BYTES) {
        showError(`File too large (${formatBytes(file.size)}). Max 200 MB.`);
        return;
    }
    clearError();
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    fileInfo.style.display  = "flex";
    uploadZone.style.display = "none";
    analyzeBtn.disabled = false;
}

function clearFile() {
    selectedFile = null;
    fileInput.value = "";
    fileInfo.style.display  = "none";
    uploadZone.style.display = "";
    analyzeBtn.disabled = true;
    clearError();
}

function showError(msg) {
    errorText.textContent = msg;
    errorBox.style.display = "flex";
}

function clearError() {
    errorBox.style.display = "none";
    errorText.textContent  = "";
}

// Drag-and-drop
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.querySelector(".upload-link").addEventListener("click", e => {
    e.stopPropagation();
    fileInput.click();
});
fileInput.addEventListener("change", e => {
    if (e.target.files[0]) setFile(e.target.files[0]);
});
fileRemove.addEventListener("click", clearFile);

uploadZone.addEventListener("dragover", e => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", e => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

// Threshold slider
thresholdSlider.addEventListener("input", () => {
    thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(2);
});

// ── Progress bar (fake) ────────────────────────────────────────────────────────
function startFakeProgress() {
    fakeProgress = 0;
    progressFill.style.width = "0%";
    progressFill.style.transition = "none";

    progressTimer = setInterval(() => {
        // Slow down as it approaches 85%
        const remaining = 85 - fakeProgress;
        const step = Math.max(0.2, remaining * 0.03);
        fakeProgress = Math.min(85, fakeProgress + step);
        progressFill.style.transition = "width 0.6s ease";
        progressFill.style.width = `${fakeProgress}%`;
    }, 600);
}

function finishProgress() {
    clearInterval(progressTimer);
    progressFill.style.transition = "width 0.4s ease";
    progressFill.style.width = "100%";
}

// ── Submit job ────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    clearError();

    const threshold = parseFloat(thresholdSlider.value);
    const fd = new FormData();
    fd.append("file", selectedFile);
    fd.append("threshold", threshold);

    showSection("processing");
    processingLabel.textContent = "Submitting job…";
    processingJob.textContent   = "";
    startFakeProgress();

    try {
        const r = await fetch(`${API}/analyze`, { method: "POST", body: fd });
        if (!r.ok) {
            const d = await r.json().catch(() => ({}));
            throw new Error(d.detail || `Server error ${r.status}`);
        }
        const { job_id, filename } = await r.json();
        currentJobId = job_id;
        processingLabel.textContent = "Analyzing…";
        processingJob.textContent   = filename;
        pollJob(job_id, threshold);
    } catch (err) {
        clearInterval(progressTimer);
        showSection("upload");
        showError(err.message || "Failed to submit job. Is the API running?");
    }
});

// ── Poll job ──────────────────────────────────────────────────────────────────
function pollJob(jobId, threshold) {
    let step = 0;
    const labels = ["Extracting frames…", "Running ResNet50…", "Scoring clips (LSTM)…", "Running YOLO + Grad-CAM…", "Finalizing…"];

    pollTimer = setInterval(async () => {
        try {
            const r = await fetch(`${API}/jobs/${jobId}`);
            if (!r.ok) throw new Error(`Poll failed: ${r.status}`);
            const job = await r.json();

            if (job.status === "processing") {
                processingLabel.textContent = labels[Math.min(step++, labels.length - 1)];
            }

            if (job.status === "done") {
                clearInterval(pollTimer);
                finishProgress();
                setTimeout(() => loadResults(jobId, threshold), 400);
            }

            if (job.status === "failed") {
                clearInterval(pollTimer);
                clearInterval(progressTimer);
                showSection("upload");
                showError(job.error || "Analysis failed.");
            }
        } catch (err) {
            // Network hiccup — keep polling
            console.warn("Poll error:", err);
        }
    }, 1500);
}

// ── Load & render results ─────────────────────────────────────────────────────
async function loadResults(jobId, threshold) {
    try {
        const r = await fetch(`${API}/jobs/${jobId}/results`);
        if (!r.ok) throw new Error(`Failed to load results: ${r.status}`);
        const data = await r.json();
        renderResults(data, jobId, threshold);
        showSection("results");
    } catch (err) {
        showSection("upload");
        showError(err.message);
    }
}

function renderResults(data, jobId, threshold) {
    // Header
    const hasAnomalies = data.anomalous_clips > 0;
    resultsTitle.textContent = hasAnomalies
        ? `${data.anomalous_clips} Anomalous Segment${data.anomalous_clips > 1 ? "s" : ""} Detected`
        : "No Anomalies Detected";
    resultsSub.textContent = `${data.clips_analyzed} clips · ${data.duration_sec.toFixed(1)}s · processed in ${data.processing_time.toFixed(1)}s`;

    // Stats
    statsGrid.innerHTML = `
        <div class="stat-card">
            <span class="stat-value">${data.clips_analyzed}</span>
            <span class="stat-label">Clips Analyzed</span>
        </div>
        <div class="stat-card ${hasAnomalies ? "stat-danger" : ""}">
            <span class="stat-value">${data.anomalous_clips}</span>
            <span class="stat-label">Anomalous Clips</span>
        </div>
        <div class="stat-card">
            <span class="stat-value">${(data.max_score * 100).toFixed(1)}%</span>
            <span class="stat-label">Peak Score</span>
        </div>
        <div class="stat-card">
            <span class="stat-value">${data.duration_sec.toFixed(1)}s</span>
            <span class="stat-label">Duration</span>
        </div>
    `;

    // Chart
    renderChart(data.clips, threshold);

    // Flagged clips
    const flagged = data.clips.filter(c => c.is_anomalous)
        .sort((a, b) => b.anomaly_score - a.anomaly_score);
    renderFlagged(flagged, jobId);
}

function renderChart(clips, threshold) {
    if (timelineChart) {
        timelineChart.destroy();
        timelineChart = null;
    }

    const labels = clips.map(c => `${c.timestamp_sec.toFixed(1)}s`);
    const scores  = clips.map(c => +(c.anomaly_score * 100).toFixed(2));
    const flagged = clips.map(c => c.is_anomalous ? +(c.anomaly_score * 100).toFixed(2) : null);

    const ctx = document.getElementById("timelineChart").getContext("2d");
    timelineChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Anomaly Score",
                    data: scores,
                    borderColor: "rgba(99,102,241,0.9)",
                    backgroundColor: "rgba(99,102,241,0.08)",
                    borderWidth: 2,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    pointBackgroundColor: scores.map((s, i) =>
                        clips[i].is_anomalous ? "#ef4444" : "rgba(99,102,241,0.9)"
                    ),
                    tension: 0.35,
                    fill: true,
                },
                {
                    label: "Flagged",
                    data: flagged,
                    borderColor: "transparent",
                    backgroundColor: "rgba(239,68,68,0.2)",
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    pointBackgroundColor: "#ef4444",
                    pointBorderColor: "rgba(239,68,68,0.4)",
                    pointBorderWidth: 2,
                    showLine: false,
                    fill: false,
                },
                {
                    label: "Threshold",
                    data: clips.map(() => +(threshold * 100).toFixed(1)),
                    borderColor: "rgba(239,68,68,0.5)",
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    fill: false,
                    tension: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: "index" },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#111128",
                    borderColor: "rgba(255,255,255,0.08)",
                    borderWidth: 1,
                    titleColor: "#f1f5f9",
                    bodyColor: "#94a3b8",
                    padding: 10,
                    callbacks: {
                        label: ctx => {
                            if (ctx.dataset.label === "Threshold") return `Threshold: ${ctx.raw}%`;
                            if (ctx.dataset.label === "Flagged" && ctx.raw !== null) return `Flagged: ${ctx.raw}%`;
                            if (ctx.dataset.label === "Anomaly Score") return `Score: ${ctx.raw}%`;
                            return null;
                        },
                    },
                },
            },
            scales: {
                x: {
                    grid:  { color: "rgba(255,255,255,0.04)" },
                    ticks: { color: "#475569", maxTicksLimit: 12, font: { size: 11 } },
                },
                y: {
                    min: 0, max: 100,
                    grid:  { color: "rgba(255,255,255,0.04)" },
                    ticks: {
                        color: "#475569", font: { size: 11 },
                        callback: v => `${v}%`,
                    },
                },
            },
        },
    });
}

function renderFlagged(flagged, jobId) {
    if (flagged.length === 0) {
        flaggedSection.innerHTML = `
            <div class="no-anomalies">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                </svg>
                No anomalous clips detected above the threshold.
            </div>`;
        return;
    }

    const header = document.createElement("div");
    header.className = "section-label";
    header.style.cssText = "margin-bottom:12px;display:block";
    header.textContent = `Flagged Events (${flagged.length})`;
    flaggedSection.innerHTML = "";
    flaggedSection.appendChild(header);

    flagged.forEach((clip, i) => {
        const card = buildClipCard(clip, jobId, i === 0);
        flaggedSection.appendChild(card);
    });
}

function buildClipCard(clip, jobId, expanded) {
    const pct  = (clip.anomaly_score * 100).toFixed(1);
    const card = document.createElement("div");
    card.className = "clip-card";

    const scoreClass = clip.anomaly_score >= 0.8 ? "score-high"
                     : clip.anomaly_score >= 0.6 ? "score-med"
                     : "score-low";

    card.innerHTML = `
        <div class="clip-header" role="button" tabindex="0" aria-expanded="${expanded}">
            <div class="clip-header-left">
                <span class="clip-time">${clip.timestamp_sec.toFixed(1)}s</span>
                <span class="clip-badge ${scoreClass}">${pct}%</span>
            </div>
            <div class="clip-header-right">
                <span class="clip-expl">${clip.explanation || "Anomaly detected"}</span>
                <svg class="clip-chevron ${expanded ? "open" : ""}" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <polyline points="6 9 12 15 18 9"/>
                </svg>
            </div>
        </div>
        <div class="clip-body" style="display:${expanded ? "grid" : "none"}">
            <div class="clip-img-wrap">
                <p class="clip-img-label">Original Frame</p>
                <img class="clip-img" src="${API}/jobs/${jobId}/clips/${clip.clip_idx}/frame"
                     alt="Frame" loading="lazy"
                     onerror="this.parentElement.innerHTML='<div class=clip-img-err>Frame unavailable</div>'">
            </div>
            <div class="clip-img-wrap">
                <p class="clip-img-label">Grad-CAM Heatmap</p>
                <img class="clip-img" src="${API}/jobs/${jobId}/clips/${clip.clip_idx}/heatmap"
                     alt="Heatmap" loading="lazy"
                     onerror="this.parentElement.innerHTML='<div class=clip-img-err>Heatmap unavailable</div>'">
            </div>
            <div class="clip-detections">
                <p class="clip-img-label">Detections</p>
                ${renderDetections(clip.detections)}
            </div>
        </div>
    `;

    const header  = card.querySelector(".clip-header");
    const body    = card.querySelector(".clip-body");
    const chevron = card.querySelector(".clip-chevron");

    const toggle = () => {
        const isOpen = body.style.display !== "none";
        body.style.display    = isOpen ? "none" : "grid";
        chevron.classList.toggle("open", !isOpen);
        header.setAttribute("aria-expanded", String(!isOpen));
    };
    header.addEventListener("click", toggle);
    header.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") toggle(); });

    return card;
}

function renderDetections(detections) {
    if (!detections || detections.length === 0) {
        return `<p class="det-empty">No objects detected</p>`;
    }
    const rows = detections.map(d => `
        <div class="det-row">
            <span class="det-name">${d.class_name}</span>
            <span class="det-conf">${(d.confidence * 100).toFixed(0)}%</span>
        </div>
    `).join("");
    return `<div class="det-list">${rows}</div>`;
}

// ── New Analysis ──────────────────────────────────────────────────────────────
newAnalysisBtn.addEventListener("click", async () => {
    if (currentJobId) {
        fetch(`${API}/jobs/${currentJobId}`, { method: "DELETE" }).catch(() => {});
        currentJobId = null;
    }
    clearInterval(pollTimer);
    clearInterval(progressTimer);
    clearFile();
    flaggedSection.innerHTML = "";
    statsGrid.innerHTML = "";
    if (timelineChart) { timelineChart.destroy(); timelineChart = null; }
    showSection("upload");
});

// ── Cleanup on unload ─────────────────────────────────────────────────────────
window.addEventListener("beforeunload", () => {
    clearInterval(pollTimer);
    clearInterval(progressTimer);
    if (currentJobId) {
        navigator.sendBeacon(`${API}/jobs/${currentJobId}`, null);
    }
});

// ── Init ──────────────────────────────────────────────────────────────────────
showSection("upload");
checkHealth();
setInterval(checkHealth, 15000);
