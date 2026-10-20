from __future__ import annotations

import hashlib
import json
import time
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="TSAD Dashboard | 时间序列异常检测",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Corporate Clean design system
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --cc-blue: #2563eb;
        --cc-blue-hover: #1d4ed8;
        --cc-blue-light: #eff6ff;
        --cc-page: #f8fafc;
        --cc-white: #ffffff;
        --cc-text: #111827;
        --cc-body: #374151;
        --cc-muted: #6b7280;
        --cc-border: #e5e7eb;
    }

    .stApp {
        background: #f8fafc;
        color: #111827;
        font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    }

    .block-container {
        max-width: 1440px;
        padding: 2rem 2rem 3rem;
    }

    [data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.96);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f1f5f9;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #111827;
        font-weight: 650;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #374151;
    }

    [data-testid="stSidebar"] hr {
        border-color: #e5e7eb;
    }

    /* Sidebar sliders: override Streamlit/BaseWeb's default red primary color */
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] {
        padding-top: 0.55rem !important;
        padding-bottom: 0.65rem !important;
        --slider-primary: #2563eb;
    }

    /* Thumb / draggable handle */
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"],
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] > div {
        background-color: #2563eb !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 0 1px #2563eb, 0 1px 3px rgba(15, 23, 42, 0.18) !important;
    }

    /* Filled rail: BaseWeb versions use different nested wrappers, so target
       the slider's inline-positioned rail and common track wrappers. */
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[style*="left:"] {
        background-color: #2563eb !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div > div,
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
        border-color: #2563eb !important;
    }

    /* Use a blue focus outline for keyboard users */
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"]:focus,
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"]:focus-visible {
        outline: 2px solid #2563eb !important;
        outline-offset: 2px !important;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBar"] {
        color: #6b7280 !important;
    }

    /* Headings and text */
    h1, h2, h3 {
        color: #111827;
        font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        line-height: 1.35 !important;
    }

    h2 {
        font-size: 1.4rem !important;
        font-weight: 650 !important;
    }

    h3 {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }

    p, li {
        color: #374151;
        line-height: 1.7;
    }

    hr {
        border-color: #e5e7eb;
        margin: 1.5rem 0;
    }

    /* Hero header */
    .cc-hero {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 28px;
        margin-bottom: 22px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        transition: box-shadow 180ms ease, transform 180ms ease;
    }

    .cc-hero:hover {
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.07);
        transform: translateY(-1px);
    }

    .cc-kicker {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        padding: 5px 10px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .cc-hero-title {
        color: #111827;
        font-size: 30px;
        font-weight: 700;
        line-height: 1.4;
        letter-spacing: -0.6px;
    }

    .cc-hero-description {
        color: #4b5563;
        font-size: 14px;
        line-height: 1.8;
        margin-top: 9px;
        max-width: 760px;
    }

    .cc-section-label {
        color: #111827;
        font-size: 1.15rem;
        font-weight: 650;
        margin: 0.2rem 0 0.8rem 0;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        transition: box-shadow 180ms ease, transform 180ms ease;
    }

    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.07);
        transform: translateY(-1px);
    }

    [data-testid="stMetricLabel"] {
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 500;
    }

    [data-testid="stMetricValue"] {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        min-height: 42px;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        transition: background-color 150ms ease-out,
                    border-color 150ms ease-out,
                    box-shadow 150ms ease-out,
                    transform 150ms ease-out;
    }

    /* Primary action button: blue background with forced white text.
       Streamlit versions differ in button attributes, so cover both selectors. */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primary"] {
        color: #ffffff !important;
        background: #2563eb !important;
        border: 1px solid #2563eb !important;
    }

    .stButton > button[kind="primary"] *,
    .stButton > button[data-testid="stBaseButton-primary"] *,
    [data-testid="stBaseButton-primary"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        color: #ffffff !important;
        background: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.16);
        transform: translateY(-1px);
    }

    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: scale(0.98) translateY(0);
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }

    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible {
        outline: 2px solid #2563eb;
        outline-offset: 2px;
    }

    .stDownloadButton > button,
    .stButton > button[kind="secondary"] {
        background: #ffffff;
        color: #374151;
        border: 1px solid #d1d5db;
    }

    .stDownloadButton > button:hover,
    .stButton > button[kind="secondary"]:hover {
        background: #f9fafb;
        border-color: #9ca3af;
        box-shadow: 0 3px 7px rgba(15, 23, 42, 0.07);
        transform: translateY(-1px);
    }

    /* Inputs */
    [data-baseweb="select"] > div {
        border-color: #d1d5db;
        border-radius: 8px;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input {
        border-radius: 8px;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.18);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #d1d5db;
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #2563eb;
        background: #f9fbff;
    }

    [data-testid="stFileUploader"] button {
        border-radius: 8px;
    }

    /* Charts and tables */
    [data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
        border-width: 1px;
    }

    [data-testid="stCheckbox"] label {
        color: #374151;
    }

    a {
        color: #2563eb;
    }

    a:hover {
        color: #1d4ed8;
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1.2rem 1rem 2rem;
        }

        .cc-hero {
            padding: 20px;
        }

        .cc-hero-title {
            font-size: 24px;
        }

        h1 {
            font-size: 1.65rem !important;
        }

        [data-testid="stMetric"] {
            padding: 14px;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.45rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Utility functions
# ============================================================
from scipy.spatial.distance import cdist

ALGORITHM_IFOREST = "Isolation Forest（滑动窗口）"
ALGORITHM_SR = "Spectral Residual（频谱残差）"
ALGORITHM_MP = "Matrix Profile（矩阵轮廓）"
ALGORITHM_LOF = "Local Outlier Factor（LOF）"
ALGORITHM_KNN = "K-Nearest Neighbors（KNN）"
CORE_ALGORITHMS = [ALGORITHM_IFOREST, ALGORITHM_SR, ALGORITHM_MP]
ALL_ALGORITHMS = CORE_ALGORITHMS + [ALGORITHM_LOF, ALGORITHM_KNN]


def make_sample_csv() -> bytes:
    """Create a reproducible multivariate monitoring CSV with injected anomalies."""
    rng = np.random.default_rng(42)
    n = 240
    timestamps = pd.date_range("2026-01-01", periods=n, freq="min")
    cpu = 38 + 7 * np.sin(np.linspace(0, 8 * np.pi, n)) + rng.normal(0, 2.5, n)
    memory = 58 + 4 * np.sin(np.linspace(0, 4 * np.pi, n) + 0.7) + rng.normal(0, 1.8, n)
    requests = 120 + 18 * np.sin(np.linspace(0, 10 * np.pi, n) + 0.3) + rng.normal(0, 8, n)
    disk = 45 + 9 * np.sin(np.linspace(0, 6 * np.pi, n) + 1.1) + rng.normal(0, 3, n)
    for idx in (58, 59, 120, 121, 190): cpu[idx] += rng.uniform(25, 38)
    for idx in (82, 150, 151): memory[idx] += rng.uniform(12, 18)
    for idx in (45, 46, 175): requests[idx] += rng.uniform(70, 100)
    for idx in (95, 96, 200): disk[idx] += rng.uniform(35, 45)
    sample = pd.DataFrame({
        "timestamp": timestamps,
        "cpu_usage": np.clip(cpu, 0, 100),
        "memory_usage": np.clip(memory, 0, 100),
        "disk_io": np.clip(disk, 0, 100),
        "request_count": np.clip(requests, 0, None).round().astype(int),
    })
    return sample.to_csv(index=False).encode("utf-8-sig")


def make_windows(values: np.ndarray, window_size: int) -> np.ndarray:
    """Causal windows: current observation plus preceding observations."""
    if window_size <= 1:
        return values
    padded = np.pad(values, ((window_size - 1, 0), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_size, axis=0)
    return windows.transpose(0, 2, 1).reshape(values.shape[0], -1)


def make_result_key(file_bytes: bytes, time_column: str | None, metric_columns: list[str],
                    algorithm: str, contamination: float, window_size: int,
                    duplicate_strategy: str, invalid_time_strategy: str,
                    normalization: str = "Z-Score", resample_rule: str = "不重采样",
                    remove_negative: bool = False, window_step: int = 1) -> str:
    digest = hashlib.sha256()
    digest.update(file_bytes)
    digest.update(repr((time_column, tuple(metric_columns), algorithm, float(contamination),
                        int(window_size), duplicate_strategy, invalid_time_strategy, normalization, resample_rule, remove_negative, int(window_step))).encode("utf-8"))
    return digest.hexdigest()


def _spectral_residual_scores(values: np.ndarray) -> np.ndarray:
    """A lightweight Spectral Residual implementation for each selected metric."""
    eps = 1e-8
    feature_scores = []
    for col in range(values.shape[1]):
        x = values[:, col].astype(float)
        # Fill any residual non-finite values before FFT.
        x = np.nan_to_num(x, nan=np.nanmedian(x) if np.isfinite(x).any() else 0.0)
        spectrum = np.fft.fft(x)
        amplitude = np.abs(spectrum)
        log_amplitude = np.log(amplitude + eps)
        # Circular moving average of log amplitude.
        kernel = np.ones(3, dtype=float) / 3.0
        avg_log_amplitude = np.convolve(np.pad(log_amplitude, (1, 1), mode="wrap"), kernel, mode="valid")
        spectral_residual = np.exp(log_amplitude - avg_log_amplitude)
        reconstructed = spectral_residual * np.exp(1j * np.angle(spectrum))
        saliency = np.abs(np.fft.ifft(reconstructed))
        # Add a local deviation term so abrupt level shifts are easier to surface.
        local = pd.Series(x).rolling(window=5, min_periods=1, center=True).median().to_numpy()
        deviation = np.abs(x - local)
        saliency = saliency + deviation / (np.nanstd(deviation) + eps)
        saliency = (saliency - np.nanmedian(saliency)) / (np.nanstd(saliency) + eps)
        feature_scores.append(np.nan_to_num(saliency, nan=0.0, posinf=0.0, neginf=0.0))
    return np.mean(np.vstack(feature_scores), axis=0)


def _matrix_profile_scores(values: np.ndarray, window_size: int) -> np.ndarray:
    """Compute a multivariate z-normalized matrix profile with an exclusion zone."""
    n = len(values)
    w = max(2, int(window_size))
    if n < w + 2:
        raise ValueError(f"Matrix Profile 至少需要 {w + 2} 行数据；当前只有 {n} 行。")
    if n - w + 1 > 2500:
        raise ValueError(
            "Matrix Profile 当前为便于本地运行采用精确两两距离计算，最多支持 2500 个滑动子序列。"
            "请先按时间聚合或下采样数据后重试。"
        )
    raw = np.lib.stride_tricks.sliding_window_view(values, window_shape=w, axis=0)
    # Shape: (subsequences, window, features). Normalize each feature within each subsequence.
    raw = raw.transpose(0, 2, 1)
    means = raw.mean(axis=1, keepdims=True)
    stds = raw.std(axis=1, keepdims=True)
    normalized = (raw - means) / np.maximum(stds, 1e-8)
    subsequences = normalized.transpose(0, 2, 1).reshape(len(raw), -1)
    count = len(subsequences)
    profile = np.full(count, np.inf, dtype=float)
    exclusion = max(1, w // 2)
    # Row-by-row distance calculation keeps memory use modest.
    for i in range(count):
        distances = cdist(subsequences[i:i + 1], subsequences, metric="euclidean")[0]
        lo, hi = max(0, i - exclusion), min(count, i + exclusion + 1)
        distances[lo:hi] = np.inf
        profile[i] = np.min(distances)
    # Align subsequence scores to their ending observation; pad the initial warm-up region.
    aligned = np.empty(n, dtype=float)
    aligned[:w - 1] = profile[0]
    aligned[w - 1:] = profile
    return np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)


def _labels_from_scores(scores: np.ndarray, contamination: float) -> tuple[np.ndarray, float]:
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    threshold = float(np.quantile(scores, 1.0 - contamination))
    labels = scores >= threshold
    # Ensure a non-empty, deterministic anomaly set for tied scores where possible.
    target = max(1, int(np.ceil(len(scores) * contamination))) if len(scores) else 0
    if labels.sum() > target:
        selected = np.argsort(scores)[-target:]
        labels = np.zeros(len(scores), dtype=bool)
        labels[selected] = True
    return labels, threshold


@st.cache_data(show_spinner=False, max_entries=16)
def run_detection(dataframe: pd.DataFrame, metric_columns: list[str], algorithm: str,
                  contamination: float, window_size: int, normalization: str = "Z-Score", window_step: int = 1) -> tuple[pd.DataFrame, float, float]:
    """Preprocess selected metrics and run a detector; return results, elapsed time, threshold."""
    if len(dataframe) < 5:
        raise ValueError("可用于检测的数据至少需要 5 行，请检查时间戳去重或无效时间戳删除设置。")
    started = time.perf_counter()
    numeric = dataframe[metric_columns].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median(axis=0).fillna(0.0)
    imputed = numeric.fillna(medians).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    scaler = MinMaxScaler() if normalization == "Min-Max" else StandardScaler()
    scaled = scaler.fit_transform(imputed)

    if algorithm == ALGORITHM_SR:
        scores = _spectral_residual_scores(scaled)
        labels, threshold = _labels_from_scores(scores, contamination)
    elif algorithm == ALGORITHM_MP:
        scores = _matrix_profile_scores(scaled, window_size)
        labels, threshold = _labels_from_scores(scores, contamination)
    else:
        features = make_windows(scaled, window_size)
        if algorithm == ALGORITHM_IFOREST:
            model = IForest(contamination=contamination, random_state=42, n_estimators=200)
        elif algorithm == ALGORITHM_LOF:
            model = LOF(contamination=contamination, n_neighbors=max(2, min(20, len(features) - 1)))
        else:
            model = KNN(contamination=contamination, n_neighbors=max(1, min(5, len(features) - 1)), method="largest")
        step = max(1, int(window_step))
        train_features = features[::step]
        model.fit(train_features)
        try:
            scores = np.asarray(model.decision_function(features), dtype=float)
        except Exception:
            # Fallback for detectors without out-of-sample scoring: interpolate training scores.
            sampled_scores = np.asarray(model.decision_scores_, dtype=float)
            scores = np.interp(np.arange(len(features)), np.arange(0, len(features), step)[:len(sampled_scores)], sampled_scores)

    # Normalize all algorithm scores to [0, 1] so the dashboard threshold is comparable.
    raw_scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    score_min, score_max = float(np.min(raw_scores)), float(np.max(raw_scores))
    scores = (raw_scores - score_min) / (score_max - score_min) if score_max > score_min else np.zeros_like(raw_scores)
    labels, threshold = _labels_from_scores(scores, contamination)

    results = dataframe.copy()
    results["raw_anomaly_score"] = raw_scores
    results["anomaly_score"] = np.asarray(scores, dtype=float)
    results["is_anomaly"] = np.asarray(labels, dtype=bool)
    results["anomaly_label"] = np.where(results["is_anomaly"], "异常", "正常")
    elapsed = time.perf_counter() - started
    return results, elapsed, threshold


def build_time_series_chart(results: pd.DataFrame, time_column: str | None,
                            metric_column: str) -> go.Figure:
    if time_column:
        x_values = results[time_column]
        parsed = pd.to_datetime(x_values, errors="coerce")
        if parsed.notna().all(): x_values = parsed
    else:
        x_values = pd.Series(np.arange(len(results)), index=results.index, name="样本序号")
    metric_values = pd.to_numeric(results[metric_column], errors="coerce")
    mask = results["is_anomaly"].to_numpy(dtype=bool)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=metric_values, mode="lines", name=metric_column,
                             line=dict(color="#2563eb", width=2), connectgaps=False))
    if mask.any():
        fig.add_trace(go.Scatter(x=np.asarray(x_values)[mask], y=metric_values.to_numpy()[mask],
                                 mode="markers", name="检测到的异常",
                                 marker=dict(color="#dc2626", size=9, line=dict(color="#ffffff", width=1.5))))
    fig.update_layout(template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="Segoe UI, Microsoft YaHei, sans-serif", color="#374151", size=12),
        title=dict(text=f"{metric_column} 时序趋势", font=dict(size=16, color="#111827")),
        xaxis_title="时间" if time_column else "样本序号", yaxis_title=metric_column,
        margin=dict(l=18, r=18, t=55, b=18), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified")
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#e5e7eb", zeroline=False)
    return fig


def build_multi_metric_chart(results: pd.DataFrame, time_column: str | None, metric_columns: list[str], mode: str) -> go.Figure:
    if time_column:
        x_values = pd.to_datetime(results[time_column], errors="coerce")
        if x_values.isna().all():
            x_values = results[time_column]
    else:
        x_values = np.arange(len(results))
    mask = results["is_anomaly"].to_numpy(dtype=bool)
    if mode == "合并曲线":
        fig = go.Figure()
        for metric in metric_columns:
            fig.add_trace(go.Scatter(x=x_values, y=pd.to_numeric(results[metric], errors="coerce"),
                mode="lines", name=metric, connectgaps=False))
        if mask.any():
            for metric in metric_columns:
                metric_y = pd.to_numeric(results[metric], errors="coerce").to_numpy()
                fig.add_trace(go.Scatter(x=np.asarray(x_values)[mask], y=metric_y[mask],
                    customdata=results["anomaly_score"].to_numpy()[mask], mode="markers", name=f"{metric} 异常",
                    marker=dict(color="#dc2626", size=8, line=dict(color="#ffffff", width=1)),
                    hovertemplate="时间：%{x}<br>指标值：%{y}<br>异常分数：%{customdata:.3f}<extra></extra>"))
        fig.update_layout(title="多指标时序总览", template="plotly_white", hovermode="x unified")
    else:
        fig = make_subplots(rows=len(metric_columns), cols=1, shared_xaxes=True, vertical_spacing=min(0.06, 0.25 / max(1, len(metric_columns))), subplot_titles=metric_columns)
        for row, metric in enumerate(metric_columns, start=1):
            y = pd.to_numeric(results[metric], errors="coerce")
            fig.add_trace(go.Scatter(x=x_values, y=y, mode="lines", name=metric, line=dict(color="#2563eb", width=1.7), connectgaps=False), row=row, col=1)
            if mask.any():
                fig.add_trace(go.Scatter(x=np.asarray(x_values)[mask], y=y.to_numpy()[mask], customdata=results["anomaly_score"].to_numpy()[mask], mode="markers", name=f"{metric} 异常", marker=dict(color="#dc2626", size=7), showlegend=(row == 1), hovertemplate="时间：%{x}<br>指标值：%{y}<br>异常分数：%{customdata:.3f}<extra></extra>"), row=row, col=1)
        fig.update_layout(title="多指标时序子图", template="plotly_white", height=max(360, 230 * len(metric_columns)), hovermode="x unified")
    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(family="Segoe UI, Microsoft YaHei, sans-serif", color="#374151", size=12), margin=dict(l=20, r=20, t=60, b=25))
    return fig


def build_score_chart(results: pd.DataFrame, time_column: str | None, threshold: float) -> go.Figure:
    if time_column:
        x_values = results[time_column]
        parsed = pd.to_datetime(x_values, errors="coerce")
        if parsed.notna().all(): x_values = parsed
    else:
        x_values = pd.Series(np.arange(len(results)), index=results.index, name="样本序号")
    mask = results["is_anomaly"].to_numpy(dtype=bool)
    scores = results["anomaly_score"].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=scores, mode="lines", name="异常分数", line=dict(color="#64748b", width=1.7)))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#d97706", annotation_text="检测阈值", annotation_position="top left")
    if mask.any():
        fig.add_trace(go.Scatter(x=np.asarray(x_values)[mask], y=scores[mask], mode="markers", name="异常",
                                 marker=dict(color="#dc2626", size=8, line=dict(color="#ffffff", width=1))))
    fig.update_layout(template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="Segoe UI, Microsoft YaHei, sans-serif", color="#374151", size=12),
        title=dict(text="异常评分与检测阈值", font=dict(size=16, color="#111827")),
        xaxis_title="时间" if time_column else "样本序号", yaxis_title="异常分数",
        margin=dict(l=18, r=18, t=55, b=18), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified")
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#e5e7eb", zeroline=False)
    return fig


def build_metric_distribution(results: pd.DataFrame, metric_columns: list[str]) -> go.Figure:
    counts = []
    for column in metric_columns:
        values = pd.to_numeric(results[column], errors="coerce")
        valid = values.notna()
        counts.append({"指标": column, "异常点数": int((results["is_anomaly"] & valid).sum()),
                       "有效样本数": int(valid.sum())})
    dist = pd.DataFrame(counts)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dist["指标"], y=dist["异常点数"], name="异常点数", marker_color="#2563eb"))
    fig.update_layout(template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        title="各指标对应的异常时间点数量（同一时间点异常会计入各指标）",
        yaxis_title="异常点数", margin=dict(l=18, r=18, t=55, b=18), font=dict(color="#374151", size=12))
    fig.update_xaxes(showgrid=False, linecolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#e5e7eb", rangemode="tozero")
    return fig


def anomaly_intervals(results: pd.DataFrame, time_column: str | None) -> pd.DataFrame:
    mask = results["is_anomaly"].to_numpy(dtype=bool)
    metric_columns = [c for c in results.columns if c not in {time_column, "raw_anomaly_score", "anomaly_score", "is_anomaly", "anomaly_label", "label", "true_label"} and pd.api.types.is_numeric_dtype(results[c])]
    if not mask.any(): return pd.DataFrame(columns=["开始时间/样本", "结束时间/样本", "持续样本数", "涉及指标", "持续时长"])
    groups = []
    start = None
    for idx, is_anomaly in enumerate(mask):
        if is_anomaly and start is None: start = idx
        is_last = idx == len(mask) - 1
        if start is not None and (not is_anomaly or is_last):
            end = idx if is_anomaly and is_last else idx - 1
            if time_column:
                start_label = str(results.iloc[start][time_column])
                end_label = str(results.iloc[end][time_column])
            else:
                start_label, end_label = f"样本 {start + 1}", f"样本 {end + 1}"
            segment = results.iloc[start:end + 1]
            involved = []
            for metric in metric_columns:
                if segment[metric].notna().any() and segment[metric].nunique(dropna=True) > 1:
                    involved.append(metric)
            duration = "—"
            if time_column:
                try:
                    delta = pd.to_datetime(results.iloc[end][time_column]) - pd.to_datetime(results.iloc[start][time_column])
                    duration = str(delta)
                except Exception:
                    pass
            groups.append({"开始时间/样本": start_label, "结束时间/样本": end_label, "持续样本数": end - start + 1, "涉及指标": "、".join(involved) if involved else "所选指标", "持续时长": duration})
            start = None
    return pd.DataFrame(groups)


# ============================================================
# Sidebar: detection configuration
# ============================================================
with st.sidebar:
    st.markdown("## TSAD Dashboard")
    st.caption("时间序列异常检测")
    st.divider()
    st.markdown("### 检测配置")
    algorithm = st.selectbox("检测算法", ALL_ALGORITHMS,
        help="Isolation Forest、LOF、KNN 为 PyOD 通用检测器；Spectral Residual 与 Matrix Profile 为本项目实现的时序检测方法。")
    contamination = st.slider("预期异常比例", min_value=0.01, max_value=0.20, value=0.05, step=0.01,
        format="%.2f", help="用于设置预期异常点比例，不等同于业务真实故障率。")
    window_size = st.slider("时间窗口大小", min_value=2, max_value=60, value=5, step=1,
        help="用于构造 Isolation Forest/LOF/KNN 的因果窗口特征；Matrix Profile 使用该值作为子序列长度。")
    window_step = st.slider("窗口步长", min_value=1, max_value=10, value=1, step=1,
        help="PyOD 模型按该步长抽取窗口训练样本，最终仍会对每个时间点评分。1 表示不跳步。")
    normalization = st.selectbox("指标归一化", ["Z-Score", "Min-Max"], index=0,
        help="对每个指标分别归一化，降低 CPU、内存、磁盘 I/O 等量纲差异对模型的影响。")
    st.divider()
    st.markdown("### 使用说明")
    st.markdown("1. 上传 CSV 数据文件。\n2. 选择时间列与数值指标。\n3. 配置算法并启动检测。\n4. 查看图表、异常区间并导出结果。")
    with st.expander("部署选项说明（中文）", expanded=False):
        st.markdown("**Streamlit Community Cloud｜社区云**\n\n适合课程项目与个人演示。连接 GitHub 仓库，选择分支和 `app.py` 后部署，通常可生成在线访问链接。")
        st.markdown("**Snowflake｜企业数据云**\n\n适合已使用 Snowflake 的企业数据分析环境；个人课程演示一般不需要选择。")
        st.markdown("**Other platforms｜其他部署平台**\n\n适合自有服务器或其他云平台，需要自行安装依赖、配置启动命令、网络与安全策略。")
    st.caption("支持常见多变量运维指标。建议先用示例 CSV 体验。")


# ============================================================
# Main page header
# ============================================================
st.markdown("""
<div class="cc-hero">
  <div class="cc-kicker">TSAD · ANOMALY ANALYTICS</div>
  <div class="cc-hero-title">时间序列异常检测平台</div>
  <div class="cc-hero-description">面向运维指标的异常分析工具。支持 CSV 数据导入、时序异常检测、交互式分析、算法对比与结果导出。</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="cc-section-label">01 数据导入与质量检查</div>', unsafe_allow_html=True)
upload_col, sample_col = st.columns([3, 1], gap="large")
with upload_col:
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"], help="每行建议代表一个时间点，每列代表时间戳或数值指标。", key="tsad_csv_uploader")
with sample_col:
    st.markdown("#### 示例数据")
    st.download_button("下载示例 CSV", data=make_sample_csv(), file_name="tsad_sample_data.csv", mime="text/csv", use_container_width=True)
    st.caption("包含时间戳、CPU、内存、磁盘 I/O 和请求量。")

with st.expander("可选：粘贴 JSON 指标数据", expanded=False):
    json_text = st.text_area("JSON 数组或对象", placeholder='[{"timestamp":"2026-01-01 00:00:00","cpu_usage":35}]', height=100)
    json_load = st.button("载入 JSON 数据", key="load_json_data")
    if json_load and json_text.strip():
        try:
            parsed_json = json.loads(json_text)
            if isinstance(parsed_json, dict):
                parsed_json = [parsed_json]
            if not isinstance(parsed_json, list) or not parsed_json or not all(isinstance(row, dict) for row in parsed_json):
                raise ValueError("请输入 JSON 对象或对象数组。")
            json_df = pd.json_normalize(parsed_json)
            st.session_state["tsad_json_csv"] = json_df.to_csv(index=False).encode("utf-8-sig")
            st.success("JSON 已转换为表格数据。请在下方使用上传的 CSV 或重新载入 JSON。")
        except Exception as exc:
            st.error(f"JSON 解析失败：{exc}")

if uploaded_file is None and not st.session_state.get("tsad_json_csv"):
    st.info("请上传 CSV 文件，或下载示例数据体验平台功能。")
    st.markdown("""<div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:24px;box-shadow:0 1px 2px rgba(15,23,42,.05)"><div style="font-size:16px;font-weight:650;color:#111827;margin-bottom:8px">分析流程</div><div style="color:#6b7280;line-height:1.9">数据上传 → 数据质量检查 → 指标选择 → 异常检测 → 可视化分析 → 结果导出</div></div>""", unsafe_allow_html=True)
    st.stop()

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    try:
        try:
            dataframe = pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig")
        except UnicodeDecodeError:
            dataframe = pd.read_csv(BytesIO(file_bytes), encoding="gb18030")
    except Exception as exc:
        st.error(f"CSV 文件读取失败，请检查编码、分隔符和文件格式。详情：{exc}")
        st.stop()
else:
    file_bytes = st.session_state["tsad_json_csv"]
    try:
        dataframe = pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig")
    except Exception as exc:
        st.error(f"JSON 转换结果读取失败：{exc}")
        st.stop()

if dataframe.empty or len(dataframe.columns) == 0:
    st.error("CSV 文件为空或没有有效列，请检查文件内容。")
    st.stop()
# Normalize headers and identify duplicate column names.
dataframe.columns = [str(c).strip() for c in dataframe.columns]
if any(not c for c in dataframe.columns):
    st.error("检测到空列名，请为 CSV 中的每一列设置名称后重试。")
    st.stop()
if dataframe.columns.duplicated().any():
    st.error("检测到重复列名，请先修改 CSV 列名，确保每列名称唯一。")
    st.stop()

numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
# Detect numeric-like strings as candidate metrics too.
for col in dataframe.columns:
    if col not in numeric_columns:
        converted = pd.to_numeric(dataframe[col], errors="coerce")
        if converted.notna().sum() >= max(3, int(len(dataframe) * 0.5)):
            numeric_columns.append(col)

label_options = ["（无标注列）"] + dataframe.columns.tolist()
label_column_option = st.selectbox("可选：真实异常标签列（用于 Precision / Recall / F1 评估）", label_options,
    help="如果数据中包含人工标注列，请选择它；0/正常/normal 表示正常，1/异常/anomaly 表示异常。")
label_column = None if label_column_option == label_options[0] else label_column_option
if label_column in numeric_columns:
    numeric_columns = [c for c in numeric_columns if c != label_column]

st.markdown('<div class="cc-section-label">02 数据概览与预处理</div>', unsafe_allow_html=True)
missings = int(dataframe.isna().sum().sum())
duplicate_rows = int(dataframe.duplicated().sum())
overview_1, overview_2, overview_3, overview_4 = st.columns(4, gap="medium")
overview_1.metric("原始数据行数", f"{len(dataframe):,}")
overview_2.metric("数据列数", f"{len(dataframe.columns):,}")
overview_3.metric("缺失单元格", f"{missings:,}")
overview_4.metric("完全重复行", f"{duplicate_rows:,}")

with st.expander("数据质量详情", expanded=False):
    quality = pd.DataFrame({"列名": dataframe.columns, "数据类型": [str(dataframe[c].dtype) for c in dataframe.columns],
        "缺失值": [int(dataframe[c].isna().sum()) for c in dataframe.columns],
        "唯一值数量": [int(dataframe[c].nunique(dropna=True)) for c in dataframe.columns]})
    st.dataframe(quality, use_container_width=True, hide_index=True)
    if duplicate_rows:
        st.warning(f"发现 {duplicate_rows} 行完全重复记录。可在下方选择去重策略。")

# Suggest likely timestamp columns by name, then parseability.
time_options = ["（不使用时间列，按样本序号显示）"] + dataframe.columns.tolist()
suggested_time_index = 0
for idx, column in enumerate(dataframe.columns, start=1):
    lower = str(column).lower()
    if any(token in lower for token in ("timestamp", "datetime", "date", "time", "时间", "日期")):
        suggested_time_index = idx
        break
    parsed_rate = pd.to_datetime(dataframe[column], errors="coerce").notna().mean() if dataframe[column].dtype == object else 0
    if parsed_rate > 0.8 and suggested_time_index == 0:
        suggested_time_index = idx

config_col1, config_col2, config_col3 = st.columns([1, 2, 1.4], gap="large")
with config_col1:
    selected_time_option = st.selectbox("时间列（可选）", time_options, index=suggested_time_index,
        help="自动建议时间戳列；如果不选，则横轴显示样本序号。")
with config_col2:
    if not numeric_columns:
        st.error("没有找到可用的数值指标列。请检查 CSV 中的指标是否为数值，或是否混有单位字符。")
        st.stop()
    # Prefer common monitoring metrics when setting the initial selection.
    preferred_tokens = ("cpu", "memory", "mem", "disk", "io", "network", "request", "latency", "load")
    preferred_metrics = [c for c in numeric_columns if any(token in str(c).lower() for token in preferred_tokens)]
    default_metrics = (preferred_metrics + [c for c in numeric_columns if c not in preferred_metrics])[:min(4, len(numeric_columns))]
    metric_columns = st.multiselect("检测指标（至少选择一项）", options=numeric_columns, default=default_metrics,
        help="自动优先勾选 CPU、内存、磁盘 I/O、网络、请求量和延迟等常见指标，可自行调整。")
with config_col3:
    duplicate_strategy = st.selectbox("重复时间戳处理", ["保留全部记录", "按时间戳聚合（数值取均值）", "删除重复时间戳（保留第一条）"],
        help="仅在选择时间列时生效；聚合策略会对数值指标取均值，其他字段保留首条。")
    invalid_time_strategy = st.selectbox("无效时间戳处理", ["保留记录", "删除无效时间戳行"],
        help="当时间列无法解析时，可选择保留原始行或删除无效时间戳记录。")

prep_col1, prep_col2 = st.columns([1, 1], gap="large")
with prep_col1:
    resample_rule = st.selectbox("时间重采样", ["不重采样", "1min", "5min", "10min", "30min", "1h"], index=0,
        help="统一不规则采样间隔；每个时间桶内数值指标取均值，时间列必须有效。")
with prep_col2:
    remove_negative = st.checkbox("将负数监控值视为无效值", value=False,
        help="适用于 CPU 使用率、内存占用率、计数等不应为负数的指标；若选中指标本身允许负值，请勿开启。")

time_column = None if selected_time_option == time_options[0] else selected_time_option
if not metric_columns:
    st.warning("请至少选择一个数值指标。")
    st.stop()

analysis_dataframe = dataframe.copy()
for metric in metric_columns:
    analysis_dataframe[metric] = pd.to_numeric(analysis_dataframe[metric], errors="coerce")
invalid_time_count = 0
if time_column:
    parsed_time = pd.to_datetime(analysis_dataframe[time_column], errors="coerce")
    invalid_time_count = int(parsed_time.isna().sum())
    if invalid_time_count:
        st.warning(f"时间列中有 {invalid_time_count} 条无法解析的时间戳。可在右侧选择保留这些记录或删除无效时间戳行。")
    # Use parsed datetimes for consistent sorting/duplicate detection when possible.
    analysis_dataframe[time_column] = parsed_time
    if invalid_time_strategy == "删除无效时间戳行":
        analysis_dataframe = analysis_dataframe.loc[parsed_time.notna()].copy()
    if parsed_time.notna().any() and not analysis_dataframe.empty:
        if duplicate_strategy == "按时间戳聚合（数值取均值）":
            non_time = [c for c in analysis_dataframe.columns if c != time_column]
            aggregation = {c: ("mean" if c in metric_columns else "first") for c in non_time}
            analysis_dataframe = analysis_dataframe.groupby(time_column, dropna=False, as_index=False).agg(aggregation)
            analysis_dataframe = analysis_dataframe.sort_values(time_column, kind="stable", na_position="last").reset_index(drop=True)
        elif duplicate_strategy == "删除重复时间戳（保留第一条）":
            analysis_dataframe = analysis_dataframe.drop_duplicates(subset=[time_column], keep="first").reset_index(drop=True)
        else:
            analysis_dataframe = analysis_dataframe.sort_values(time_column, kind="stable", na_position="last").reset_index(drop=True)
else:
    if duplicate_strategy != "保留全部记录":
        st.caption("未选择时间列，重复时间戳处理策略不会生效。")
    if invalid_time_strategy != "保留记录":
        st.caption("未选择时间列，无效时间戳处理策略不会生效。")

if remove_negative:
    for metric in metric_columns:
        analysis_dataframe.loc[pd.to_numeric(analysis_dataframe[metric], errors="coerce") < 0, metric] = np.nan

if resample_rule != "不重采样":
    if not time_column:
        st.warning("重采样需要先选择时间列；当前暂不重采样。")
    else:
        valid_time_df = analysis_dataframe.loc[analysis_dataframe[time_column].notna()].copy()
        if valid_time_df.empty:
            st.warning("没有可用的有效时间戳，无法重采样。")
        else:
            other_columns = [c for c in valid_time_df.columns if c not in metric_columns and c != time_column]
            aggregations = {c: "mean" for c in metric_columns}
            aggregations.update({c: "first" for c in other_columns})
            analysis_dataframe = (valid_time_df.set_index(time_column).sort_index()
                .resample(resample_rule).agg(aggregations).reset_index())
            st.info(f"已按 {resample_rule} 重采样，当前数据 {len(analysis_dataframe):,} 行。")

if len(analysis_dataframe) < 5:
    st.warning("预处理后数据少于 5 行，检测结果可能不稳定。")
if len(analysis_dataframe) != len(dataframe):
    st.info(f"预处理后用于检测的数据为 {len(analysis_dataframe):,} 行（原始 {len(dataframe):,} 行）。")
with st.expander("查看原始数据预览", expanded=False):
    st.dataframe(dataframe.head(10), use_container_width=True, hide_index=True)

st.markdown('<div class="cc-section-label">03 异常检测与算法对比</div>', unsafe_allow_html=True)
current_key = make_result_key(file_bytes, time_column, metric_columns, algorithm, contamination, window_size, duplicate_strategy, invalid_time_strategy, normalization, resample_rule, remove_negative, window_step)
detect_col, config_summary_col = st.columns([1, 3], gap="large")
with detect_col:
    run_clicked = st.button("开始异常检测", type="primary", use_container_width=True, key="run_detection_button")
with config_summary_col:
    st.caption(f"当前配置：{algorithm} · 异常比例 {contamination:.0%} · 窗口 {window_size} · 步长 {window_step} · 归一化 {normalization} · 指标数 {len(metric_columns)}")

if run_clicked:
    try:
        with st.spinner("正在进行数据预处理与异常检测……"):
            result_df, elapsed_seconds, score_threshold = run_detection(analysis_dataframe, metric_columns, algorithm, contamination, window_size, normalization, window_step)
        st.session_state["tsad_result_df"] = result_df
        st.session_state["tsad_result_key"] = current_key
        st.session_state["tsad_elapsed_seconds"] = elapsed_seconds
        st.session_state["tsad_score_threshold"] = score_threshold
        st.success(f"检测完成，用时 {elapsed_seconds:.2f} 秒。")
    except Exception as exc:
        st.error(f"检测失败：{exc}")
        st.stop()

compare_col, compare_note = st.columns([1, 3], gap="large")
with compare_col:
    compare_clicked = st.button("运行三算法对比", use_container_width=True, key="compare_algorithms_button")
with compare_note:
    st.caption("对比 Isolation Forest、Spectral Residual 和 Matrix Profile；Matrix Profile 对数据规模有额外限制。")
if compare_clicked:
    comparison_rows = []
    comparison_results = {}
    try:
        progress = st.progress(0, text="正在运行算法对比……")
        for i, name in enumerate(CORE_ALGORITHMS):
            one_df, one_elapsed, one_threshold = run_detection(analysis_dataframe, metric_columns, name, contamination, window_size, normalization, window_step)
            comparison_results[name] = one_df
            comparison_rows.append({"算法": name, "异常点数": int(one_df["is_anomaly"].sum()),
                "异常比例": float(one_df["is_anomaly"].mean()), "检测耗时（秒）": round(one_elapsed, 4),
                "异常评分阈值": one_threshold})
            progress.progress((i + 1) / len(CORE_ALGORITHMS), text=f"已完成：{name}")
        progress.empty()
        st.session_state["tsad_comparison_df"] = pd.DataFrame(comparison_rows)
        st.session_state["tsad_comparison_results"] = comparison_results
        st.session_state["tsad_comparison_key"] = current_key
        st.success("三算法对比完成。")
    except Exception as exc:
        st.error(f"算法对比未能完成：{exc}")

if "tsad_result_df" not in st.session_state:
    st.info("确认数据与参数后，点击“开始异常检测”查看结果。")
    st.stop()
if st.session_state.get("tsad_result_key") != current_key:
    st.warning("数据或检测配置已发生变化。请重新运行检测，避免显示过期结果。")
    st.stop()

results = st.session_state["tsad_result_df"]
elapsed_seconds = st.session_state.get("tsad_elapsed_seconds", 0.0)
score_threshold = st.session_state.get("tsad_score_threshold", float(results["anomaly_score"].quantile(1 - contamination)))
st.markdown("#### 统一异常分数阈值")
score_threshold = st.slider("异常分数阈值（0–1）", min_value=0.0, max_value=1.0, value=float(np.clip(score_threshold, 0, 1)), step=0.01,
    help="所有算法的异常分数均归一化到 [0, 1]。低于该阈值视为正常，高于或等于该阈值视为异常。", key=f"score_threshold_{current_key[:10]}")
results["is_anomaly"] = results["anomaly_score"] >= score_threshold
results["anomaly_label"] = np.where(results["is_anomaly"], "异常", "正常")
results["异常类型"] = "正常"
for metric in metric_columns:
    series = pd.to_numeric(results[metric], errors="coerce")
    delta = series.diff().abs()
    delta_scale = float(delta.std()) if delta.notna().any() else 0.0
    spike_mask = results["is_anomaly"] & (delta > max(delta_scale * 3, 1e-9))
    results.loc[spike_mask, "异常类型"] = "Spike（尖峰突变）"
    rolling = series.rolling(max(3, window_size), min_periods=2).mean()
    drift_mask = results["is_anomaly"] & (results["异常类型"] == "正常") & (rolling.diff().abs() > max(float(rolling.diff().std()) * 1.5 if rolling.diff().notna().any() else 0, 1e-9))
    results.loc[drift_mask, "异常类型"] = "Drift（缓慢漂移）"
results.loc[results["is_anomaly"] & (results["异常类型"] == "正常"), "异常类型"] = "Contextual（上下文异常）"

st.markdown('<div class="cc-section-label">04 异常分析看板</div>', unsafe_allow_html=True)
total_count = len(results)
anomaly_count = int(results["is_anomaly"].sum())
normal_count = total_count - anomaly_count
anomaly_rate = anomaly_count / total_count if total_count else 0.0
metric1, metric2, metric3, metric4 = st.columns(4, gap="medium")
metric1.metric("检测样本数", f"{total_count:,}")
metric2.metric("异常点数量", f"{anomaly_count:,}")
metric3.metric("实际异常比例", f"{anomaly_rate:.2%}")
metric4.metric("检测耗时", f"{elapsed_seconds:.3f} 秒")
st.caption(f"算法：{algorithm} · 窗口大小：{window_size} · 归一化：{normalization} · 设定异常比例：{contamination:.0%} · 统一分数阈值：{score_threshold:.2f}")
if label_column:
    truth_raw = results[label_column].astype(str).str.strip().str.lower()
    positive_values = {"1", "true", "yes", "异常", "anomaly", "outlier", "abnormal"}
    negative_values = {"0", "false", "no", "正常", "normal", "inlier"}
    truth = truth_raw.map(lambda x: True if x in positive_values else (False if x in negative_values else np.nan))
    valid = truth.notna()
    if valid.sum() > 0 and truth[valid].nunique() == 2:
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_true = truth[valid].astype(bool).to_numpy()
        y_pred = results.loc[valid, "is_anomaly"].astype(bool).to_numpy()
        ev1, ev2, ev3 = st.columns(3)
        ev1.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
        ev2.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
        ev3.metric("F1", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
    else:
        st.info("已选择标签列，但未识别到同时包含正常与异常的有效 0/1 或文本标签，暂不计算评估指标。")

intervals = anomaly_intervals(results, time_column)
if not intervals.empty:
    st.markdown("#### 异常时间区间")
    st.dataframe(intervals, use_container_width=True, hide_index=True)
else:
    st.info("当前配置下未检测到异常区间。")

st.markdown('<div class="cc-section-label">05 时序曲线与异常可视化</div>', unsafe_allow_html=True)
chart_mode = st.radio("时序展示模式", ["分指标子图（推荐）", "合并曲线"], horizontal=True)
chart_mode_internal = "分指标子图" if chart_mode.startswith("分指标") else "合并曲线"
chart_left, chart_right = st.columns(2, gap="large")
with chart_left:
    st.plotly_chart(build_multi_metric_chart(results, time_column, metric_columns, chart_mode_internal), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
with chart_right:
    st.plotly_chart(build_score_chart(results, time_column, score_threshold), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
st.plotly_chart(build_metric_distribution(results, metric_columns), use_container_width=True, config={"displaylogo": False})

if st.session_state.get("tsad_comparison_key") == current_key and "tsad_comparison_df" in st.session_state:
    st.markdown("#### 三种时序检测算法对比")
    compare_df = st.session_state["tsad_comparison_df"].copy()
    comparison_results = st.session_state.get("tsad_comparison_results", {})
    # Re-label all comparison outputs with the same interactive threshold for fairer comparison.
    for name, compare_result in comparison_results.items():
        compare_result["is_anomaly"] = compare_result["anomaly_score"] >= score_threshold
        compare_result["anomaly_label"] = np.where(compare_result["is_anomaly"], "异常", "正常")
        row_mask = compare_df["算法"] == name
        compare_df.loc[row_mask, "异常点数"] = int(compare_result["is_anomaly"].sum())
        compare_df.loc[row_mask, "异常比例"] = float(compare_result["is_anomaly"].mean())
        compare_df.loc[row_mask, "异常评分阈值"] = score_threshold
    st.dataframe(compare_df, use_container_width=True, hide_index=True)
    if len(comparison_results) >= 2:
        overlap_rows = []
        names = list(comparison_results)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = comparison_results[names[i]]["is_anomaly"].to_numpy(dtype=bool)
                b = comparison_results[names[j]]["is_anomaly"].to_numpy(dtype=bool)
                union = int(np.logical_or(a, b).sum())
                intersection = int(np.logical_and(a, b).sum())
                overlap_rows.append({"算法 A": names[i], "算法 B": names[j], "共同异常点": intersection, "异常重合度（Jaccard）": intersection / union if union else 1.0})
        st.markdown("##### 算法异常重合度")
        st.dataframe(pd.DataFrame(overlap_rows), use_container_width=True, hide_index=True)
    compare_fig = go.Figure()
    compare_fig.add_trace(go.Bar(x=compare_df["算法"], y=compare_df["异常点数"], name="异常点数", marker_color="#2563eb"))
    compare_fig.update_layout(template="plotly_white", title="不同算法检出的异常点数量", yaxis_title="异常点数", margin=dict(l=18, r=18, t=55, b=18))
    st.plotly_chart(compare_fig, use_container_width=True, config={"displaylogo": False})
    st.download_button("下载算法对比 CSV", data=compare_df.to_csv(index=False).encode("utf-8-sig"), file_name="tsad_algorithm_comparison.csv", mime="text/csv")

st.markdown('<div class="cc-section-label">06 异常记录与结果导出</div>', unsafe_allow_html=True)
table_controls_col, table_info_col = st.columns([1, 3], gap="medium")
with table_controls_col:
    only_anomalies = st.checkbox("仅显示异常记录", value=True)
with table_info_col:
    st.caption(f"共 {anomaly_count:,} 条异常、{normal_count:,} 条正常。缺失值仅在模型输入阶段填补；原始指标列保留预处理后的原始值。")
displayed = results[results["is_anomaly"]] if only_anomalies else results
display_columns = ([time_column] if time_column and time_column in displayed.columns else []) + metric_columns + ["anomaly_score", "anomaly_label", "异常类型"]
display_columns = list(dict.fromkeys(display_columns))
if only_anomalies and not displayed.empty:
    displayed = displayed.sort_values("anomaly_score", ascending=False)
st.dataframe(displayed[display_columns], use_container_width=True, hide_index=True)
full_csv = results.to_csv(index=False).encode("utf-8-sig")
anomaly_csv = results[results["is_anomaly"]].to_csv(index=False).encode("utf-8-sig")
download_col1, download_col2, download_col3 = st.columns([1, 1, 2], gap="medium")
with download_col1:
    st.download_button("下载完整检测结果", data=full_csv, file_name="tsad_detection_results.csv", mime="text/csv", use_container_width=True)
with download_col2:
    st.download_button("下载异常记录", data=anomaly_csv, file_name="tsad_anomaly_records.csv", mime="text/csv", use_container_width=True)
with download_col3:
    st.caption("CSV 使用 UTF-8 BOM 编码，便于在 Excel 中查看中文字段。")

parameter_snapshot = pd.DataFrame([{"算法": algorithm, "预期异常比例": contamination, "最终分数阈值": score_threshold,
    "窗口大小": window_size, "窗口步长": window_step, "归一化": normalization, "重采样": resample_rule, "负值过滤": remove_negative,
    "时间列": time_column or "样本序号", "检测指标": ", ".join(metric_columns), "检测时间": pd.Timestamp.now().isoformat(timespec="seconds")}])
st.download_button("下载本次参数快照 CSV", data=parameter_snapshot.to_csv(index=False).encode("utf-8-sig"), file_name="tsad_parameter_snapshot.csv", mime="text/csv")
html_chart = build_multi_metric_chart(results, time_column, metric_columns, "分指标子图").to_html(full_html=False, include_plotlyjs="cdn")
html_score = build_score_chart(results, time_column, score_threshold).to_html(full_html=False, include_plotlyjs=False)
html_report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TSAD 异常检测报告</title><style>body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#f8fafc;color:#111827;max-width:1200px;margin:32px auto;padding:0 20px}}section{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:24px;margin:18px 0}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #e5e7eb;padding:8px;text-align:left}}</style></head><body><h1>TSAD 时间序列异常检测报告</h1><section><h2>检测摘要</h2>{parameter_snapshot.to_html(index=False)}<p>样本数：{total_count}；异常点：{anomaly_count}；异常比例：{anomaly_rate:.2%}；耗时：{elapsed_seconds:.3f} 秒</p></section><section><h2>时序图</h2>{html_chart}</section><section><h2>异常分数</h2>{html_score}</section><section><h2>异常时间区间</h2>{intervals.to_html(index=False)}</section><section><h2>异常记录</h2>{results.loc[results['is_anomaly'], display_columns].to_html(index=False)}</section></body></html>"""
st.download_button("下载交互式 HTML 看板报告", data=html_report.encode("utf-8"), file_name="tsad_anomaly_report.html", mime="text/html")

st.markdown("---")
st.caption("TSAD Dashboard · 面向多变量监控数据的异常分析演示平台。Isolation Forest、LOF、KNN 属于通用异常检测器；Spectral Residual 与 Matrix Profile 在本项目中以可复现的基础实现提供，结果需结合业务场景验证。")
