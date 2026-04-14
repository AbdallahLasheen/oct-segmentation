import streamlit as st
import numpy as np
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from fpdf import FPDF
import time
import io
import datetime

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — PAGE CONFIG & STYLING
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VisionOCT Pro | Alamein International University",
    layout="wide",
    page_icon="👁",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ─── Base ─── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.stApp { background-color: #0A0E1A !important; }
.main .block-container {
    background-color: #0A0E1A !important;
    padding: 1.2rem 1.5rem !important;
    max-width: 100% !important;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background-color: #0F1525 !important;
    border-right: 1px solid #1E2840 !important;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; font-family: 'DM Sans', sans-serif !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #F1F5F9 !important; font-size: 13px !important; font-weight: 600 !important;
    letter-spacing: 0.5px !important; text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #141929 !important; border: 1px solid #1E2840 !important;
    color: #F1F5F9 !important; border-radius: 8px !important; font-size: 13px !important;
}
[data-testid="stSidebar"] hr { border-color: #1E2840 !important; }

/* ─── Hero topbar ─── */
.visionoct-topbar {
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(135deg, #0F1525 0%, #111827 100%);
    border: 1px solid #1E2840; border-radius: 16px;
    padding: 14px 24px; margin-bottom: 20px;
    animation: fadeSlideDown 0.6s ease both;
}
@keyframes fadeSlideDown { from { opacity:0; transform:translateY(-16px); } to { opacity:1; transform:translateY(0); } }
@keyframes fadeSlideUp   { from { opacity:0; transform:translateY(16px);  } to { opacity:1; transform:translateY(0); } }
@keyframes fadeIn        { from { opacity:0; }                              to { opacity:1; } }
@keyframes shimmer       { 0%,100% { opacity:1; } 50% { opacity:0.35; } }
@keyframes scanline {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(400%); }
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(37,99,235,0.4); }
    50%      { box-shadow: 0 0 0 8px rgba(37,99,235,0); }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.topbar-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #1D4ED8, #0D9488);
    border-radius: 14px; display: flex; align-items: center;
    justify-content: center; font-size: 28px;
    animation: pulse 3s infinite;
}
.topbar-title { font-size: 28px; font-weight: 700; color: #F8FAFC; letter-spacing: -0.5px; }
.topbar-sub   { font-size: 13px; color: #94A3B8; margin-top: 2px; }
.topbar-badges { display: flex; gap: 8px; align-items: center; }
.badge-live {
    padding: 4px 12px; border-radius: 20px;
    background: rgba(13,148,136,0.18); color: #6EE7B7;
    border: 1px solid rgba(13,148,136,0.35); font-size: 11px;
    font-family: 'DM Mono', monospace; font-weight: 500;
    animation: shimmer 2.5s infinite;
}
.badge-model {
    padding: 4px 12px; border-radius: 20px;
    background: rgba(37,99,235,0.18); color: #BFDBFE;
    border: 1px solid rgba(37,99,235,0.35); font-size: 11px;
    font-family: 'DM Mono', monospace;
}
.badge-dr {
    padding: 4px 12px; border-radius: 20px;
    background: #141929; color: #CBD5E1; border: 1px solid #263050; font-size: 11px;
}

/* ─── Section headers ─── */
.section-header {
    font-size: 12px; font-weight: 600; color: #F1F5F9; margin-bottom: 10px;
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 8px; border-bottom: 1px solid #1E2840;
}
.section-tag {
    font-size: 10px; font-family: 'DM Mono', monospace; color: #64748B;
    padding: 2px 7px; background: #1E2840; border-radius: 4px;
}

/* ─── Metric cards ─── */
.metric-card {
    background: #141929; border: 1px solid #1E2840;
    border-radius: 12px; padding: 14px 16px;
    animation: fadeSlideUp 0.5s ease both;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #2563EB; }
.metric-label {
    font-size: 10px; font-weight: 500; color: #64748B;
    letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 6px;
}
.metric-value {
    font-size: 24px; font-weight: 600; font-family: 'DM Mono', monospace;
    letter-spacing: -0.5px; margin: 0;
}
.metric-value-sm {
    font-size: 16px; font-weight: 600; font-family: 'DM Mono', monospace;
    color: #E2E8F0; margin: 0;
}
.metric-sub { font-size: 11px; color: #64748B; margin-top: 4px; }

/* ─── Progress bars ─── */
.fluid-bar-wrap { margin: 8px 0; }
.fluid-bar-label { display: flex; justify-content: space-between; margin-bottom: 5px; }
.fluid-bar-text  { font-size: 11px; color: #64748B; }
.fluid-bar-bg    { height: 6px; background: #1E2840; border-radius: 3px; overflow: hidden; }
.fluid-bar-fill  {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, #0D9488, #2563EB);
    background-size: 200% 100%;
    animation: gradientShift 3s ease infinite;
}

/* ─── Finding tags ─── */
.findings-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.finding-tag {
    padding: 3px 9px; border-radius: 5px; font-size: 11px; font-weight: 500;
    font-family: 'DM Mono', monospace; letter-spacing: 0.3px;
}
.tag-irf  { background: rgba(170,223,235,0.12); color: #7DD3E8; border: 1px solid rgba(170,223,235,0.25); }
.tag-srf  { background: rgba(42,125,209,0.12);  color: #60A5FA; border: 1px solid rgba(42,125,209,0.25); }
.tag-ped  { background: rgba(255,53,94,0.12);   color: #FB7185; border: 1px solid rgba(255,53,94,0.25); }
.tag-hrf  { background: rgba(240,120,240,0.12); color: #E879F9; border: 1px solid rgba(240,120,240,0.25); }
.tag-shrm { background: rgba(204,153,51,0.12);  color: #FCD34D; border: 1px solid rgba(204,153,51,0.25); }
.tag-layer{ background: rgba(99,102,241,0.12);  color: #A5B4FC; border: 1px solid rgba(99,102,241,0.25); }

/* ─── Scan panel ─── */
.scan-panel-header {
    padding: 8px 14px; border-bottom: 1px solid #1E2840;
    display: flex; align-items: center; justify-content: space-between;
}
.scan-panel-title { font-size: 11px; font-weight: 500; color: #CBD5E1; }

/* ─── Patient bar ─── */
.patient-bar {
    background: #141929; border: 1px solid #1E2840; border-radius: 12px;
    padding: 12px 18px; display: flex; align-items: center;
    gap: 16px; margin-bottom: 16px; flex-wrap: wrap;
    animation: fadeIn 0.7s ease both;
}
.patient-name { font-size: 15px; font-weight: 600; color: #F1F5F9; }
.patient-mrn  { font-size: 11px; color: #64748B; font-family: 'DM Mono', monospace; }
.patient-meta { display: flex; gap: 16px; margin-left: auto; flex-wrap: wrap; }
.meta-item    { font-size: 12px; color: #94A3B8; }
.meta-item span { color: #64748B; margin-right: 4px; }
.status-complete {
    padding: 4px 12px; border-radius: 20px;
    background: rgba(13,148,136,0.15); color: #6EE7B7;
    border: 1px solid rgba(13,148,136,0.3); font-size: 10px;
    font-family: 'DM Mono', monospace;
}

/* ─── Report card ─── */
.report-card {
    background: #141929; border: 1px solid #1E2840;
    border-radius: 12px; padding: 18px; margin-bottom: 16px;
}
.ai-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; color: #BFDBFE; font-weight: 500; margin-bottom: 14px;
}
.ai-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #2563EB;
    display: inline-block; animation: shimmer 1.8s infinite;
}
.report-body {
    font-size: 12px; line-height: 1.8; color: #CBD5E1;
    border-left: 2px solid #2563EB; padding-left: 14px;
    font-family: 'DM Mono', monospace; white-space: pre-wrap;
}

/* ─── Layer legend badge ─── */
.layer-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border-radius: 6px; font-size: 11px;
    font-family: 'DM Mono', monospace; margin: 3px;
}

/* ─── Composite info box ─── */
.composite-info {
    background: linear-gradient(135deg,rgba(37,99,235,0.1),rgba(13,148,136,0.1));
    border: 1px solid rgba(37,99,235,0.25); border-radius: 10px;
    padding: 12px 16px; margin: 10px 0;
    font-size: 12px; color: #CBD5E1; line-height: 1.7;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    background: #0F1525 !important; border-bottom: 1px solid #1E2840 !important;
    gap: 0 !important; padding: 0 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important; border-radius: 0 !important;
    color: #64748B !important; font-size: 12px !important; font-weight: 500 !important;
    padding: 10px 16px !important;
}
.stTabs [aria-selected="true"] {
    color: #BFDBFE !important; border-bottom: 2px solid #2563EB !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { background: #0A0E1A !important; padding: 16px 0 !important; }

/* ─── Widgets ─── */
[data-testid="stFileUploader"] {
    background: #141929 !important; border: 1px dashed #263050 !important; border-radius: 10px !important;
}
[data-testid="stFileUploader"] * { color: #CBD5E1 !important; }
.stButton > button {
    background: #2563EB !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-size: 13px !important; font-weight: 500 !important;
    padding: 8px 18px !important; transition: background 0.2s;
}
.stButton > button:hover { background: #1D4ED8 !important; }
.stDownloadButton > button {
    background: #1A2035 !important; color: #BFDBFE !important;
    border: 1px solid #263050 !important; border-radius: 8px !important; font-size: 13px !important;
}
.stTextInput input, .stNumberInput input {
    background: #141929 !important; border: 1px solid #263050 !important;
    color: #F1F5F9 !important; border-radius: 8px !important; font-size: 13px !important;
    caret-color: #F1F5F9 !important;
}
.stTextInput input::placeholder, .stNumberInput input::placeholder { color: #475569 !important; }
.stSelectbox > div > div {
    background: #141929 !important; border: 1px solid #263050 !important;
    border-radius: 8px !important; color: #F1F5F9 !important;
}
[data-baseweb="select"] * { background: #141929 !important; color: #F1F5F9 !important; }
[data-baseweb="popover"], [data-baseweb="menu"] { background: #1A2035 !important; border: 1px solid #263050 !important; }
[data-baseweb="option"]:hover { background: #263050 !important; }
.stTextArea textarea {
    background: #141929 !important; border: 1px solid #263050 !important;
    color: #F1F5F9 !important; border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important; font-size: 12px !important; line-height: 1.7 !important;
}
label, [data-testid="stWidgetLabel"] {
    color: #CBD5E1 !important; font-size: 12px !important; font-weight: 400 !important;
}
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background: #141929 !important; border: 1px solid #1E2840 !important;
    border-radius: 8px !important; color: #CBD5E1 !important; font-size: 12px !important;
}
.streamlit-expanderContent, [data-testid="stExpander"] > div:last-child {
    background: #0F1525 !important; border: 1px solid #1E2840 !important;
    border-top: none !important; border-radius: 0 0 8px 8px !important;
}
[data-testid="stChatInput"] {
    background: #141929 !important; border: 1px solid #1E2840 !important; border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea { color: #F1F5F9 !important; background: #141929 !important; }

/* ─── Typography defaults ─── */
p, span, div, li, small { color: #CBD5E1; }
h1, h2, h3, h4, h5, h6 { color: #F1F5F9 !important; }
strong, b { color: #F1F5F9 !important; }
code { color: #BFDBFE !important; background: #1A2035 !important; }
[data-testid="stMarkdownContainer"] p      { color: #CBD5E1 !important; }
[data-testid="stMarkdownContainer"] strong { color: #F1F5F9 !important; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3    { color: #F1F5F9 !important; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1E2840; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #263050; }

/* ─── Hide Streamlit chrome ─── */
#MainMenu, footer, header { visibility: hidden; }

/* ─── Dataframe ─── */
[data-testid="stDataFrame"] { animation: fadeSlideUp 0.4s ease both; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — CLINICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════

# ── Lesion / Feature model (8 classes) ──
LABEL_MAP = [
    ("Background",          (0,   0,   0  )),
    ("Drusenoid PED",       (248, 231, 180)),
    ("Fibrovascular PED",   (255, 53,  94 )),
    ("HRF",                 (240, 120, 240)),
    ("IRF",                 (170, 223, 235)),
    ("PH",                  (51,  221, 255)),
    ("SHRM",                (204, 153, 51 )),
    ("SRF",                 (42,  125, 209)),
]
FLUID_CLASSES  = ["IRF", "SRF", "Drusenoid PED", "Fibrovascular PED"]
CLASS_COLORS   = np.array([c for _, c in LABEL_MAP], dtype=np.uint8)

# ── Layer model (4 classes: background + 3 layers) ──
LAYER_MAP = [
    ("Background", (0,   0,   0  )),
    ("Choroid",    (255, 180, 50 )),   # warm amber
    ("NSR",        (80,  200, 120)),   # soft green
    ("RPE",        (120, 80,  220)),   # violet
]
LAYER_COLORS = np.array([c for _, c in LAYER_MAP], dtype=np.uint8)
LAYER_NAMES  = [n for n, _ in LAYER_MAP]

TAG_CLASSES = {
    "IRF":              "tag-irf",
    "SRF":              "tag-srf",
    "Fibrovascular PED":"tag-ped",
    "Drusenoid PED":    "tag-ped",
    "HRF":              "tag-hrf",
    "SHRM":             "tag-shrm",
}

SEVERITY_WEIGHTS = {
    "IRF":              1.5,
    "SRF":              1.2,
    "Fibrovascular PED":2.0,
    "Drusenoid PED":    1.0,
    "HRF":              0.8,
    "SHRM":             1.3,
    "PH":               0.5,
}

PIXEL_TO_MM2 = (6.0 / 256.0) ** 2
TOTAL_PX     = 256 * 256


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL LOADING
# ══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_lesion_model():
    """EfficientNet-B0 U-Net — 8-class lesion/feature segmentation."""
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=len(LABEL_MAP),
    )
    try:
        with open("unet_oct_best_v2.pth", "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["weights"])
    except Exception as e:
        st.error(f"Lesion model load failed: {e}")
    model.eval()
    return model


@st.cache_resource
def load_layer_model():
    """EfficientNet-B5 Attention U-Net — 4-class retinal layer segmentation."""
    model = smp.Unet(
        encoder_name="efficientnet-b5",
        encoder_weights=None,
        in_channels=3,
        classes=len(LAYER_MAP),
        decoder_attention_type="scse",
    )
    try:
        with open("layer_seg_best.pth", "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
        state = ckpt.get("weights", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state)
    except Exception as e:
        st.warning(f"Layer model not found ({e}). Layer-aware measurements will use full-image fallback.")
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — INFERENCE & COMPOSITE RENDERING
# ══════════════════════════════════════════════════════════════════

def _preprocess(img: Image.Image, size: int = 256):
    resized = img.resize((size, size), Image.BILINEAR)
    return TF.normalize(
        TF.to_tensor(resized),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ).unsqueeze(0)


def run_lesion_model(img: Image.Image, model) -> np.ndarray:
    """Returns (256,256) integer mask with class indices."""
    tensor = _preprocess(img, 256)
    with torch.no_grad():
        return model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()


def run_layer_model(img: Image.Image, model) -> np.ndarray:
    """Returns (256,256) integer mask with layer indices (0=BG,1=Choroid,2=NSR,3=RPE)."""
    tensor = _preprocess(img, 256)
    with torch.no_grad():
        return model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()


def build_composite_overlay(
    original: Image.Image,
    lesion_mask: np.ndarray,
    layer_mask: np.ndarray,
    layer_alpha: float = 0.35,
    lesion_alpha: float = 0.65,
) -> Image.Image:
    """
    Compose three layers into one image:
      1. Original grayscale B-scan (base)
      2. Layer segmentation (Choroid / NSR / RPE) — semi-transparent fill
      3. Lesion/feature mask rendered on top — stronger opacity
    """
    orig_resized = original.resize((256, 256), Image.BILINEAR).convert("RGBA")

    # ── Layer fill ──
    layer_rgb  = LAYER_COLORS[layer_mask].astype(np.uint8)
    layer_rgba = np.concatenate(
        [layer_rgb, np.full((*layer_mask.shape, 1), int(255 * layer_alpha), dtype=np.uint8)],
        axis=-1,
    )
    layer_img = Image.fromarray(layer_rgba, mode="RGBA")

    # ── Lesion overlay (non-background pixels only) ──
    lesion_rgb  = CLASS_COLORS[lesion_mask].astype(np.uint8)
    lesion_a    = np.where(lesion_mask == 0, 0, int(255 * lesion_alpha)).astype(np.uint8)
    lesion_rgba = np.concatenate([lesion_rgb, lesion_a[:, :, None]], axis=-1)
    lesion_img  = Image.fromarray(lesion_rgba, mode="RGBA")

    # ── Composite ──
    composite = Image.alpha_composite(orig_resized, layer_img)
    composite = Image.alpha_composite(composite, lesion_img)
    return composite.convert("RGB").resize(original.size, Image.NEAREST)


def compute_layer_aware_measurements(
    lesion_mask: np.ndarray,
    layer_mask: np.ndarray,
) -> dict:
    """
    For each lesion class, compute:
      - total pixels
      - which retinal layer it predominantly occupies
      - area as fraction of that layer (not the whole image)
    """
    results = {}
    for idx, (name, _) in enumerate(LABEL_MAP):
        if name == "Background":
            continue
        lesion_px_mask = lesion_mask == idx
        px_total = int(lesion_px_mask.sum())

        # Determine which layer each lesion pixel falls in
        layer_dist = {}
        for l_idx, (l_name, _) in enumerate(LAYER_MAP):
            if l_name == "Background":
                continue
            overlap = int((lesion_px_mask & (layer_mask == l_idx)).sum())
            if overlap > 0:
                layer_dist[l_name] = overlap

        dominant_layer = max(layer_dist, key=layer_dist.get) if layer_dist else "Unknown"
        dominant_layer_px = layer_dist.get(dominant_layer, 0)

        # Layer total pixels (denominator for layer-relative coverage)
        if dominant_layer != "Unknown":
            l_idx_dom = LAYER_NAMES.index(dominant_layer)
            layer_total_px = int((layer_mask == l_idx_dom).sum())
        else:
            layer_total_px = TOTAL_PX  # fallback

        layer_pct = round(dominant_layer_px / max(layer_total_px, 1) * 100, 2)
        image_pct = round(px_total / TOTAL_PX * 100, 2)
        area_mm2  = round(px_total * PIXEL_TO_MM2, 3)

        results[name] = {
            "px":              px_total,
            "area_mm2":        area_mm2,
            "image_pct":       image_pct,
            "dominant_layer":  dominant_layer,
            "layer_overlap_px":dominant_layer_px,
            "layer_pct":       layer_pct,
            "layer_dist":      layer_dist,
        }
    return results


def analyze_scan(img: Image.Image, lesion_model, layer_model):
    orig_size = img.size

    lesion_mask = run_lesion_model(img, lesion_model)
    layer_mask  = run_layer_model(img, layer_model)

    # ── Basic stats ──
    stats = {name: int((lesion_mask == i).sum()) for i, (name, _) in enumerate(LABEL_MAP)}

    # ── Fluid index ──
    fluid_px  = sum(stats[cls] for cls in FLUID_CLASSES)
    fluid_idx = round((fluid_px / TOTAL_PX) * 100, 2)

    # ── Severity score ──
    weighted_sum = sum(stats.get(cls, 0) * w for cls, w in SEVERITY_WEIGHTS.items())
    max_possible = TOTAL_PX * 2.0
    sev_score    = round(min((weighted_sum / max_possible) * 100, 100), 1)

    if sev_score > 60:
        sev_grade, sev_color = "SEVERE",   "#F87171"
    elif sev_score > 30:
        sev_grade, sev_color = "MODERATE", "#FCD34D"
    elif sev_score > 5:
        sev_grade, sev_color = "MILD",     "#34D399"
    else:
        sev_grade, sev_color = "MINIMAL",  "#94A3B8"

    # ── Layer-aware measurements ──
    measurements = compute_layer_aware_measurements(lesion_mask, layer_mask)

    # ── Layer pixel counts ──
    layer_stats = {name: int((layer_mask == i).sum()) for i, (name, _) in enumerate(LAYER_MAP)}

    # ── Composite image ──
    composite_img = build_composite_overlay(img, lesion_mask, layer_mask)

    # ── Plain lesion mask (for legacy display) ──
    lesion_rgb = Image.fromarray(CLASS_COLORS[lesion_mask].astype(np.uint8)).resize(orig_size, Image.NEAREST)
    layer_rgb  = Image.fromarray(LAYER_COLORS[layer_mask].astype(np.uint8)).resize(orig_size, Image.NEAREST)

    return (
        lesion_rgb, layer_rgb, composite_img,
        stats, layer_stats,
        fluid_idx, sev_score, sev_grade, sev_color,
        measurements,
    )


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — AI & PDF
# ══════════════════════════════════════════════════════════════════

def get_groq_ai_response(prompt: str) -> str:
    client = OpenAI(
        api_key=st.secrets["GROK_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=[
            {"role": "system", "content": (
                "You are a senior retinal specialist assistant. Write a detailed but concise clinical report "
                "that fits in ONE page. Do NOT include Patient ID, Date, or Modality fields. "
                "Start directly with the clinical summary. Use clear sections: Clinical Summary, "
                "Key Findings, Impression, and Recommendations. Do NOT use markdown bold (**) markers."
            )},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def _safe(text: str) -> str:
    """Replace non-latin1 characters so FPDF (latin-1) does not crash."""
    replacements = {
        "\u2014": "-",  # em dash
        "\u2013": "-",  # en dash
        "\u2022": "*",  # bullet
        "\u00b2": "2",  # superscript 2
        "\u00b0": " deg",
        "%": "%",       # percent is fine in latin-1
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def create_medical_pdf(p_info, dr_name, report_text, visit_summary=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    try:
        pdf.image("uni_logo.png", x=10, y=8, w=25)
    except Exception:
        pass

    pdf.set_font("Arial", "B", 14)
    pdf.set_x(40)
    pdf.cell(0, 10, _safe("ALAMEIN INTERNATIONAL UNIVERSITY - AIU"), ln=True)
    pdf.set_font("Arial", size=9)
    pdf.set_x(40)
    pdf.cell(0, 5, _safe("Center for Precision Ophthalmic Intelligence"), ln=True)
    pdf.ln(10)

    pdf.set_fill_color(245, 247, 250)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 7, _safe(" PATIENT INFORMATION"), ln=True, fill=True)
    pdf.set_font("Arial", size=9)
    pdf.cell(95, 6, _safe(f" Name: {p_info['name']}"), border=1)
    pdf.cell(95, 6, _safe(f" ID: {p_info['id']}"),     border=1, ln=True)
    pdf.cell(95, 6, _safe(f" Age: {p_info['age']}"),   border=1)
    pdf.cell(95, 6, _safe(f" Gender: {p_info['gender']}"), border=1, ln=True)

    if visit_summary:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 7, _safe(" VISIT SUMMARY"), ln=True, fill=True)
        pdf.set_font("Arial", size=9)
        for vs in visit_summary:
            line = (
                f"  {vs['Visit']} ({vs['Date']}) - "
                f"Avg Severity: {vs['Avg Severity Score']}/100 [{vs['Grade']}]  |  "
                f"Avg Fluid: {vs['Avg Fluid Index (%)']}%"
            )
            pdf.cell(0, 6, _safe(line), ln=True)

    pdf.ln(8)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 7, _safe(" CLINICAL FINDINGS & AI ANALYSIS"), ln=True)
    pdf.set_font("Arial", size=9)
    safe_report = _safe(report_text[:2800])
    pdf.multi_cell(0, 5, safe_report)

    pdf.set_y(250)
    pdf.line(10, 250, 200, 250)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(0, 5, _safe("Digitally Verified by:"), ln=True, align="R")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 5, _safe(f"Dr. {dr_name}"), ln=True, align="R")
    pdf.set_font("Arial", size=7)
    pdf.cell(0, 4, _safe("Consultant Specialist | AIU Clinical Diagnostic Suite"), ln=True, align="R")

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — UI HELPERS
# ══════════════════════════════════════════════════════════════════

def fluid_color(val):
    if val > 1.5: return "#F87171"
    if val > 1.0: return "#FCD34D"
    return "#34D399"

def severity_color(val):
    if val > 60:  return "#F87171"
    if val > 30:  return "#FCD34D"
    if val > 5:   return "#34D399"
    return "#94A3B8"

def metric_card(label, value, sub, color="#E2E8F0", small=False):
    sz = "metric-value-sm" if small else "metric-value"
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="{sz}" style="color:{color}">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>"""

def fluid_bar(pct, color):
    width = min(pct * 15, 100)
    return f"""
    <div class="fluid-bar-wrap">
      <div class="fluid-bar-label">
        <span class="fluid-bar-text">Fluid Index</span>
        <span style="font-size:11px;font-family:'DM Mono',monospace;color:{color}">{pct}%</span>
      </div>
      <div class="fluid-bar-bg"><div class="fluid-bar-fill" style="width:{width}%"></div></div>
    </div>"""

def severity_bar(score, color):
    return f"""
    <div class="fluid-bar-wrap">
      <div class="fluid-bar-label">
        <span class="fluid-bar-text">Severity Score</span>
        <span style="font-size:11px;font-family:'DM Mono',monospace;color:{color}">{score}/100</span>
      </div>
      <div class="fluid-bar-bg"><div class="fluid-bar-fill" style="width:{score}%;background:{color}"></div></div>
    </div>"""

def layer_color_hex(name):
    mapping = {"Choroid": "#FFB432", "NSR": "#50C878", "RPE": "#7850DC", "Unknown": "#64748B"}
    return mapping.get(name, "#94A3B8")


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — SIDEBAR
# ══════════════════════════════════════════════════════════════════

st.sidebar.markdown("**CLINICIAN SETUP**")
dr_input = st.sidebar.text_input("Physician Name", value="Ahmed Younis")

st.sidebar.markdown("---")
st.sidebar.markdown("**OCT DEVICE CALIBRATION**")
device_choice = st.sidebar.selectbox(
    "Device",
    ["Zeiss Cirrus (6mm)", "Heidelberg Spectralis (6mm)", "Topcon DRI (7mm)"],
)
device_px_map = {
    "Zeiss Cirrus (6mm)":         (6.0 / 256.0) ** 2,
    "Heidelberg Spectralis (6mm)":(6.0 / 320.0) ** 2,
    "Topcon DRI (7mm)":           (7.0 / 512.0) ** 2,
}
PIXEL_TO_MM2 = device_px_map[device_choice]

st.sidebar.markdown("---")
st.sidebar.markdown("**LESION LEGEND**")
for name, (r, g, b) in LABEL_MAP:
    if name == "Background":
        continue
    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;">'
        f'<div style="width:10px;height:10px;border-radius:50%;background:rgb({r},{g},{b});flex-shrink:0"></div>'
        f'<span style="font-size:12px;color:#CBD5E1">{name}</span></div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**LAYER LEGEND**")
for name, (r, g, b) in LAYER_MAP:
    if name == "Background":
        continue
    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;">'
        f'<div style="width:10px;height:10px;border-radius:3px;background:rgb({r},{g},{b});flex-shrink:0"></div>'
        f'<span style="font-size:12px;color:#CBD5E1">{name}</span></div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:11px;color:#64748B;line-height:2">'
    'Lesion Model: EfficientNet-B0 U-Net<br>'
    'Layer Model: EfficientNet-B5 Attn U-Net<br>'
    'Classes: 8 lesions + 3 retinal layers<br>'
    'Resolution: 256x256 px | PyTorch + SMP'
    '</div>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════
# SECTION 8 — TOP BAR
# ══════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="visionoct-topbar">
  <div style="display:flex;align-items:center;gap:14px">
    <div class="topbar-icon">👁</div>
    <div>
      <div class="topbar-title">VisionOCT Pro</div>
      <div class="topbar-sub">Alamein International University &mdash; Dual-Model Neural Imaging Suite</div>
    </div>
  </div>
  <div class="topbar-badges">
    <span class="badge-live">&#9679; LIVE</span>
    <span class="badge-model">Dual-Model AI</span>
    <span class="badge-dr">Dr. {dr_input}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 9 — VISIT UPLOAD
# ══════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="section-header">Patient Visit Setup '
    '<span class="section-tag">Multi-Visit Tracking</span></div>',
    unsafe_allow_html=True,
)

n_visits = st.number_input("Number of visits", min_value=1, max_value=10, value=2, step=1)

visits_input = []
for i in range(n_visits):
    with st.expander(f"Visit {i+1}", expanded=(i == 0)):
        visit_date  = st.date_input("Visit date", key=f"date_{i}", value=datetime.date.today())
        visit_files = st.file_uploader(
            "Upload B-Scans (JPG / PNG)",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key=f"visit_{i}",
            help="Upload all B-scan slices from this visit",
        )
        if visit_files:
            visits_input.append({
                "visit_num": i + 1,
                "date":      str(visit_date),
                "files":     visit_files,
            })


# ══════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN ANALYSIS LOOP
# ══════════════════════════════════════════════════════════════════

if visits_input:
    LESION_MODEL = load_lesion_model()
    LAYER_MODEL  = load_layer_model()

    visit_summaries  = []
    all_scan_details = []

    for visit in visits_input:
        scans_this_visit = []
        with st.status(
            f"Analyzing Visit {visit['visit_num']} ({visit['date']})...", expanded=True
        ) as status:
            for f in visit["files"]:
                img = Image.open(f).convert("RGB")
                (
                    lesion_mask_img, layer_mask_img, composite_img,
                    stats, layer_stats,
                    f_idx, sev_score, sev_grade, sev_color,
                    measurements,
                ) = analyze_scan(img, LESION_MODEL, LAYER_MODEL)

                entry = {
                    "Visit":           f"Visit {visit['visit_num']}",
                    "Date":            visit["date"],
                    "Filename":        f.name,
                    "Original":        img,
                    "LesionMask":      lesion_mask_img,
                    "LayerMask":       layer_mask_img,
                    "Composite":       composite_img,
                    "Fluid Index (%)": round(f_idx, 2),
                    "Severity Score":  sev_score,
                    "Severity Grade":  sev_grade,
                    "Severity Color":  sev_color,
                    "Measurements":    measurements,
                    "LayerStats":      layer_stats,
                }
                entry.update(stats)
                scans_this_visit.append(entry)
                all_scan_details.append(entry)
                st.write(
                    f"Processed {f.name} — "
                    f"Severity: **{sev_score}/100** [{sev_grade}] | "
                    f"Fluid: **{round(f_idx,2)}%**"
                )
            status.update(
                label=f"Visit {visit['visit_num']} complete",
                state="complete",
                expanded=False,
            )

        scans_df   = pd.DataFrame(scans_this_visit)
        grade_mode = scans_df["Severity Grade"].mode()
        visit_summaries.append({
            "Visit":               f"Visit {visit['visit_num']}",
            "Date":                visit["date"],
            "Scans":               len(scans_this_visit),
            "Avg Severity Score":  round(scans_df["Severity Score"].mean(), 1),
            "Peak Severity Score": scans_df["Severity Score"].max(),
            "Avg Fluid Index (%)": round(scans_df["Fluid Index (%)"].mean(), 2),
            "Grade":               grade_mode[0] if len(grade_mode) > 0 else "-",
            "Raw Scans":           scans_this_visit,
        })

    summary_df = pd.DataFrame(visit_summaries)
    detail_df  = pd.DataFrame(all_scan_details)

    total_scans = len(all_scan_details)
    peak_sev    = summary_df["Peak Severity Score"].max()
    latest_sev  = summary_df["Avg Severity Score"].iloc[-1]
    peak_fluid  = detail_df["Fluid Index (%)"].max()
    curr_fluid  = detail_df["Fluid Index (%)"].iloc[-1]

    st.markdown(f"""
    <div class="patient-bar">
      <div>
        <div class="patient-mrn">Multi-Visit Dual-Model Analysis</div>
        <div class="patient-name">{len(visits_input)} Visits &middot; {total_scans} B-Scans</div>
      </div>
      <div class="patient-meta">
        <div class="meta-item"><span>Peak Severity</span>{peak_sev}/100</div>
        <div class="meta-item"><span>Latest Avg Severity</span>{latest_sev}/100</div>
        <div class="meta-item"><span>Peak Fluid</span>{peak_fluid}%</div>
        <div class="meta-item"><span>Physician</span>Dr. {dr_input}</div>
      </div>
      <span class="status-complete">Analysis Complete</span>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──
    tab_trends, tab_visits, tab_scans, tab_report = st.tabs([
        "📈  Visit Trends",
        "🗂  Per-Visit Summary",
        "🖼  Scans & Composite",
        "📑  Official Report",
    ])

    # ══════════════════════════
    # TAB 1 — VISIT TRENDS
    # ══════════════════════════
    with tab_trends:
        sev_reduction   = round(((peak_sev - latest_sev) / peak_sev) * 100, 1) if peak_sev > 0 else 0
        fluid_reduction = round(((peak_fluid - curr_fluid) / peak_fluid) * 100, 1) if peak_fluid > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Peak Severity", f"{peak_sev}/100",
                summary_df.loc[summary_df["Peak Severity Score"].idxmax(), "Visit"],
                color=severity_color(peak_sev)), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Latest Avg Severity", f"{latest_sev}/100",
                summary_df["Visit"].iloc[-1],
                color=severity_color(latest_sev)), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Severity Reduction", f"{sev_reduction}%",
                "From peak visit", color="#34D399"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Total Visits", str(len(visits_input)),
                f"{total_scans} B-Scans total", color="#BFDBFE"), unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Severity trend
        st.markdown(
            '<div class="section-header">Severity Score — Visit Progression'
            '<span class="section-tag">Score / 100</span></div>',
            unsafe_allow_html=True,
        )
        fig_sev = go.Figure()
        fig_sev.add_trace(go.Scatter(
            x=summary_df["Date"], y=summary_df["Avg Severity Score"],
            name="Avg Severity", mode="lines+markers",
            line=dict(color="#F87171", width=2.5),
            marker=dict(size=9, color=[severity_color(v) for v in summary_df["Avg Severity Score"]],
                        line=dict(color="#141929", width=1.5)),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.07)",
            hovertemplate="<b>%{x}</b><br>Avg Severity: %{y}/100<extra></extra>",
        ))
        fig_sev.add_trace(go.Scatter(
            x=summary_df["Date"], y=summary_df["Peak Severity Score"],
            name="Peak Severity", mode="lines+markers",
            line=dict(color="#FCD34D", width=1.5, dash="dot"),
            marker=dict(size=7, color="#FCD34D"),
            hovertemplate="<b>%{x}</b><br>Peak: %{y}/100<extra></extra>",
        ))
        fig_sev.update_layout(
            paper_bgcolor="#141929", plot_bgcolor="#141929",
            font=dict(family="DM Mono", color="#CBD5E1", size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1E2840", linecolor="#1E2840",
                       tickfont=dict(size=10, color="#94A3B8")),
            yaxis=dict(gridcolor="#1E2840", linecolor="#1E2840",
                       tickfont=dict(size=10, color="#94A3B8"),
                       range=[0, 110], ticksuffix="/100"),
            legend=dict(bgcolor="#0F1525", bordercolor="#1E2840", borderwidth=1,
                        font=dict(size=10, color="#CBD5E1")),
            height=260,
        )
        st.plotly_chart(fig_sev, use_container_width=True)

        # Fluid trend
        st.markdown(
            '<div class="section-header">Fluid Index — Visit Progression'
            '<span class="section-tag">Avg Fluid %</span></div>',
            unsafe_allow_html=True,
        )
        fig_fluid = go.Figure()
        fig_fluid.add_trace(go.Scatter(
            x=summary_df["Date"], y=summary_df["Avg Fluid Index (%)"],
            mode="lines+markers",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=8,
                        color=[fluid_color(v) for v in summary_df["Avg Fluid Index (%)"]],
                        line=dict(color="#141929", width=1.5)),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
            hovertemplate="<b>%{x}</b><br>Avg Fluid: %{y}%<extra></extra>",
        ))
        fig_fluid.update_layout(
            paper_bgcolor="#141929", plot_bgcolor="#141929",
            font=dict(family="DM Mono", color="#CBD5E1", size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#1E2840", linecolor="#1E2840",
                       tickfont=dict(size=10, color="#94A3B8")),
            yaxis=dict(gridcolor="#1E2840", linecolor="#1E2840",
                       tickfont=dict(size=10, color="#94A3B8"),
                       ticksuffix="%"),
            height=220, showlegend=False,
        )
        st.plotly_chart(fig_fluid, use_container_width=True)

        # Fluid compartment stacked bar
        fluid_cols = [c for c in FLUID_CLASSES if c in detail_df.columns]
        if fluid_cols:
            st.markdown(
                '<div class="section-header">Fluid Compartment — All Scans'
                '<span class="section-tag">px per class</span></div>',
                unsafe_allow_html=True,
            )
            comp_colors = {
                "IRF":              "#7DD3E8",
                "SRF":              "#60A5FA",
                "Drusenoid PED":    "#FB7185",
                "Fibrovascular PED":"#F472B6",
            }
            fig_comp = go.Figure()
            for cls in fluid_cols:
                fig_comp.add_trace(go.Bar(
                    name=cls, x=detail_df["Filename"], y=detail_df[cls],
                    marker_color=comp_colors.get(cls, "#94A3B8"), opacity=0.85,
                    hovertemplate=f"<b>{cls}</b>: %{{y:,}} px<extra></extra>",
                ))
            fig_comp.update_layout(
                paper_bgcolor="#141929", plot_bgcolor="#141929",
                font=dict(family="DM Mono", color="#CBD5E1", size=11),
                margin=dict(l=10, r=10, t=10, b=10), barmode="stack",
                legend=dict(bgcolor="#0F1525", bordercolor="#1E2840", borderwidth=1,
                            font=dict(size=10, color="#CBD5E1")),
                xaxis=dict(gridcolor="#1E2840", linecolor="#1E2840",
                           tickfont=dict(size=10, color="#94A3B8")),
                yaxis=dict(gridcolor="#1E2840", linecolor="#1E2840",
                           tickfont=dict(size=10, color="#94A3B8"),
                           ticksuffix=" px"),
                height=220,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # ══════════════════════════
    # TAB 2 — PER-VISIT SUMMARY
    # ══════════════════════════
    with tab_visits:
        for vs in visit_summaries:
            sev_c = severity_color(vs["Avg Severity Score"])
            fl_c  = fluid_color(vs["Avg Fluid Index (%)"])
            with st.expander(
                f"{vs['Visit']}  |  {vs['Date']}  |  {vs['Scans']} scans  |  "
                f"Avg Severity: {vs['Avg Severity Score']}/100 [{vs['Grade']}]",
                expanded=True,
            ):
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.markdown(metric_card("Avg Severity Score",
                        f"{vs['Avg Severity Score']}/100", vs["Grade"], color=sev_c), unsafe_allow_html=True)
                with mc2:
                    st.markdown(metric_card("Peak Severity",
                        f"{vs['Peak Severity Score']}/100", "Worst scan", color=sev_c), unsafe_allow_html=True)
                with mc3:
                    st.markdown(metric_card("Avg Fluid Index",
                        f"{vs['Avg Fluid Index (%)']}%", "Mean of all scans", color=fl_c), unsafe_allow_html=True)
                with mc4:
                    st.markdown(metric_card("B-Scans",
                        str(vs["Scans"]), "Slices analyzed", color="#BFDBFE"), unsafe_allow_html=True)

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-header">Layer-Aware Lesion Measurements</div>',
                    unsafe_allow_html=True,
                )

                lesion_rows = []
                for lesion_name in SEVERITY_WEIGHTS:
                    total_px   = sum(s.get(lesion_name, 0) for s in vs["Raw Scans"])
                    avg_px     = round(total_px / vs["Scans"], 1)
                    total_mm2  = round(total_px * PIXEL_TO_MM2, 3)
                    avg_pct    = round(total_px / (TOTAL_PX * vs["Scans"]) * 100, 2)

                    # Aggregate dominant layer across scans
                    layer_counts: dict = {}
                    for scan in vs["Raw Scans"]:
                        dm = scan["Measurements"].get(lesion_name, {}).get("dominant_layer", "Unknown")
                        layer_counts[dm] = layer_counts.get(dm, 0) + 1
                    dom_layer = max(layer_counts, key=layer_counts.get) if layer_counts else "Unknown"

                    lesion_rows.append({
                        "Lesion":           lesion_name,
                        "Dominant Layer":   dom_layer,
                        "Sev. Weight":      SEVERITY_WEIGHTS[lesion_name],
                        "Total Pixels":     f"{total_px:,}",
                        "Avg px / scan":    f"{avg_px:,}",
                        "Total Area (mm2)": total_mm2,
                        "Avg Coverage (%)": avg_pct,
                    })

                st.dataframe(
                    pd.DataFrame(lesion_rows),
                    hide_index=True,
                    use_container_width=True,
                )

    # ══════════════════════════
    # TAB 3 — SCANS & COMPOSITE
    # ══════════════════════════
    with tab_scans:
        st.markdown(
            '<div class="composite-info">'
            '<strong style="color:#BFDBFE">Composite Image</strong> &mdash; '
            'Each scan is rendered as a 3-layer overlay: '
            '<span style="color:#FFB432">Choroid</span>, '
            '<span style="color:#50C878">NSR</span>, '
            '<span style="color:#7850DC">RPE</span> layers are shown as translucent fills, '
            'with lesion features (IRF, SRF, PED, HRF, SHRM) rendered on top at higher opacity. '
            'Fluid area measurements are normalised to the layer the lesion occupies.'
            '</div>',
            unsafe_allow_html=True,
        )

        for vs in visit_summaries:
            st.markdown(
                f'<div class="section-header">{vs["Visit"]} &mdash; {vs["Date"]}</div>',
                unsafe_allow_html=True,
            )
            for data in vs["Raw Scans"]:
                f_val  = data["Fluid Index (%)"]
                s_val  = data["Severity Score"]
                fc     = fluid_color(f_val)
                sc     = severity_color(s_val)
                status_label = "High" if f_val > 1.5 else ("Moderate" if f_val > 1 else "Low")

                with st.expander(
                    f"{data['Filename']}   |   Severity: {s_val}/100 [{data['Severity Grade']}]"
                    f"   |   Fluid: {f_val}%   [{status_label}]"
                ):
                    # ── Four-panel image row ──
                    c1, c2, c3, c4 = st.columns([1.8, 1.8, 1.8, 1.4])

                    with c1:
                        st.markdown(
                            '<div class="scan-panel-header">'
                            '<span class="scan-panel-title">Input B-Scan</span></div>',
                            unsafe_allow_html=True,
                        )
                        st.image(data["Original"], use_container_width=True)

                    with c2:
                        st.markdown(
                            '<div class="scan-panel-header">'
                            '<span class="scan-panel-title">Layer Segmentation</span></div>',
                            unsafe_allow_html=True,
                        )
                        st.image(data["LayerMask"], use_container_width=True)

                    with c3:
                        st.markdown(
                            '<div class="scan-panel-header">'
                            '<span class="scan-panel-title">Composite Overlay</span></div>',
                            unsafe_allow_html=True,
                        )
                        st.image(data["Composite"], use_container_width=True)

                    with c4:
                        # Metrics + findings
                        st.markdown(f"""
                        <div style="padding:8px 0">
                          <div class="metric-label">Severity Score</div>
                          <div class="metric-value" style="color:{sc}">{s_val}/100</div>
                          <div class="metric-sub" style="color:{sc}">{data['Severity Grade']}</div>
                          {severity_bar(s_val, sc)}
                          <div style="height:8px"></div>
                          <div class="metric-label">Fluid Index</div>
                          <div class="metric-value" style="color:{fc};font-size:18px">{f_val}%</div>
                          {fluid_bar(f_val, fc)}
                        </div>
                        """, unsafe_allow_html=True)

                        # Layer stats
                        layer_tags = ""
                        for l_name, _ in LAYER_MAP:
                            if l_name == "Background":
                                continue
                            lpx = data["LayerStats"].get(l_name, 0)
                            if lpx > 0:
                                lc = layer_color_hex(l_name)
                                layer_tags += (
                                    f'<span class="finding-tag tag-layer" '
                                    f'style="border-color:{lc}40;color:{lc}">'
                                    f'{l_name}: {lpx:,}px</span>'
                                )

                        # Lesion tags
                        lesion_tags = ""
                        for cls in FLUID_CLASSES:
                            if data.get(cls, 0) > 0:
                                tc = TAG_CLASSES.get(cls, "tag-irf")
                                meas = data["Measurements"].get(cls, {})
                                dl   = meas.get("dominant_layer", "?")
                                lp   = meas.get("layer_pct", 0)
                                lesion_tags += (
                                    f'<span class="finding-tag {tc}">'
                                    f'{cls}: {data[cls]:,}px in {dl} ({lp}%)</span>'
                                )
                        if data.get("HRF", 0) > 0:
                            lesion_tags += f'<span class="finding-tag tag-hrf">HRF: {data["HRF"]:,}px</span>'
                        if data.get("SHRM", 0) > 0:
                            lesion_tags += f'<span class="finding-tag tag-shrm">SHRM: {data["SHRM"]:,}px</span>'

                        st.markdown(
                            f'<div style="margin-top:8px"><div class="metric-label">Layers</div>'
                            f'<div class="findings-row">{layer_tags}</div></div>'
                            f'<div style="margin-top:8px"><div class="metric-label">Lesions</div>'
                            f'<div class="findings-row">{lesion_tags}</div></div>',
                            unsafe_allow_html=True,
                        )

                    # ── Layer-aware measurements table ──
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    st.markdown(
                        '<div class="section-header">Layer-Aware Pixel Measurements</div>',
                        unsafe_allow_html=True,
                    )
                    meas_rows = []
                    for lesion, vals in data["Measurements"].items():
                        if vals["px"] > 0:
                            meas_rows.append({
                                "Lesion":              lesion,
                                "Dominant Layer":      vals["dominant_layer"],
                                "Pixels":              f"{vals['px']:,}",
                                "Area (mm2)":          vals["area_mm2"],
                                "Image Coverage (%)":  vals["image_pct"],
                                "Layer Coverage (%)":  vals["layer_pct"],
                                "Sev. Weight":         SEVERITY_WEIGHTS.get(lesion, "-"),
                            })
                    if meas_rows:
                        st.dataframe(pd.DataFrame(meas_rows), hide_index=True, use_container_width=True)

    # ══════════════════════════
    # TAB 4 — OFFICIAL REPORT
    # ══════════════════════════
    with tab_report:
        st.markdown(
            '<div class="section-header">Patient Information</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            p_name = st.text_input("Patient Full Name", placeholder="e.g. Mohamed Al-Rashidi")
            p_id   = st.text_input("Patient ID / MRN",  placeholder="e.g. MRN-004024")
        with col2:
            p_age    = st.number_input("Age", 1, 120, 45)
            p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.button("Generate AI Clinical Draft", type="primary"):
            if not p_name or not p_id:
                st.warning("Please enter patient name and ID first.")
            else:
                with st.spinner("Synthesizing clinical findings with Llama 3.3..."):
                    visit_lines = [
                        f"{vs['Visit']} ({vs['Date']}): "
                        f"Avg Severity={vs['Avg Severity Score']}/100 [{vs['Grade']}], "
                        f"Avg Fluid={vs['Avg Fluid Index (%)']}%, "
                        f"Peak Severity={vs['Peak Severity Score']}/100"
                        for vs in visit_summaries
                    ]
                    cols = (
                        ["Visit", "Date", "Filename", "Fluid Index (%)", "Severity Score", "Severity Grade"]
                        + [c for c in FLUID_CLASSES if c in detail_df.columns]
                    )
                    scan_table = detail_df[cols].to_string(index=False)

                    # Include dominant layer context
                    layer_context = ""
                    for scan in all_scan_details[:3]:
                        for lesion, vals in scan["Measurements"].items():
                            if vals["px"] > 50:
                                layer_context += (
                                    f"  {lesion}: {vals['px']}px in {vals['dominant_layer']} "
                                    f"({vals['layer_pct']}% of that layer)\n"
                                )

                    prompt = (
                        f"Write a detailed one-page clinical OCT report for a retinal specialist.\n"
                        f"Patient: {p_name}, Age: {p_age}, Gender: {p_gender}\n\n"
                        f"VISIT SUMMARY:\n" + "\n".join(visit_lines) + "\n\n"
                        f"DETAILED SCAN DATA:\n{scan_table}\n\n"
                        f"LAYER-AWARE LESION CONTEXT (selected scans):\n{layer_context}\n\n"
                        f"Include: clinical summary, key findings per visit, "
                        f"retinal layer involvement (Choroid/NSR/RPE), overall impression, "
                        f"treatment recommendations, and disease activity trend."
                    )
                    st.session_state["report_text"] = get_groq_ai_response(prompt)

        if "report_text" in st.session_state:
            st.markdown('<hr style="border-color:#1E2840;margin:14px 0">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-header">Review & AI Refinement</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"""
            <div class="report-card">
              <div class="ai-badge"><div class="ai-dot"></div> AI Clinical Draft &mdash; Llama 3.3 70B</div>
              <div class="report-body">{st.session_state['report_text']}</div>
            </div>
            """, unsafe_allow_html=True)

            final_report = st.text_area(
                "Edit Clinical Summary",
                value=st.session_state["report_text"],
                height=280,
            )
            st.session_state["report_text"] = final_report

            user_instruction = st.chat_input(
                "Ask AI to refine report  (e.g. 'Make it more formal' / 'Add ICD-10 codes')"
            )
            if user_instruction:
                with st.spinner("AI refining report..."):
                    refine_prompt = (
                        f"Original:\n{st.session_state['report_text']}\n\n"
                        f"Instruction: {user_instruction}"
                    )
                    st.session_state["report_text"] = get_groq_ai_response(refine_prompt)
                    st.rerun()

            st.markdown('<hr style="border-color:#1E2840;margin:14px 0">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-header">Finalize Official Document</div>',
                unsafe_allow_html=True,
            )

            col_pdf, col_prnt = st.columns(2)
            p_info = {"name": p_name, "age": p_age, "gender": p_gender, "id": p_id}

            with col_pdf:
                clean_name    = (p_name or "patient").replace(" ", "_")
                clean_id      = (p_id or "id").replace(" ", "_")
                download_name = f"{clean_name}_{clean_id}.pdf"
                pdf_data      = create_medical_pdf(
                    p_info, dr_input,
                    st.session_state["report_text"],
                    visit_summary=visit_summaries,
                )
                st.download_button(
                    label="Download Signed PDF",
                    data=io.BytesIO(pdf_data),
                    file_name=download_name,
                    mime="application/pdf",
                )

            with col_prnt:
                if st.button("Print Preview"):
                    import re
                    rh = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>",
                                st.session_state["report_text"])
                    rh = re.sub(r"\*(.*?)\*", r"<em>\1</em>", rh)
                    rh = rh.replace("\n\n", '</p><p style="margin:6px 0;font-size:13px;color:#1a1a1a;line-height:1.7;">')
                    rh = rh.replace("\n", "<br>")

                    visit_rows_html = ""
                    for vs in visit_summaries:
                        sc = {
                            "SEVERE": "#dc2626", "MODERATE": "#d97706",
                            "MILD":   "#16a34a", "MINIMAL":  "#64748b",
                        }.get(vs["Grade"], "#64748b")
                        visit_rows_html += f"""
                        <tr>
                          <td>{vs['Visit']}</td>
                          <td>{vs['Date']}</td>
                          <td>{vs['Scans']}</td>
                          <td style="color:{sc};font-weight:700">{vs['Avg Severity Score']}/100</td>
                          <td style="color:{sc}">{vs['Grade']}</td>
                          <td>{vs['Avg Fluid Index (%)']}%</td>
                        </tr>"""

                    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#f0f4f8;font-family:Arial,sans-serif;padding:20px}}
  .page{{background:#fff;max-width:780px;margin:0 auto;border:1px solid #c8d6e5;border-radius:4px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.08)}}
  .hospital-header{{background:#0a2744;color:white;padding:18px 30px;display:flex;align-items:center;justify-content:space-between}}
  .hospital-name{{font-size:15px;font-weight:700;color:#fff}}
  .hospital-sub{{font-size:11px;color:#93c5fd;margin-top:2px}}
  .report-badge{{background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);padding:5px 14px;border-radius:4px;font-size:11px;color:#bfdbfe;text-transform:uppercase}}
  .title-bar{{background:#f8fafc;border-bottom:2px solid #e2e8f0;padding:14px 30px;text-align:center}}
  .report-title{{font-size:17px;font-weight:700;color:#0a2744}}
  .report-subtitle{{font-size:11px;color:#64748b;margin-top:3px}}
  .info-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;border-bottom:2px solid #e2e8f0}}
  .info-cell{{padding:10px 20px;border-right:1px solid #e2e8f0}}
  .info-cell:last-child{{border-right:none}}
  .info-label{{font-size:9px;font-weight:700;color:#94a3b8;letter-spacing:1px;text-transform:uppercase;margin-bottom:3px}}
  .info-value{{font-size:13px;font-weight:600;color:#0f172a}}
  .section-bar{{background:#0a2744;color:white;padding:7px 30px;font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase}}
  .visit-table{{width:100%;border-collapse:collapse;font-size:12px}}
  .visit-table th{{background:#f1f5f9;color:#475569;font-size:10px;letter-spacing:0.8px;text-transform:uppercase;padding:8px 12px;text-align:left;border-bottom:1px solid #e2e8f0}}
  .visit-table td{{padding:7px 12px;border-bottom:1px solid #f1f5f9;color:#1e293b}}
  .findings-body{{padding:12px 30px}}
  .findings-body p{{font-size:13px;color:#1a1a1a;line-height:1.7;margin:6px 0}}
  .stamp-row{{display:flex;justify-content:space-between;align-items:flex-end;padding:16px 30px;border-top:2px solid #e2e8f0;background:#f8fafc}}
  .stamp-left{{font-size:11px;color:#64748b;line-height:1.7}}
  .sig-line{{width:180px;border-top:1px solid #334155;margin-bottom:4px;margin-left:auto}}
  .sig-name{{font-size:13px;font-weight:700;color:#0a2744}}
  .sig-title{{font-size:10px;color:#64748b}}
  .footer{{background:#0a2744;color:#93c5fd;font-size:9px;text-align:center;padding:6px;letter-spacing:0.5px}}
  @media print{{body{{padding:0;background:#fff}}.page{{box-shadow:none!important;border:none!important;max-width:100%}}}}
</style>
</head>
<body>
<div class="page">
  <div class="hospital-header">
    <div>
      <div class="hospital-name">ALAMEIN INTERNATIONAL UNIVERSITY &mdash; AIU</div>
      <div class="hospital-sub">Center for Precision Ophthalmic Intelligence &nbsp;&middot;&nbsp; VisionOCT Pro Suite</div>
    </div>
    <div class="report-badge">Ophthalmology Report</div>
  </div>
  <div class="title-bar">
    <div class="report-title">OCT RETINAL ANALYSIS &mdash; MULTI-VISIT CLINICAL REPORT</div>
    <div class="report-subtitle">AI-Assisted Dual-Model Diagnostic Assessment | Confidential Medical Document</div>
  </div>
  <div class="info-grid">
    <div class="info-cell"><div class="info-label">Patient Name</div><div class="info-value">{p_name or '-'}</div></div>
    <div class="info-cell"><div class="info-label">MRN / ID</div><div class="info-value">{p_id or '-'}</div></div>
    <div class="info-cell"><div class="info-label">Report Date</div><div class="info-value">{time.strftime("%d %b %Y")}</div></div>
    <div class="info-cell"><div class="info-label">Age</div><div class="info-value">{p_age} yrs</div></div>
    <div class="info-cell"><div class="info-label">Gender</div><div class="info-value">{p_gender}</div></div>
    <div class="info-cell"><div class="info-label">Modality</div><div class="info-value">OCT B-Scan (Dual-Model)</div></div>
  </div>
  <div class="section-bar">Visit Summary</div>
  <table class="visit-table">
    <tr>
      <th>Visit</th><th>Date</th><th>Scans</th>
      <th>Avg Severity</th><th>Grade</th><th>Avg Fluid</th>
    </tr>
    {visit_rows_html}
  </table>
  <div class="section-bar">Clinical Findings &amp; AI Analysis</div>
  <div class="findings-body">
    <p style="margin:6px 0;font-size:13px;color:#1a1a1a;line-height:1.7;">{rh}</p>
  </div>
  <div class="stamp-row">
    <div class="stamp-left">
      <strong style="color:#0a2744">Referring Physician:</strong> Dr. {dr_input}<br>
      Consultant Specialist | AIU Clinical Diagnostic Suite<br>
      Report generated: {time.strftime("%d %B %Y, %H:%M")}
    </div>
    <div>
      <div class="sig-line"></div>
      <div class="sig-name">Dr. {dr_input}</div>
      <div class="sig-title">Digitally Verified Signature</div>
    </div>
  </div>
  <div class="footer">
    ALAMEIN INTERNATIONAL UNIVERSITY &nbsp;&middot;&nbsp; VisionOCT PRO SUITE &nbsp;&middot;&nbsp;
    DEVELOPED BY ABDO LASHEEN &nbsp;&middot;&nbsp; 2026 &nbsp;&middot;&nbsp; CONFIDENTIAL
  </div>
</div>
<script>setTimeout(function(){{window.print();}},400);</script>
</body>
</html>"""
                    st.components.v1.html(html_content, height=900, scrolling=True)

# ══════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div style="background:#141929;border:1px dashed #263050;border-radius:16px;
                padding:56px 32px;text-align:center;margin-top:24px;
                animation:fadeSlideUp 0.7s ease both;">
      <div style="font-size:48px;margin-bottom:18px;animation:pulse 3s infinite">👁</div>
      <div style="font-size:18px;font-weight:700;color:#F1F5F9;margin-bottom:10px">
        VisionOCT Dual-Model Diagnostic Suite
      </div>
      <div style="font-size:13px;color:#64748B;line-height:1.8;max-width:520px;margin:0 auto">
        Set the number of visits above, then upload B-scan images for each visit.<br>
        The dual AI models will segment <strong style="color:#CBD5E1">retinal layers</strong>
        and <strong style="color:#CBD5E1">lesion features</strong> simultaneously,
        producing a composite overlay with layer-aware fluid measurements.
      </div>
      <div style="margin-top:24px;display:flex;gap:10px;justify-content:center;flex-wrap:wrap">
        <span style="padding:5px 14px;border-radius:20px;background:rgba(37,99,235,0.12);color:#BFDBFE;
              border:1px solid rgba(37,99,235,0.25);font-size:11px;font-family:'DM Mono',monospace">
          IRF / SRF / PED Detection</span>
        <span style="padding:5px 14px;border-radius:20px;background:rgba(255,180,50,0.12);color:#FFB432;
              border:1px solid rgba(255,180,50,0.25);font-size:11px;font-family:'DM Mono',monospace">
          Choroid / NSR / RPE Layers</span>
        <span style="padding:5px 14px;border-radius:20px;background:rgba(80,200,120,0.12);color:#50C878;
              border:1px solid rgba(80,200,120,0.25);font-size:11px;font-family:'DM Mono',monospace">
          Layer-Aware Fluid Measurement</span>
        <span style="padding:5px 14px;border-radius:20px;background:rgba(240,120,240,0.12);color:#E879F9;
              border:1px solid rgba(240,120,240,0.25);font-size:11px;font-family:'DM Mono',monospace">
          Composite Overlay Rendering</span>
        <span style="padding:5px 14px;border-radius:20px;background:rgba(248,113,113,0.12);color:#F87171;
              border:1px solid rgba(248,113,113,0.25);font-size:11px;font-family:'DM Mono',monospace">
          Severity Score + AI Report</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:48px;padding-top:16px;border-top:1px solid #1E2840;text-align:center">
  <span style="font-size:11px;color:#475569;font-family:'DM Mono',monospace">
    VisionOCT Pro Suite &nbsp;&middot;&nbsp; Alamein International University
    &nbsp;&middot;&nbsp; Developed by Abdo Lasheen &nbsp;&middot;&nbsp; 2026
  </span>
</div>
""", unsafe_allow_html=True)
