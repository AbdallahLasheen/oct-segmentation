import streamlit as st
import numpy as np
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from fpdf import FPDF
import time
import io
import datetime

# ── SECTION 1: SYSTEM CONFIGURATION ──
st.set_page_config(
    page_title="VisionOCT Pro | Alamein International University",
    layout="wide",
    page_icon="👁",
    initial_sidebar_state="expanded"
)

# ── DARK MEDICAL UI STYLING ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }

.stApp { background-color: #0A0E1A !important; }
.main .block-container {
    background-color: #0A0E1A !important;
    padding: 1.2rem 1.5rem !important;
    max-width: 100% !important;
}

[data-testid="stSidebar"] {
    background-color: #0F1525 !important;
    border-right: 1px solid #1E2840 !important;
}
[data-testid="stSidebar"] * { color: #94A3B8 !important; font-family: 'DM Sans', sans-serif !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #E2E8F0 !important; font-size: 13px !important; font-weight: 600 !important;
    letter-spacing: 0.5px !important; text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #141929 !important; border: 1px solid #1E2840 !important;
    color: #E2E8F0 !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 13px !important;
}
[data-testid="stSidebar"] hr { border-color: #1E2840 !important; }

.visionoct-topbar {
    display: flex; align-items: center; justify-content: space-between;
    background: #0F1525; border: 1px solid #1E2840; border-radius: 12px;
    padding: 12px 20px; margin-bottom: 16px;
}
.topbar-logo { display: flex; align-items: center; gap: 12px; }
.topbar-icon {
    width: 48px; height: 48px; background: #2563EB; border-radius: 12px;
    display: flex; align-items: center; justify-content: center; font-size: 35px;
}
.topbar-title { font-size: 30px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.5px; margin: 0; }
.topbar-sub { font-size: 18px; color: #94A3B8; margin: 2px 0 0 0; letter-spacing: 0.1px; }
.topbar-badges { display: flex; gap: 8px; align-items: center; }
.badge-live {
    padding: 3px 10px; border-radius: 20px;
    background: rgba(13,148,136,0.15); color: #34D399;
    border: 1px solid rgba(13,148,136,0.3); font-size: 11px; font-weight: 500;
    font-family: 'DM Mono', monospace;
}
.badge-model {
    padding: 3px 10px; border-radius: 20px;
    background: rgba(37,99,235,0.15); color: #93C5FD;
    border: 1px solid rgba(37,99,235,0.3); font-size: 11px; font-weight: 500;
    font-family: 'DM Mono', monospace;
}
.badge-dr {
    padding: 4px 12px; border-radius: 20px;
    background: #141929; color: #94A3B8; border: 1px solid #263050; font-size: 11px;
}

.metric-card {
    background: #141929; border: 1px solid #1E2840;
    border-radius: 10px; padding: 14px 16px;
}
.metric-label {
    font-size: 10px; font-weight: 500; color: #475569;
    letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 6px;
}
.metric-value {
    font-size: 24px; font-weight: 600; font-family: 'DM Mono', monospace;
    letter-spacing: -0.5px; margin: 0;
}
.metric-sub { font-size: 11px; color: #475569; margin-top: 3px; }
.metric-value-sm {
    font-size: 16px; font-weight: 600; font-family: 'DM Mono', monospace;
    color: #E2E8F0; margin: 0;
}

.fluid-bar-wrap { margin: 8px 0; }
.fluid-bar-label { display: flex; justify-content: space-between; margin-bottom: 5px; }
.fluid-bar-text { font-size: 11px; color: #475569; }
.fluid-bar-bg { height: 6px; background: #1E2840; border-radius: 3px; overflow: hidden; }
.fluid-bar-fill {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, #0D9488, #2563EB);
}

.findings-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.finding-tag {
    padding: 3px 9px; border-radius: 5px; font-size: 11px; font-weight: 500;
    font-family: 'DM Mono', monospace; letter-spacing: 0.3px;
}
.tag-irf { background: rgba(170,223,235,0.1); color: #7DD3E8; border: 1px solid rgba(170,223,235,0.2); }
.tag-srf { background: rgba(42,125,209,0.1); color: #60A5FA; border: 1px solid rgba(42,125,209,0.2); }
.tag-ped { background: rgba(255,53,94,0.1); color: #FB7185; border: 1px solid rgba(255,53,94,0.2); }
.tag-hrf { background: rgba(240,120,240,0.1); color: #E879F9; border: 1px solid rgba(240,120,240,0.2); }
.tag-shrm { background: rgba(204,153,51,0.1); color: #FCD34D; border: 1px solid rgba(204,153,51,0.2); }

.section-header {
    font-size: 12px; font-weight: 600; color: #E2E8F0; margin-bottom: 10px;
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 8px; border-bottom: 1px solid #1E2840;
}
.section-tag {
    font-size: 10px; font-family: 'DM Mono', monospace; color: #475569;
    padding: 2px 7px; background: #1E2840; border-radius: 4px;
}

.scan-panel { background: #141929; border: 1px solid #1E2840; border-radius: 10px; overflow: hidden; }
.scan-panel-header {
    padding: 8px 14px; border-bottom: 1px solid #1E2840;
    display: flex; align-items: center; justify-content: space-between;
}
.scan-panel-title { font-size: 11px; font-weight: 500; color: #94A3B8; }

.patient-bar {
    background: #141929; border: 1px solid #1E2840; border-radius: 10px;
    padding: 10px 16px; display: flex; align-items: center;
    gap: 16px; margin-bottom: 14px; flex-wrap: wrap;
}
.patient-name { font-size: 14px; font-weight: 600; color: #E2E8F0; }
.patient-mrn { font-size: 11px; color: #475569; font-family: 'DM Mono', monospace; }
.patient-meta { display: flex; gap: 14px; margin-left: auto; flex-wrap: wrap; }
.meta-item { font-size: 11px; color: #94A3B8; }
.meta-item span { color: #475569; margin-right: 4px; }
.status-complete {
    padding: 3px 10px; border-radius: 20px;
    background: rgba(13,148,136,0.15); color: #34D399;
    border: 1px solid rgba(13,148,136,0.3); font-size: 10px; font-weight: 500;
    font-family: 'DM Mono', monospace;
}
.status-pending {
    padding: 3px 10px; border-radius: 20px;
    background: rgba(217,119,6,0.15); color: #FCD34D;
    border: 1px solid rgba(217,119,6,0.3); font-size: 10px; font-weight: 500;
    font-family: 'DM Mono', monospace;
}

.report-card {
    background: #141929; border: 1px solid #1E2840;
    border-radius: 10px; padding: 16px; margin-bottom: 14px;
}
.ai-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; color: #93C5FD; font-weight: 500; margin-bottom: 12px;
}
.ai-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #2563EB; display: inline-block; animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
.report-body {
    font-size: 12px; line-height: 1.75; color: #94A3B8;
    border-left: 2px solid #2563EB; padding-left: 14px;
    font-family: 'DM Mono', monospace; white-space: pre-wrap;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0F1525 !important; border-bottom: 1px solid #1E2840 !important;
    gap: 0 !important; padding: 0 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important; border-radius: 0 !important;
    color: #475569 !important; font-size: 12px !important; font-weight: 500 !important;
    padding: 10px 16px !important; font-family: 'DM Sans', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    color: #93C5FD !important; border-bottom: 2px solid #2563EB !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { background: #0A0E1A !important; padding: 16px 0 !important; }

.js-plotly-plot .plotly .bg { fill: #141929 !important; }

[data-testid="stFileUploader"] {
    background: #141929 !important; border: 1px dashed #263050 !important; border-radius: 10px !important;
}
[data-testid="stFileUploader"] * { color: #94A3B8 !important; }
[data-testid="stFileUploaderFile"] { color: #E2E8F0 !important; }
[data-testid="stFileUploaderFileName"] { color: #E2E8F0 !important; }
[data-testid="stFileUploaderFileData"] { color: #94A3B8 !important; }

.stButton > button {
    background: #2563EB !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important; padding: 8px 18px !important;
}
.stButton > button:hover { background: #1D4ED8 !important; }
.stDownloadButton > button {
    background: #1A2035 !important; color: #93C5FD !important;
    border: 1px solid #263050 !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 13px !important;
}

.stTextInput input {
    background: #141929 !important; border: 1px solid #263050 !important;
    color: #E2E8F0 !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 13px !important;
    caret-color: #E2E8F0 !important;
}
.stTextInput input::placeholder { color: #475569 !important; }
.stTextInput input:focus { border-color: #2563EB !important; outline: none !important; }

.stNumberInput input {
    background: #141929 !important; border: 1px solid #263050 !important;
    color: #E2E8F0 !important; border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important; font-size: 13px !important;
    caret-color: #E2E8F0 !important;
}
.stNumberInput button {
    background: #1E2840 !important; color: #94A3B8 !important; border: 1px solid #263050 !important;
}
.stNumberInput button:hover { background: #263050 !important; color: #E2E8F0 !important; }

.stSelectbox > div > div {
    background: #141929 !important; border: 1px solid #263050 !important;
    border-radius: 8px !important; color: #E2E8F0 !important;
}
.stSelectbox > div > div > div { color: #E2E8F0 !important; }
.stSelectbox svg { fill: #475569 !important; }
[data-baseweb="select"] * { background: #141929 !important; color: #E2E8F0 !important; }
[data-baseweb="popover"] { background: #1A2035 !important; border: 1px solid #263050 !important; }
[data-baseweb="menu"] { background: #1A2035 !important; }
[data-baseweb="option"]:hover { background: #263050 !important; }

.stTextArea textarea {
    background: #141929 !important; border: 1px solid #263050 !important;
    color: #E2E8F0 !important; border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important; font-size: 12px !important;
    line-height: 1.7 !important; caret-color: #E2E8F0 !important;
}
.stTextArea textarea::placeholder { color: #475569 !important; }

label, .stTextInput label, .stNumberInput label,
.stSelectbox label, .stTextArea label,
[data-testid="stWidgetLabel"] {
    color: #94A3B8 !important; font-size: 12px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 400 !important;
}

.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background: #141929 !important; border: 1px solid #1E2840 !important;
    border-radius: 8px !important; color: #94A3B8 !important; font-size: 12px !important;
}
[data-testid="stExpander"] summary:hover { background: #1A2035 !important; }
.streamlit-expanderContent, [data-testid="stExpander"] > div:last-child {
    background: #0F1525 !important; border: 1px solid #1E2840 !important;
    border-top: none !important; border-radius: 0 0 8px 8px !important;
}

.stAlert { background: #141929 !important; border: 1px solid #1E2840 !important; border-radius: 8px !important; }
.stAlert * { color: #94A3B8 !important; }
[data-testid="stNotification"] { background: #141929 !important; }
[data-testid="stStatusWidget"] {
    background: #141929 !important; border: 1px solid #1E2840 !important; border-radius: 8px !important;
}
[data-testid="stStatusWidget"] * { color: #94A3B8 !important; }
[data-testid="stSpinner"] * { color: #94A3B8 !important; }

[data-testid="stChatInput"] {
    background: #141929 !important; border: 1px solid #1E2840 !important; border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea {
    color: #E2E8F0 !important; background: #141929 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 13px !important;
    caret-color: #E2E8F0 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #475569 !important; }

p, span, div, li, small { color: #94A3B8; }
h1, h2, h3, h4, h5, h6 { color: #E2E8F0 !important; }
strong, b { color: #E2E8F0 !important; }
code { color: #93C5FD !important; background: #1A2035 !important; }
[data-testid="stMarkdownContainer"] p { color: #94A3B8 !important; }
[data-testid="stMarkdownContainer"] strong { color: #E2E8F0 !important; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color: #E2E8F0 !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1E2840; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #263050; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── SECTION 2: CLINICAL CONSTANTS ──
LABEL_MAP = [
    ("Background",          (0,   0,   0  )),
    ("Drosenoid PED",       (248, 231, 180)),
    ("Fibrovascular PED",   (255, 53,  94 )),
    ("HRF",                 (240, 120, 240)),
    ("IRF",                 (170, 223, 235)),
    ("PH",                  (51,  221, 255)),
    ("SHRM",                (204, 153, 51 )),
    ("SRF",                 (42,  125, 209)),
]
FLUID_CLASSES = ["IRF", "SRF", "Drosenoid PED", "Fibrovascular PED"]
CLASS_COLORS  = np.array([c for _, c in LABEL_MAP], dtype=np.uint8)

TAG_CLASSES = {
    "IRF":              "tag-irf",
    "SRF":              "tag-srf",
    "Fibrovascular PED":"tag-ped",
    "Drosenoid PED":    "tag-ped",
    "HRF":              "tag-hrf",
    "SHRM":             "tag-shrm",
}

SEVERITY_WEIGHTS = {
    "IRF":              1.5,
    "SRF":              1.2,
    "Fibrovascular PED":2.0,
    "Drosenoid PED":    1.0,
    "HRF":              0.8,
    "SHRM":             1.3,
    "PH":               0.5,
}

PIXEL_TO_MM2 = (6.0 / 256.0) ** 2   # ≈ 0.00549 mm² per pixel (Zeiss Cirrus 6mm scan)
TOTAL_PX     = 256 * 256             # 65,536

# ── SECTION 3: CORE AI ENGINE ──
@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=len(LABEL_MAP)
    )
    try:
        with open("unet_oct_best_v2.pth", "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["weights"])
    except Exception as e:
        st.error(f"⚠ Model load failed: {e}")
    model.eval()
    return model

def analyze_scan(img, model):
    orig_size = img.size
    resized   = img.resize((256, 256), Image.BILINEAR)
    tensor    = TF.normalize(
        TF.to_tensor(resized),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ).unsqueeze(0)
    with torch.no_grad():
        mask_idx = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
    stats    = {name: int((mask_idx == i).sum()) for i, (name, _) in enumerate(LABEL_MAP)}
    fluid_px = sum(stats[cls] for cls in FLUID_CLASSES)
    fluid_idx = (fluid_px / TOTAL_PX) * 100
    mask_rgb  = Image.fromarray(
        CLASS_COLORS[mask_idx].astype(np.uint8)
    ).resize(orig_size, Image.NEAREST)

    # ── Severity Score ──
    weighted_sum = sum(
        stats.get(cls, 0) * w for cls, w in SEVERITY_WEIGHTS.items()
    )
    max_possible = TOTAL_PX * 2.0   # worst case: all pixels = FibroVasc PED (w=2.0)
    sev_score    = round(min((weighted_sum / max_possible) * 100, 100), 1)

    if sev_score > 60:
        sev_grade, sev_color = "SEVERE",   "#F87171"
    elif sev_score > 30:
        sev_grade, sev_color = "MODERATE", "#FCD34D"
    elif sev_score > 5:
        sev_grade, sev_color = "MILD",     "#34D399"
    else:
        sev_grade, sev_color = "MINIMAL",  "#94A3B8"

    # ── Per-lesion area measurements ──
    measurements = {}
    for name, _ in LABEL_MAP:
        if name == "Background":
            continue
        px = stats.get(name, 0)
        measurements[name] = {
            "px":       px,
            "area_mm2": round(px * PIXEL_TO_MM2, 3),
            "pct":      round(px / TOTAL_PX * 100, 2),
        }

    return mask_rgb, stats, fluid_idx, sev_score, sev_grade, sev_color, measurements

# ── SECTION 4: AI & PDF ──
def get_groq_ai_response(prompt):
    client = OpenAI(
        api_key=st.secrets["GROK_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=[
            {"role": "system", "content": (
                "You are a senior retinal specialist assistant. Write a detailed but concise clinical report "
                "that fits in ONE page. Do NOT include Patient ID, Date, or Modality fields — these are already "
                "in the header. Start directly with the clinical summary. Use clear sections: Clinical Summary, "
                "Key Findings, Impression, and Recommendations."
            )},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

def create_medical_pdf(p_info, dr_name, report_text, visit_summary=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    try:
        pdf.image("uni_logo.png", x=10, y=8, w=25)
    except:
        pass
    pdf.set_font("Arial", 'B', 14); pdf.set_x(40)
    pdf.cell(0, 10, "ALAMEIN INTERNATIONAL UNIVERSITY - AIU", ln=True)
    pdf.set_font("Arial", size=9); pdf.set_x(40)
    pdf.cell(0, 5, "Center for Precision Ophthalmic Intelligence", ln=True)
    pdf.ln(10)
    pdf.set_fill_color(245, 247, 250)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 7, " PATIENT INFORMATION", ln=True, fill=True)
    pdf.set_font("Arial", size=9)
    pdf.cell(95, 6, f" Name: {p_info['name']}", border=1)
    pdf.cell(95, 6, f" ID: {p_info['id']}",     border=1, ln=True)
    pdf.cell(95, 6, f" Age: {p_info['age']}",    border=1)
    pdf.cell(95, 6, f" Gender: {p_info['gender']}", border=1, ln=True)
    if visit_summary:
        pdf.ln(4)
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 7, " VISIT SUMMARY", ln=True, fill=True)
        pdf.set_font("Arial", size=9)
        for vs in visit_summary:
            pdf.cell(0, 6,
                f"  {vs['Visit']} ({vs['Date']}) — Avg Severity: {vs['Avg Severity Score']}/100 "
                f"[{vs['Grade']}]  |  Avg Fluid: {vs['Avg Fluid Index (%)']}%",
                ln=True
            )
    pdf.ln(8)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 7, " CLINICAL FINDINGS & AI ANALYSIS", ln=True)
    pdf.set_font("Arial", size=9)
    safe_report = report_text[:2800].replace("**", "")
    pdf.multi_cell(0, 5, safe_report)
    pdf.set_y(250); pdf.line(10, 250, 200, 250); pdf.ln(5)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 5, "Digitally Verified by:", ln=True, align='R')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 5, f"Dr. {dr_name}", ln=True, align='R')
    pdf.set_font("Arial", size=7)
    pdf.cell(0, 4, "Consultant Specialist | AIU Clinical Diagnostic Suite", ln=True, align='R')
    return bytes(pdf.output())

# ── HELPERS ──
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
    size = "metric-value-sm" if small else "metric-value"
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="{size}" style="color:{color}">{value}</div>
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
      <div class="fluid-bar-bg">
        <div class="fluid-bar-fill" style="width:{width}%"></div>
      </div>
    </div>"""

def severity_bar(score, color):
    return f"""
    <div class="fluid-bar-wrap">
      <div class="fluid-bar-label">
        <span class="fluid-bar-text">Severity Score</span>
        <span style="font-size:11px;font-family:'DM Mono',monospace;color:{color}">{score}/100</span>
      </div>
      <div class="fluid-bar-bg">
        <div class="fluid-bar-fill" style="width:{score}%;background:{color}"></div>
      </div>
    </div>"""

# ── SECTION 5: SIDEBAR ──
st.sidebar.markdown("**CLINICIAN SETUP**")
dr_input = st.sidebar.text_input("Physician Name", value="Ahmed Younis")

st.sidebar.markdown("---")
st.sidebar.markdown("**OCT DEVICE CALIBRATION**")
device_choice = st.sidebar.selectbox(
    "Device",
    ["Zeiss Cirrus (6mm)", "Heidelberg Spectralis (6mm)", "Topcon DRI (7mm)"]
)
device_px_map = {
    "Zeiss Cirrus (6mm)":        (6.0 / 256.0) ** 2,
    "Heidelberg Spectralis (6mm)":(6.0 / 320.0) ** 2,
    "Topcon DRI (7mm)":          (7.0 / 512.0) ** 2,
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
        f'<span style="font-size:12px;color:#94A3B8">{name}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:11px;color:#475569;line-height:1.9">'
    'Model: EfficientNet-B0 U-Net<br>'
    'Classes: 8 retinal lesions<br>'
    'Resolution: 256×256 px<br>'
    'Framework: PyTorch + SMP'
    '</div>',
    unsafe_allow_html=True
)

# ── TOPBAR ──
st.markdown(f"""
<div class="visionoct-topbar">
  <div class="topbar-logo">
    <div class="topbar-icon">👁</div>
    <div>
      <div class="topbar-title">VisionOCT Pro</div>
      <div class="topbar-sub">Alamein International University — Neural Imaging & AI Copilot</div>
    </div>
  </div>
  <div class="topbar-badges">
    <span class="badge-live">● LIVE</span>
    <span class="badge-model">EfficientNet-B0 U-Net</span>
    <span class="badge-dr">Dr. {dr_input}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── VISIT UPLOAD SECTION ──
st.markdown('<div class="section-header">Patient Visit Setup <span class="section-tag">Multi-Visit Tracking</span></div>', unsafe_allow_html=True)

n_visits = st.number_input("Number of visits", min_value=1, max_value=10, value=2, step=1)

visits_input = []
for i in range(n_visits):
    with st.expander(f"Visit {i+1}", expanded=(i == 0)):
        visit_date  = st.date_input(f"Visit date", key=f"date_{i}", value=datetime.date.today())
        visit_files = st.file_uploader(
            "Upload B-Scans (JPG / PNG)",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key=f"visit_{i}",
            help="Upload all B-scan slices from this visit"
        )
        if visit_files:
            visits_input.append({
                "visit_num": i + 1,
                "date":      str(visit_date),
                "files":     visit_files,
            })

# ── MAIN LOGIC ──
if visits_input:
    MODEL = load_model()
    visit_summaries  = []   # one row per visit (aggregated)
    all_scan_details = []   # one row per scan (flat, for cross-visit charts)

    for visit in visits_input:
        scans_this_visit = []
        with st.status(f"⚡ Analyzing Visit {visit['visit_num']} ({visit['date']})...", expanded=True) as status:
            for f in visit["files"]:
                img = Image.open(f).convert("RGB")
                mask, stats, f_idx, sev_score, sev_grade, sev_color, meas = analyze_scan(img, MODEL)
                entry = {
                    "Visit":           f"Visit {visit['visit_num']}",
                    "Date":            visit["date"],
                    "Filename":        f.name,
                    "Original":        img,
                    "Mask":            mask,
                    "Fluid Index (%)": round(f_idx, 2),
                    "Severity Score":  sev_score,
                    "Severity Grade":  sev_grade,
                    "Severity Color":  sev_color,
                    "Measurements":    meas,
                }
                entry.update(stats)
                scans_this_visit.append(entry)
                all_scan_details.append(entry)
                st.write(f"✓ {f.name} — Severity: **{sev_score}/100** [{sev_grade}]  |  Fluid: **{round(f_idx,2)}%**")
            status.update(label=f"✅ Visit {visit['visit_num']} Complete", state="complete", expanded=False)

        scans_df = pd.DataFrame(scans_this_visit)
        grade_mode = scans_df["Severity Grade"].mode()
        visit_summaries.append({
            "Visit":                  f"Visit {visit['visit_num']}",
            "Date":                   visit["date"],
            "Scans":                  len(scans_this_visit),
            "Avg Severity Score":     round(scans_df["Severity Score"].mean(), 1),
            "Peak Severity Score":    scans_df["Severity Score"].max(),
            "Avg Fluid Index (%)":    round(scans_df["Fluid Index (%)"].mean(), 2),
            "Grade":                  grade_mode[0] if len(grade_mode) > 0 else "—",
            "Raw Scans":              scans_this_visit,
        })

    summary_df = pd.DataFrame(visit_summaries)
    detail_df  = pd.DataFrame(all_scan_details)

    # ── SESSION HEADER ──
    total_scans = len(all_scan_details)
    peak_sev    = summary_df["Peak Severity Score"].max()
    latest_sev  = summary_df["Avg Severity Score"].iloc[-1]
    peak_fluid  = detail_df["Fluid Index (%)"].max()
    curr_fluid  = detail_df["Fluid Index (%)"].iloc[-1]

    st.markdown(f"""
    <div class="patient-bar">
      <div>
        <div class="patient-mrn">Multi-Visit Analysis</div>
        <div class="patient-name">{len(visits_input)} Visits · {total_scans} B-Scans</div>
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
        "📈  Visit Trends", "🗂  Per-Visit Summary", "🖼  Scans", "📑  Official Report"
    ])

    # ══════════════════════════════════════════════
    # TAB 1 — VISIT TRENDS
    # ══════════════════════════════════════════════
    with tab_trends:
        sev_reduction = round(
            ((peak_sev - latest_sev) / peak_sev) * 100, 1
        ) if peak_sev > 0 else 0
        fluid_reduction = round(
            ((peak_fluid - curr_fluid) / peak_fluid) * 100, 1
        ) if peak_fluid > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(metric_card("Peak Severity", f"{peak_sev}/100",
                f"{summary_df.loc[summary_df['Peak Severity Score'].idxmax(), 'Visit']}",
                color=severity_color(peak_sev)), unsafe_allow_html=True)
        with col2:
            st.markdown(metric_card("Latest Avg Severity", f"{latest_sev}/100",
                summary_df["Visit"].iloc[-1],
                color=severity_color(latest_sev)), unsafe_allow_html=True)
        with col3:
            st.markdown(metric_card("Severity Reduction", f"{sev_reduction}%",
                "From peak visit", color="#34D399"), unsafe_allow_html=True)
        with col4:
            st.markdown(metric_card("Total Visits", str(len(visits_input)),
                f"{total_scans} B-Scans total", color="#93C5FD"), unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # ── Severity over visits ──
        st.markdown('<div class="section-header">Severity Score — Visit Progression <span class="section-tag">Score / 100</span></div>', unsafe_allow_html=True)
        fig_sev = go.Figure()
        fig_sev.add_trace(go.Scatter(
            x=summary_df["Date"], y=summary_df["Avg Severity Score"],
            name="Avg Severity", mode='lines+markers',
            line=dict(color='#F87171', width=2.5),
            marker=dict(size=9, color=[severity_color(v) for v in summary_df["Avg Severity Score"]],
                        line=dict(color='#141929', width=1.5)),
            fill='tozeroy', fillcolor='rgba(248,113,113,0.07)',
            hovertemplate='<b>%{x}</b><br>Avg Severity: %{y}/100<extra></extra>'
        ))
        fig_sev.add_trace(go.Scatter(
            x=summary_df["Date"], y=summary_df["Peak Severity Score"],
            name="Peak Severity", mode='lines+markers',
            line=dict(color='#FCD34D', width=1.5, dash='dot'),
            marker=dict(size=7, color='#FCD34D'),
            hovertemplate='<b>%{x}</b><br>Peak Severity: %{y}/100<extra></extra>'
        ))
        fig_sev.update_layout(
            paper_bgcolor='#141929', plot_bgcolor='#141929',
            font=dict(family='DM Mono', color='#94A3B8', size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor='#1E2840', linecolor='#1E2840', tickfont=dict(size=10, color='#475569')),
            yaxis=dict(gridcolor='#1E2840', linecolor='#1E2840', tickfont=dict(size=10, color='#475569'),
                       range=[0, 110], ticksuffix='/100'),
            legend=dict(bgcolor='#0F1525', bordercolor='#1E2840', borderwidth=1,
                        font=dict(size=10, color='#94A3B8')),
            height=260,
        )
        st.plotly_chart(fig_sev, use_container_width=True)

        # ── Fluid over visits ──
        st.markdown('<div class="section-header">Fluid Index — Visit Progression <span class="section-tag">Avg Fluid %</span></div>', unsafe_allow_html=True)
        fig_fluid = go.Figure()
        fig_fluid.add_trace(go.Scatter(
            x=summary_df["Date"], y=summary_df["Avg Fluid Index (%)"],
            mode='lines+markers',
            line=dict(color='#2563EB', width=2.5),
            marker=dict(size=8, color=[fluid_color(v) for v in summary_df["Avg Fluid Index (%)"]],
                        line=dict(color='#141929', width=1.5)),
            fill='tozeroy', fillcolor='rgba(37,99,235,0.08)',
            hovertemplate='<b>%{x}</b><br>Avg Fluid: %{y}%<extra></extra>'
        ))
        fig_fluid.update_layout(
            paper_bgcolor='#141929', plot_bgcolor='#141929',
            font=dict(family='DM Mono', color='#94A3B8', size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor='#1E2840', linecolor='#1E2840', tickfont=dict(size=10, color='#475569')),
            yaxis=dict(gridcolor='#1E2840', linecolor='#1E2840', tickfont=dict(size=10, color='#475569'),
                       ticksuffix='%'),
            height=220, showlegend=False,
        )
        st.plotly_chart(fig_fluid, use_container_width=True)

        # ── Fluid compartment breakdown (all scans) ──
        fluid_cols = [c for c in FLUID_CLASSES if c in detail_df.columns]
        if fluid_cols:
            st.markdown('<div class="section-header">Fluid Compartment — All Scans <span class="section-tag">px per class</span></div>', unsafe_allow_html=True)
            colors_map = {
                "IRF":              "#7DD3E8",
                "SRF":              "#60A5FA",
                "Drosenoid PED":    "#FB7185",
                "Fibrovascular PED":"#F472B6",
            }
            fig_comp = go.Figure()
            for cls in fluid_cols:
                fig_comp.add_trace(go.Bar(
                    name=cls, x=detail_df['Filename'], y=detail_df[cls],
                    marker_color=colors_map.get(cls, '#94A3B8'), opacity=0.85,
                    hovertemplate=f'<b>{cls}</b>: %{{y:,}} px<extra></extra>'
                ))
            fig_comp.update_layout(
                paper_bgcolor='#141929', plot_bgcolor='#141929',
                font=dict(family='DM Mono', color='#94A3B8', size=11),
                margin=dict(l=10, r=10, t=10, b=10), barmode='stack',
                legend=dict(bgcolor='#0F1525', bordercolor='#1E2840', borderwidth=1,
                            font=dict(size=10, color='#94A3B8')),
                xaxis=dict(gridcolor='#1E2840', linecolor='#1E2840', tickfont=dict(size=10, color='#475569')),
                yaxis=dict(gridcolor='#1E2840', linecolor='#1E2840', tickfont=dict(size=10, color='#475569'),
                           ticksuffix=' px'),
                height=220,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # ══════════════════════════════════════════════
    # TAB 2 — PER-VISIT SUMMARY
    # ══════════════════════════════════════════════
    with tab_visits:
        for vs in visit_summaries:
            sev_c = severity_color(vs["Avg Severity Score"])
            fl_c  = fluid_color(vs["Avg Fluid Index (%)"])
            with st.expander(
                f"{vs['Visit']}  ·  {vs['Date']}  ·  {vs['Scans']} scans  ·  "
                f"Avg Severity: {vs['Avg Severity Score']}/100 [{vs['Grade']}]",
                expanded=True
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
                        str(vs["Scans"]), "Slices analyzed", color="#93C5FD"), unsafe_allow_html=True)

                # Measurements table for this visit (aggregated)
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Aggregated Lesion Measurements</div>', unsafe_allow_html=True)

                lesion_rows = []
                for lesion_name in SEVERITY_WEIGHTS:
                    total_px   = sum(s.get(lesion_name, 0) for s in vs["Raw Scans"])
                    avg_px     = round(total_px / vs["Scans"], 1)
                    total_mm2  = round(total_px * PIXEL_TO_MM2, 3)
                    avg_pct    = round(total_px / (TOTAL_PX * vs["Scans"]) * 100, 2)
                    lesion_rows.append({
                        "Lesion":            lesion_name,
                        "Weight":            SEVERITY_WEIGHTS[lesion_name],
                        "Total Pixels":      f"{total_px:,}",
                        "Avg px / scan":     f"{avg_px:,}",
                        "Total Area (mm²)":  total_mm2,
                        "Avg Coverage (%)":  avg_pct,
                    })
                st.dataframe(
                    pd.DataFrame(lesion_rows),
                    hide_index=True,
                    use_container_width=True
                )

    # ══════════════════════════════════════════════
    # TAB 3 — SCANS
    # ══════════════════════════════════════════════
    with tab_scans:
        for vs in visit_summaries:
            st.markdown(f'<div class="section-header">{vs["Visit"]} — {vs["Date"]}</div>', unsafe_allow_html=True)
            for data in vs["Raw Scans"]:
                f_val  = data["Fluid Index (%)"]
                s_val  = data["Severity Score"]
                fc     = fluid_color(f_val)
                sc     = severity_color(s_val)
                status_label = "High" if f_val > 1.5 else ("Moderate" if f_val > 1 else "Low")

                with st.expander(
                    f"{data['Filename']}   —   Severity: {s_val}/100 [{data['Severity Grade']}]"
                    f"   |   Fluid: {f_val}%   [{status_label}]"
                ):
                    c1, c2, c3 = st.columns([2.2, 2.2, 1.3])

                    with c1:
                        st.markdown('<div class="scan-panel-header"><span class="scan-panel-title">Input B-Scan</span></div>', unsafe_allow_html=True)
                        st.image(data['Original'], use_container_width=True)

                    with c2:
                        st.markdown('<div class="scan-panel-header"><span class="scan-panel-title">AI Segmentation Mask</span></div>', unsafe_allow_html=True)
                        st.image(data['Mask'], use_container_width=True)

                    with c3:
                        st.markdown(f"""
                        <div style="padding:10px 0">
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

                        tags_html = '<div class="findings-row">'
                        for cls in FLUID_CLASSES:
                            if data.get(cls, 0) > 0:
                                tag_cls = TAG_CLASSES.get(cls, "tag-irf")
                                tags_html += f'<span class="finding-tag {tag_cls}">{cls}: {data[cls]:,}px</span>'
                        if data.get("HRF", 0) > 0:
                            tags_html += f'<span class="finding-tag tag-hrf">HRF: {data["HRF"]:,}px</span>'
                        if data.get("SHRM", 0) > 0:
                            tags_html += f'<span class="finding-tag tag-shrm">SHRM: {data["SHRM"]:,}px</span>'
                        tags_html += '</div>'
                        st.markdown(tags_html, unsafe_allow_html=True)

                    # ── Detailed measurements table ──
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    st.markdown('<div class="section-header">Pixel Measurements & Area Estimates</div>', unsafe_allow_html=True)
                    meas_rows = []
                    for lesion, vals in data["Measurements"].items():
                        if vals["px"] > 0:
                            meas_rows.append({
                                "Lesion":         lesion,
                                "Pixels":         f"{vals['px']:,}",
                                "Area (mm²)":     vals["area_mm2"],
                                "Coverage (%)":   vals["pct"],
                                "Sev. Weight":    SEVERITY_WEIGHTS.get(lesion, "—"),
                            })
                    if meas_rows:
                        st.dataframe(pd.DataFrame(meas_rows), hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════════
    # TAB 4 — OFFICIAL REPORT
    # ══════════════════════════════════════════════
    with tab_report:
        st.markdown('<div class="section-header">Patient Information</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            p_name = st.text_input("Patient Full Name", placeholder="e.g. Mohamed Al-Rashidi")
            p_id   = st.text_input("Patient ID / MRN",  placeholder="e.g. MRN-004024")
        with col2:
            p_age    = st.number_input("Age", 1, 120, 45)
            p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.button("⚡ Generate AI Clinical Draft", type="primary"):
            if not p_name or not p_id:
                st.warning("Please enter patient name and ID first.")
            else:
                with st.spinner("Synthesizing clinical findings with Llama 3.3..."):
                    visit_lines = []
                    for vs in visit_summaries:
                        visit_lines.append(
                            f"{vs['Visit']} ({vs['Date']}): "
                            f"Avg Severity={vs['Avg Severity Score']}/100 [{vs['Grade']}], "
                            f"Avg Fluid={vs['Avg Fluid Index (%)']}%, "
                            f"Peak Severity={vs['Peak Severity Score']}/100"
                        )
                    visit_summary_str = "\n".join(visit_lines)

                    cols = ["Visit", "Date", "Filename", "Fluid Index (%)", "Severity Score", "Severity Grade"] + \
                           [c for c in FLUID_CLASSES if c in detail_df.columns]
                    scan_table = detail_df[cols].to_string(index=False)

                    prompt = (
                        f"Write a detailed one-page clinical OCT report for a retinal specialist.\n"
                        f"Patient: {p_name}, Age: {p_age}, Gender: {p_gender}\n\n"
                        f"VISIT SUMMARY:\n{visit_summary_str}\n\n"
                        f"DETAILED SCAN DATA:\n{scan_table}\n\n"
                        f"Include: clinical summary, key findings per visit, overall impression, "
                        f"treatment recommendations, and disease activity trend."
                    )
                    st.session_state['report_text'] = get_groq_ai_response(prompt)

        if 'report_text' in st.session_state:
            st.markdown('<hr style="border-color:#1E2840;margin:14px 0">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Review & AI Refinement</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="report-card">
              <div class="ai-badge"><div class="ai-dot"></div> AI Clinical Draft — Llama 3.3 70B</div>
              <div class="report-body">{st.session_state['report_text']}</div>
            </div>
            """, unsafe_allow_html=True)

            final_report = st.text_area(
                "Edit Clinical Summary",
                value=st.session_state['report_text'],
                height=280
            )
            st.session_state['report_text'] = final_report

            user_instruction = st.chat_input("Ask AI to refine report  (e.g. 'Make it more formal' / 'Add ICD-10 codes')")
            if user_instruction:
                with st.spinner("AI refining report..."):
                    refine_prompt = f"Original:\n{st.session_state['report_text']}\n\nInstruction: {user_instruction}"
                    st.session_state['report_text'] = get_groq_ai_response(refine_prompt)
                    st.rerun()

            st.markdown('<hr style="border-color:#1E2840;margin:14px 0">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Finalize Official Document</div>', unsafe_allow_html=True)

            col_pdf, col_prnt = st.columns(2)
            p_info = {"name": p_name, "age": p_age, "gender": p_gender, "id": p_id}

            with col_pdf:
                clean_name    = p_name.replace(" ", "_")
                clean_id      = p_id.replace(" ", "_")
                download_name = f"{clean_name}_{clean_id}.pdf"
                pdf_data      = create_medical_pdf(
                    p_info, dr_input, st.session_state['report_text'],
                    visit_summary=visit_summaries
                )
                st.download_button(
                    label="📥 Download Signed PDF",
                    data=io.BytesIO(pdf_data),
                    file_name=download_name,
                    mime="application/pdf"
                )

            with col_prnt:
                if st.button("🖨 Print Preview"):
                    import re
                    report_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', st.session_state["report_text"])
                    report_html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', report_html)
                    report_html = report_html.replace('\n\n', '</p><p style="margin:6px 0;font-size:13px;color:#1a1a1a;line-height:1.7;">')
                    report_html = report_html.replace('\n', '<br>')

                    visit_rows_html = ""
                    for vs in visit_summaries:
                        sc = {"SEVERE": "#dc2626", "MODERATE": "#d97706", "MILD": "#16a34a", "MINIMAL": "#64748b"}.get(vs["Grade"], "#64748b")
                        visit_rows_html += f"""
                        <tr>
                          <td>{vs['Visit']}</td>
                          <td>{vs['Date']}</td>
                          <td>{vs['Scans']}</td>
                          <td style="color:{sc};font-weight:700">{vs['Avg Severity Score']}/100</td>
                          <td style="color:{sc}">{vs['Grade']}</td>
                          <td>{vs['Avg Fluid Index (%)']}%</td>
                        </tr>"""

                    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#f0f4f8; font-family:Arial,sans-serif; padding:20px; }}
  .page {{ background:#fff; max-width:780px; margin:0 auto; border:1px solid #c8d6e5; border-radius:4px; overflow:hidden; box-shadow:0 2px 12px rgba(0,0,0,0.08); }}
  .hospital-header {{ background:#0a2744; color:white; padding:18px 30px; display:flex; align-items:center; justify-content:space-between; }}
  .hospital-name {{ font-size:15px; font-weight:700; color:#fff; }}
  .hospital-sub {{ font-size:11px; color:#93c5fd; margin-top:2px; }}
  .report-badge {{ background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.25); padding:5px 14px; border-radius:4px; font-size:11px; color:#bfdbfe; text-transform:uppercase; }}
  .title-bar {{ background:#f8fafc; border-bottom:2px solid #e2e8f0; padding:14px 30px; text-align:center; }}
  .report-title {{ font-size:17px; font-weight:700; color:#0a2744; }}
  .report-subtitle {{ font-size:11px; color:#64748b; margin-top:3px; }}
  .info-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; border-bottom:2px solid #e2e8f0; }}
  .info-cell {{ padding:10px 20px; border-right:1px solid #e2e8f0; }}
  .info-cell:last-child {{ border-right:none; }}
  .info-label {{ font-size:9px; font-weight:700; color:#94a3b8; letter-spacing:1px; text-transform:uppercase; margin-bottom:3px; }}
  .info-value {{ font-size:13px; font-weight:600; color:#0f172a; }}
  .section-bar {{ background:#0a2744; color:white; padding:7px 30px; font-size:10px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; }}
  .visit-table {{ width:100%; border-collapse:collapse; font-size:12px; margin:0; }}
  .visit-table th {{ background:#f1f5f9; color:#475569; font-size:10px; letter-spacing:0.8px; text-transform:uppercase; padding:8px 12px; text-align:left; border-bottom:1px solid #e2e8f0; }}
  .visit-table td {{ padding:7px 12px; border-bottom:1px solid #f1f5f9; color:#1e293b; }}
  .visit-table tr:last-child td {{ border-bottom:none; }}
  .findings-body {{ padding:12px 30px; }}
  .findings-body p {{ font-size:13px; color:#1a1a1a; line-height:1.7; margin:6px 0; }}
  .findings-body strong {{ color:#0a2744; font-weight:700; }}
  .stamp-row {{ display:flex; justify-content:space-between; align-items:flex-end; padding:16px 30px; border-top:2px solid #e2e8f0; background:#f8fafc; }}
  .stamp-left {{ font-size:11px; color:#64748b; line-height:1.7; }}
  .sig-line {{ width:180px; border-top:1px solid #334155; margin-bottom:4px; margin-left:auto; }}
  .sig-name {{ font-size:13px; font-weight:700; color:#0a2744; }}
  .sig-title {{ font-size:10px; color:#64748b; }}
  .footer {{ background:#0a2744; color:#93c5fd; font-size:9px; text-align:center; padding:6px; letter-spacing:0.5px; }}
  @media print {{
    body {{ padding:0; background:#fff; }}
    .page {{ box-shadow:none !important; border:none !important; max-width:100%; }}
  }}
</style>
</head>
<body>
<div class="page">
  <div class="hospital-header">
    <div>
      <div class="hospital-name">ALAMEIN INTERNATIONAL UNIVERSITY — AIU</div>
      <div class="hospital-sub">Center for Precision Ophthalmic Intelligence &nbsp;·&nbsp; VisionOCT Pro Suite</div>
    </div>
    <div class="report-badge">Ophthalmology Report</div>
  </div>
  <div class="title-bar">
    <div class="report-title">OCT RETINAL ANALYSIS — MULTI-VISIT CLINICAL REPORT</div>
    <div class="report-subtitle">AI-Assisted Diagnostic Assessment | Confidential Medical Document</div>
  </div>
  <div class="info-grid">
    <div class="info-cell"><div class="info-label">Patient Name</div><div class="info-value">{p_name if p_name else '—'}</div></div>
    <div class="info-cell"><div class="info-label">MRN / ID</div><div class="info-value">{p_id if p_id else '—'}</div></div>
    <div class="info-cell"><div class="info-label">Report Date</div><div class="info-value">{time.strftime("%d %b %Y")}</div></div>
    <div class="info-cell"><div class="info-label">Age</div><div class="info-value">{p_age} yrs</div></div>
    <div class="info-cell"><div class="info-label">Gender</div><div class="info-value">{p_gender}</div></div>
    <div class="info-cell"><div class="info-label">Modality</div><div class="info-value">OCT B-Scan</div></div>
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
    <p style="margin:6px 0;font-size:13px;color:#1a1a1a;line-height:1.7;">{report_html}</p>
  </div>
  <div class="stamp-row">
    <div class="stamp-left">
      <strong style="color:#0a2744">Referring Physician:</strong> Dr. {dr_input}<br>
      Consultant Specialist | AIU Clinical Diagnostic Suite<br>
      Report generated: {time.strftime("%d %B %Y, %H:%M")}
    </div>
    <div class="stamp-right">
      <div class="sig-line"></div>
      <div class="sig-name">Dr. {dr_input}</div>
      <div class="sig-title">Digitally Verified Signature</div>
    </div>
  </div>
  <div class="footer">
    ALAMEIN INTERNATIONAL UNIVERSITY &nbsp;·&nbsp; VisionOCT PRO SUITE &nbsp;·&nbsp; DEVELOPED BY ABDO LASHEEN &nbsp;·&nbsp; 2026 &nbsp;·&nbsp; CONFIDENTIAL
  </div>
</div>
<script>setTimeout(function(){{ window.print(); }}, 400);</script>
</body>
</html>
"""
                    st.components.v1.html(html_content, height=900, scrolling=True)

else:
    st.markdown("""
    <div style="background:#141929;border:1px dashed #263050;border-radius:12px;padding:48px 32px;text-align:center;margin-top:20px">
      <div style="font-size:36px;margin-bottom:16px">👁</div>
      <div style="font-size:16px;font-weight:600;color:#E2E8F0;margin-bottom:8px">VisionOCT Diagnostic Suite</div>
      <div style="font-size:13px;color:#475569;line-height:1.7;max-width:480px;margin:0 auto">
        Set the number of visits above, then upload B-scan images for each visit.<br>
        The system will track Severity Score and Fluid Index across all visits.
      </div>
      <div style="margin-top:20px;display:flex;gap:10px;justify-content:center;flex-wrap:wrap">
        <span style="padding:4px 12px;border-radius:20px;background:rgba(37,99,235,0.1);color:#93C5FD;border:1px solid rgba(37,99,235,0.2);font-size:11px;font-family:'DM Mono',monospace">IRF Detection</span>
        <span style="padding:4px 12px;border-radius:20px;background:rgba(13,148,136,0.1);color:#34D399;border:1px solid rgba(13,148,136,0.2);font-size:11px;font-family:'DM Mono',monospace">SRF Mapping</span>
        <span style="padding:4px 12px;border-radius:20px;background:rgba(255,53,94,0.1);color:#FB7185;border:1px solid rgba(255,53,94,0.2);font-size:11px;font-family:'DM Mono',monospace">PED Classification</span>
        <span style="padding:4px 12px;border-radius:20px;background:rgba(248,113,113,0.1);color:#F87171;border:1px solid rgba(248,113,113,0.2);font-size:11px;font-family:'DM Mono',monospace">Severity Score</span>
        <span style="padding:4px 12px;border-radius:20px;background:rgba(240,120,240,0.1);color:#E879F9;border:1px solid rgba(240,120,240,0.2);font-size:11px;font-family:'DM Mono',monospace">AI Clinical Report</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<div style="margin-top:40px;padding-top:16px;border-top:1px solid #1E2840;text-align:center">
  <span style="font-size:11px;color:#475569;font-family:'DM Mono',monospace">
    VisionOCT Pro Suite &nbsp;·&nbsp; Alamein International University &nbsp;·&nbsp; Developed by Abdo Lasheen &nbsp;·&nbsp; 2026
  </span>
</div>
""", unsafe_allow_html=True)
