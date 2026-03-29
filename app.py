import streamlit as st
import numpy as np
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import time

# ── 1. Page Configuration & Professional Branding
st.set_page_config(
    page_title="VisionOCT Pro | Clinical Diagnostic Suite", 
    layout="wide", 
    page_icon="🏥"
)

# Custom CSS for a high-end medical software feel
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 3rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #334155;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] { 
        gap: 15px; 
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background-color: #f1f5f9;
        border-radius: 12px;
        padding: 0 30px;
        font-weight: 600;
        color: #475569;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
        border: 1px solid #3b82f6 !important;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    .report-box {
        background-color: #ffffff;
        border-left: 8px solid #3b82f6;
        padding: 30px;
        border-radius: 15px;
        font-size: 1.1rem;
        line-height: 1.8;
        color: #1e293b;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #dcfce7;
        color: #166534;
    }
    </style>
    """, unsafe_allow_html=True)

# ── 2. Clinical Constants & Label Mapping
LABEL_MAP = [
    ("Background", (0,0,0)), ("Drosenoid PED", (248,231,180)),
    ("Fibrovascular PED", (255,53,94)), ("HRF", (240,120,240)),
    ("IRF", (170,223,235)), ("PH", (51,221,255)),
    ("SHRM", (204,153,51)), ("SRF", (42,125,209)),
]
# Critical Fluid classes for medical progression tracking
FLUID_CLASSES = ["IRF", "SRF", "Drosenoid PED", "Fibrovascular PED"]
CLASS_COLORS = np.array([c for _, c in LABEL_MAP], dtype=np.uint8)

# ── 3. Optimized AI Segmentation Engine
@st.cache_resource
def load_model():
    """Loads U-Net model with EfficientNet backbone."""
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=len(LABEL_MAP))
    try:
        # Loading using binary mode to prevent EOFError issues
        with open("unet_oct_best_v2.pth", "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["weights"])
    except Exception as e:
        st.error(f"Critical Error: Failed to load model weights. Details: {e}")
    model.eval()
    return model

def analyze_scan(img, model):
    """Predicts segmentation and calculates fluid concentration index."""
    orig_size = img.size
    resized = img.resize((256, 256), Image.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(resized), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
    
    with torch.no_grad():
        mask_idx = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
    
    stats = {name: int((mask_idx == i).sum()) for i, (name, _) in enumerate(LABEL_MAP)}
    fluid_px = sum(stats[cls] for cls in FLUID_CLASSES)
    
    # Mathematical Model for Fluid Concentration
    # $$\text{Fluid Index} = \frac{\sum \text{Fluid Pixels}}{\text{Total Pixels}} \times 100$$
    fluid_idx = (fluid_px / (256 * 256)) * 100
    
    mask_rgb = Image.fromarray(CLASS_COLORS[mask_idx].astype(np.uint8)).resize(orig_size, Image.NEAREST)
    return mask_rgb, stats, fluid_idx

# ── 4. Groq-Powered Clinical Intelligence
def get_groq_intelligence_report(df_summary):
    """Uses Groq LPU technology for near-instant clinical reasoning."""
    # Note: Using Groq endpoint with llama-3.3-70b for high clinical accuracy
    client = OpenAI(
        api_key=st.secrets["GROK_API_KEY"], 
        base_url="https://api.groq.com/openai/v1" 
    )
    
    stats_string = df_summary.to_string(index=False)
    
    prompt = f"""
    You are an expert Retinal Image Analysis AI. Review this quantitative OCT progression data:
    {stats_string}
    
    Required Analysis:
    1. Trend Summary: Describe the change in fluid types (IRF, SRF, etc.) across the temporal sequence.
    2. Response to Treatment: Based on the Fluid Index trend, is the pathology resolving or progressing?
    3. Clinical Advice: Provide 3 high-level recommendations for the attending ophthalmologist.
    
    Tone: Professional, clinical, and data-driven.
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a board-certified ophthalmic diagnostic assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1 # Low temperature for medical consistency
    )
    return response.choices[0].message.content

# ── 5. Main Clinical Dashboard
st.markdown("""
    <div class="main-header">
        <h1>🏥 VisionOCT Diagnostic Suite</h1>
        <p>Advanced Neural Segmentation & Real-Time AI Clinical Reporting</p>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔬 System Configuration")
    st.markdown('<span class="status-badge">AI Engine: LPU Enabled ⚡</span>', unsafe_allow_html=True)
    st.divider()
    st.subheader("🎨 Lesion Color Map")
    for name, (r, g, b) in LABEL_MAP:
        if name == "Background": continue
        st.markdown(f'''
            <div style="display:flex; align-items:center; margin-bottom:8px;">
                <div style="width:14px; height:14px; background:rgb({r},{g},{b}); border-radius:4px; margin-right:10px; border:1px solid #cbd5e1;"></div>
                <span style="font-size:0.9rem; color:#475569;">{name}</span>
            </div>
            ''', unsafe_allow_html=True)

# Main File Upload Interface
uploaded_files = st.file_uploader(
    "📥 Drag and drop a sequence of patient OCT scans (JPG, PNG)", 
    type=["jpg", "png", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    MODEL = load_model()
    clinical_history = []
    
    # Sequence Processing
    with st.status("Analyzing Retinal Layers & Fluid Patterns...", expanded=True) as status:
        for f in uploaded_files:
            img = Image.open(f).convert("RGB")
            mask, stats, f_idx = analyze_scan(img, MODEL)
            
            # Flattened data structure for reliable DataFrame indexing
            entry = {
                "Filename": f.name, 
                "Original": img, 
                "Mask": mask,
                "Fluid Index (%)": round(f_idx, 2)
            }
            entry.update(stats) # Integrate lesion counts directly
            clinical_history.append(entry)
            
        status.update(label="Sequence Analysis Complete!", state="complete", expanded=False)

    df = pd.DataFrame(clinical_history)
    
    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs(["📉 Temporal Trends", "🖼️ Segmentation Gallery", "📑 AI Clinical Report"])
    
    with tab1:
        st.subheader("Temporal Progression of Retinal Fluid")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Filename'], y=df['Fluid Index (%)'], 
            mode='lines+markers', 
            line=dict(color='#3b82f6', width=4),
            marker=dict(size=14, color='white', line=dict(color='#3b82f6', width=2))
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Scan Filename (Sequence Order)", yaxis_title="Fluid Concentration (%)",
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Clinical Metrics
        if len(df) > 1:
            change = df['Fluid Index (%)'].iloc[-1] - df['Fluid Index (%)'].iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Global Trend", f"{change:+.2f}%", delta=change, delta_color="inverse")
            with c2: 
                status_text = "Improving ✅" if change < 0 else "Worsening ⚠️"
                st.markdown(f'<div class="metric-card"><h5>Clinical Status</h5><h3>{status_text}</h3></div>', unsafe_allow_html=True)
            with c3: 
                st.markdown(f'<div class="metric-card"><h5>Fluid Delta</h5><p>{df["Fluid Index (%)"].iloc[0]}% → {df["Fluid Index (%)"].iloc[-1]}%</p></div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("Automated Segmentation Detail")
        for data in clinical_history:
            with st.expander(f"👁️ Analysis: {data['Filename']}"):
                col_a, col_b, col_c = st.columns([2, 2, 1.2])
                with col_a: st.image(data['Original'], caption="Retinal B-Scan", width='stretch')
                with col_b: st.image(data['Mask'], caption="AI Fluid Mapping", width='stretch')
                with col_c:
                    st.write("**Quantification**")
                    st.write(f"Fluid Index: `{data['Fluid Index (%)']}%`")
                    st.divider()
                    for cls in FLUID_CLASSES:
                        px_count = data.get(cls, 0)
                        if px_count > 0:
                            st.write(f"• {cls}: **{px_count:,} px**")

    with tab3:
        st.subheader("🤖 Groq-AI Clinical Interpretation")
        if st.button("Generate Diagnostic Summary", type="primary"):
            with st.spinner("LPU Engine is synthesizing clinical findings..."):
                try:
                    # Select only necessary columns for the AI report
                    analysis_cols = ["Filename", "Fluid Index (%)"] + [c for c in FLUID_CLASSES if c in df.columns]
                    report = get_groq_intelligence_report(df[analysis_cols]) 
                    st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Intelligence Engine Error: {e}")

else:
    st.info("🏥 Welcome, Dr. Abdo. Please upload a patient's OCT scan sequence to initiate AI analysis.")

# Footer
st.markdown("""
    <br><hr>
    <div style="text-align:center; color:#64748b; font-size:0.9rem; padding-bottom:20px;">
        VisionOCT Pro Suite | Clinical Decision Support System | Developed by Abdo Lasheen | 2026
    </div>
    """, unsafe_allow_html=True)
