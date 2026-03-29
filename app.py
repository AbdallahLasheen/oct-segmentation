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

# ── 1. Page Configuration & Professional Styling
st.set_page_config(page_title="VisionOCT Pro | Clinical Analytics", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 0 25px;
        font-weight: 600;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .report-box {
        background-color: #f1f5f9;
        border-left: 6px solid #0f172a;
        padding: 25px;
        border-radius: 10px;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #1e293b;
    }
    </style>
    """, unsafe_allow_html=True)

# ── 2. Constants & Medical Mapping
LABEL_MAP = [
    ("Background", (0,0,0)), ("Drosenoid PED", (248,231,180)),
    ("Fibrovascular PED", (255,53,94)), ("HRF", (240,120,240)),
    ("IRF", (170,223,235)), ("PH", (51,221,255)),
    ("SHRM", (204,153,51)), ("SRF", (42,125,209)),
]
FLUID_CLASSES = ["IRF", "SRF", "Drosenoid PED", "Fibrovascular PED"]
CLASS_COLORS = np.array([c for _, c in LABEL_MAP], dtype=np.uint8)

# ── 3. Core AI Engine
@st.cache_resource
def load_model():
    """Initializes the U-Net architecture and loads weights."""
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=len(LABEL_MAP))
    ckpt = torch.load("unet_oct_best_v2.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["weights"])
    model.eval()
    return model

def analyze_scan(img, model):
    """Predicts segmentation mask and calculates fluid metrics."""
    orig_size = img.size
    resized = img.resize((256, 256), Image.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(resized), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
    
    with torch.no_grad():
        mask_idx = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
    
    stats = {name: int((mask_idx == i).sum()) for i, (name, _) in enumerate(LABEL_MAP)}
    fluid_px = sum(stats[cls] for cls in FLUID_CLASSES)
    
    # Fluid Index Formula:
    # $$\text{Fluid Index} = \frac{\sum \text{Fluid Pixels}}{\text{Total Pixels}} \times 100$$
    fluid_idx = (fluid_px / (256 * 256)) * 100
    
    mask_rgb = Image.fromarray(CLASS_COLORS[mask_idx].astype(np.uint8)).resize(orig_size, Image.NEAREST)
    return mask_rgb, stats, fluid_idx

def get_grok_report(df_summary):
    """Sends quantitative data to Grok Cloud for clinical reasoning."""
    client = OpenAI(
        api_key=st.secrets["GROK_API_KEY"], 
        base_url="https://api.x.ai/v1"
    )
    
    prompt = f"""
    You are Grok, an expert Retinal Specialist AI. Analyze this sequence of OCT segmentation data:
    {df_summary.to_string(index=False)}
    
    1. Summarize the progression of fluid/lesions. 
    2. Determine if the patient is responding to treatment (improving) or worsening.
    3. Provide 3 specific clinical recommendations.
    Keep it professional and concise.
    """
    
    response = client.chat.completions.create(
        model="grok-beta",
        messages=[
            {"role": "system", "content": "You are a clinical assistant for ophthalmologists."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ── 4. Main UI Logic
st.markdown('<div class="main-header"><h1>🏥 VisionOCT Analytics Suite</h1><p>Automated Multi-Scan Segmentation & Grok AI Progression Tracking</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("🎨 Color Legend")
    for name, (r, g, b) in LABEL_MAP:
        if name == "Background": continue
        st.markdown(f'<div style="display:flex;align-items:center;margin-bottom:4px;"><div style="width:12px;height:12px;background:rgb({r},{g},{b});border-radius:3px;margin-right:8px;"></div><small>{name}</small></div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload OCT scans to analyze progression", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    MODEL = load_model()
    clinical_history = []
    
    # --- FIXED: Batch Processing with Flattened Data Structure ---
    with st.status("Analyzing Clinical Sequence...", expanded=True) as status:
        for f in uploaded_files:
            img = Image.open(f).convert("RGB")
            mask, stats, f_idx = analyze_scan(img, MODEL)
            
            # Create a flat dictionary (No nesting inside 'Stats')
            entry = {
                "Filename": f.name, 
                "Original": img, 
                "Mask": mask,
                "Fluid %": round(f_idx, 2)
            }
            entry.update(stats) # Puts IRF, SRF, etc. directly into the entry
            clinical_history.append(entry)
            
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # Create the DataFrame - Now columns like 'IRF' actually exist in the index
    df = pd.DataFrame(clinical_history)
    
    # ── 5. Dashboard Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Progression Insights", "🖼️ Scan Gallery", "🤖 AI Clinical Report"])
    
    with tab1:
        st.subheader("📈 Total Fluid Concentration Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Filename'], y=df['Fluid %'], mode='lines+markers', line=dict(color='#3b82f6', width=4), marker=dict(size=12)))
        fig.update_layout(xaxis_title="Scan Filename", yaxis_title="Fluid Index (%)", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        if len(df) > 1:
            change = df['Fluid %'].iloc[-1] - df['Fluid %'].iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Overall Progression", f"{change:+.2f}%", delta=change, delta_color="inverse")
            with c2: st.markdown(f'<div class="metric-card"><h4>Condition</h4><h3>{"Improving ✅" if change < 0 else "Worsening ⚠️"}</h3></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><h4>Baseline vs Latest</h4><p>{df["Fluid %"].iloc[0]}% → {df["Fluid %"].iloc[-1]}%</p></div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("Detailed Scan Results")
        for data in clinical_history:
            with st.expander(f"🔍 Details: {data['Filename']}"):
                col_a, col_b, col_c = st.columns([2, 2, 1.2])
                with col_a: st.image(data['Original'], caption="Input Scan", width='stretch')
                with col_b: st.image(data['Mask'], caption="AI Mask", width='stretch')
                with col_c:
                    st.write("**Quantification**")
                    st.write(f"Fluid Index: `{data['Fluid %']}%`")
                    st.divider()
                    for cls in FLUID_CLASSES:
                        # Access keys directly since we flattened the dictionary
                        pixel_count = data.get(cls, 0)
                        if pixel_count > 0:
                            st.write(f"• {cls}: **{pixel_count:,} px**")

    with tab3:
        st.subheader("🤖 Grok-Powered Clinical Summary")
        if st.button("Generate Intelligence Report", type="primary"):
            with st.spinner("Grok is reviewing clinical findings..."):
                try:
                    # Select only the relevant columns for analysis
                    analysis_cols = ["Filename", "Fluid %"] + [c for c in FLUID_CLASSES if c in df.columns]
                    report = get_grok_report(df[analysis_cols]) 
                    st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Grok API Error: {e}")

else:
    st.info("👋 Welcome to VisionOCT Pro. Please upload OCT scans to begin analysis.")

st.markdown("<br><hr><center style='color:#64748b;'>VisionOCT Pro Suite | Developed by Abdo Lasheen | 2026</center>", unsafe_allow_html=True)
