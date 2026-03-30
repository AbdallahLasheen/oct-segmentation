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
from fpdf import FPDF
import time
import io

# ── 1. Page Configuration & Professional Branding
st.set_page_config(
    page_title="VisionOCT Pro | Clinical Diagnostic Suite", 
    layout="wide", 
    page_icon="🏥"
)

# Professional Medical-Grade UI Styling
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
    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
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
        border: 1px solid #e2e8f0;
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
FLUID_CLASSES = ["IRF", "SRF", "Drosenoid PED", "Fibrovascular PED"]
CLASS_COLORS = np.array([c for _, c in LABEL_MAP], dtype=np.uint8)

# ── 3. Optimized AI Segmentation Engine
@st.cache_resource
def load_model():
    """Initializes the AI model and loads weights from the .pth file."""
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=len(LABEL_MAP))
    try:
        with open("unet_oct_best_v2.pth", "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["weights"])
    except Exception as e:
        st.error(f"Critical Error: Failed to load model weights. {e}")
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
    
    # Fluid Index: Total fluid pixels / Total scan pixels
    fluid_idx = (fluid_px / (256 * 256)) * 100
    
    mask_rgb = Image.fromarray(CLASS_COLORS[mask_idx].astype(np.uint8)).resize(orig_size, Image.NEAREST)
    return mask_rgb, stats, fluid_idx

# ── 4. Groq Intelligence & PDF Generator
def get_groq_report(df_summary):
    """Generates a professional clinical summary using Groq Cloud (Llama 3.3)."""
    client = OpenAI(api_key=st.secrets["GROK_API_KEY"], base_url="https://api.groq.com/openai/v1")
    prompt = f"Analyze this quantitative OCT sequence data and provide a detailed clinical report:\n{df_summary.to_string(index=False)}"
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "You are a professional board-certified retinal specialist assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

def create_medical_pdf(p_info, dr_name, report_text):
    """Generates an official medical PDF report. Returns bytes."""
    pdf = FPDF()
    pdf.add_page()
    
    # Header: Branding & Logo
    try:
        pdf.image("uni_logo.png", x=10, y=8, w=30)
    except: pass
    
    pdf.set_font("Arial", 'B', 16)
    pdf.set_x(45)
    pdf.cell(0, 10, "NILE UNIVERSITY - CLINICAL DIAGNOSTICS", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.set_x(45)
    pdf.cell(0, 5, "VisionOCT Advanced AI Diagnostic Laboratory", ln=True)
    pdf.ln(15)
    
    # Patient Demographic Box
    pdf.set_fill_color(245, 247, 250)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, " PATIENT INFORMATION", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(95, 7, f" Name: {p_info['name']}", border=1)
    pdf.cell(95, 7, f" Patient ID: {p_info['id']}", border=1, ln=True)
    pdf.cell(95, 7, f" Age: {p_info['age']}", border=1)
    pdf.cell(95, 7, f" Gender: {p_info['gender']}", border=1, ln=True)
    pdf.ln(10)
    
    # Report Findings
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, " CLINICAL FINDINGS & AI ANALYSIS", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, report_text.replace("**", ""))
    
    # Signature Footer
    pdf.set_y(-35)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, "Digitally Verified by:", ln=True, align='R')
    pdf.set_font("Arial", 'I', 11)
    pdf.cell(0, 6, f"Dr. {dr_name}", ln=True, align='R')
    pdf.set_font("Arial", size=8)
    pdf.cell(0, 5, "Consultant Specialist | VisionOCT System", ln=True, align='R')
    
    # FIX: Output as bytes
    return pdf.output()

# ── 5. Main Application Logic
st.markdown('<div class="main-header"><h1>🏥 VisionOCT Diagnostic Suite</h1><p>Clinical Neural Segmentation & LPU-Powered AI Reporting</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("👨‍⚕️ Clinician Information")
    dr_input = st.text_input("Consultant Physician Name", value="Ahmed Younis")
    st.divider()
    st.subheader("🎨 Lesion Map")
    for name, (r, g, b) in LABEL_MAP:
        if name == "Background": continue
        st.markdown(f'<div style="display:flex;align-items:center;margin-bottom:5px;"><div style="width:12px;height:12px;background:rgb({r},{g},{b});border-radius:3px;margin-right:10px;"></div><small>{name}</small></div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload Patient OCT Sequence (JPG, PNG)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    MODEL = load_model()
    clinical_history = []
    
    with st.status("Analyzing Retinal Scans...", expanded=True) as status:
        for f in uploaded_files:
            img = Image.open(f).convert("RGB")
            mask, stats, f_idx = analyze_scan(img, MODEL)
            entry = {"Filename": f.name, "Original": img, "Mask": mask, "Fluid Index (%)": round(f_idx, 2)}
            entry.update(stats)
            clinical_history.append(entry)
        status.update(label="Sequence Analysis Complete", state="complete", expanded=False)

    df = pd.DataFrame(clinical_history)
    t1, t2, t3 = st.tabs(["📉 Temporal Trends", "🖼️ Results Gallery", "📑 Official Report"])

    with t1:
        st.subheader("Fluid Concentration Trends Over Sequence")
        fig = go.Figure(go.Scatter(x=df['Filename'], y=df['Fluid Index (%)'], mode='lines+markers', line=dict(color='#3b82f6', width=4), marker=dict(size=12)))
        st.plotly_chart(fig, use_container_width=True)
        
    with t2:
        for data in clinical_history:
            with st.expander(f"👁️ View Details: {data['Filename']}"):
                c1, c2, c3 = st.columns([2, 2, 1.2])
                c1.image(data['Original'], caption="Input Scan", width='stretch')
                c2.image(data['Mask'], caption="AI Segmentation", width='stretch')
                with c3:
                    st.write(f"**Fluid Index: {data['Fluid Index (%)']}%**")
                    st.divider()
                    for cls in FLUID_CLASSES:
                        if data.get(cls, 0) > 0: st.write(f"• {cls}: {data[cls]:,} px")

    with t3:
        st.subheader("📝 Official Clinical Documentation")
        
        # Step 1: Patient Demographic Entry
        with st.expander("👤 Step 1: Patient Information", expanded=True):
            col1, col2 = st.columns(2)
            p_name = col1.text_input("Patient Full Name")
            p_id = col1.text_input("Patient MRN / ID")
            p_age = col2.number_input("Age", 1, 120, 45)
            p_gender = col2.selectbox("Gender", ["Male", "Female", "Other"])

        # Step 2: AI Generation
        if st.button("🚀 Generate Diagnostic Summary", type="primary"):
            if not p_name or not dr_input:
                st.warning("Please ensure Patient and Physician names are filled.")
            else:
                with st.spinner("LPU Engine is synthesizing clinical findings..."):
                    cols = ["Filename", "Fluid Index (%)"] + [c for c in FLUID_CLASSES if c in df.columns]
                    st.session_state['report_text'] = get_groq_report(df[cols])

        # Step 3: Review, Edit & Finalize
        if 'report_text' in st.session_state:
            st.markdown("---")
            st.markdown("### ✍️ Step 2: Professional Review & Edit")
            st.info("You can manually edit the AI-generated summary below before finalizing the report.")
            
            final_report = st.text_area("Clinical Summary Editor", value=st.session_state['report_text'], height=400)
            st.session_state['report_text'] = final_report # Save changes
            
            st.markdown("---")
            st.markdown("### 🖨️ Step 3: Finalize Official Document")
            col_pdf, col_prnt = st.columns(2)
            
            p_info = {"name": p_name, "age": p_age, "gender": p_gender, "id": p_id}
            
            with col_pdf:
                # FIX: Generate PDF as bytes and wrap in BytesIO for st.download_button
                pdf_data = create_medical_pdf(p_info, dr_input, st.session_state['report_text'])
                pdf_buffer = io.BytesIO(pdf_data)
                
                st.download_button(
                    label="📥 Download Signed Medical PDF", 
                    data=pdf_buffer, 
                    file_name=f"OCT_Report_{p_name.replace(' ', '_')}.pdf", 
                    mime="application/pdf"
                )
            
            with col_prnt:
                if st.button("🖨️ Open Print Preview"):
                    st.markdown(f'<div style="background:white; color:black; padding:40px; border:2px solid #3b82f6; border-radius:15px; font-family:serif; text-align:left;">'
                                f'<h2 style="text-align:center;">Clinical Assessment Report</h2><hr>'
                                f'<p><b>Physician:</b> Dr. {dr_input} | <b>Date:</b> {time.strftime("%Y-%m-%d")}</p>'
                                f'<p><b>Patient:</b> {p_name} | <b>ID:</b> {p_id}</p><hr>'
                                f'<p style="white-space:pre-wrap; font-size:1.1rem;">{st.session_state["report_text"]}</p>'
                                f'<br><br><p style="text-align:right;"><b>Verified by Dr. {dr_input}</b></p></div>', unsafe_allow_html=True)
                    st.components.v1.html("<script>window.print();</script>", height=0)

else:
    st.info("👋 Welcome, Dr. Abdo. Please upload the patient's OCT scans to initiate analysis.")

st.markdown("<br><hr><center style='color:#64748b;'>VisionOCT Pro Suite | Developed by Abdo Lasheen | 2026</center>", unsafe_allow_html=True)
