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

# ==========================================================
# SECTION 1: SYSTEM CONFIGURATION & UI STYLING
# ==========================================================
st.set_page_config(
    page_title="VisionOCT Pro | Alamein International University", 
    layout="wide", 
    page_icon="🏥"
)

# Professional Medical UI Styling
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
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# SECTION 2: CLINICAL CONSTANTS
# ==========================================================
LABEL_MAP = [
    ("Background", (0,0,0)), ("Drosenoid PED", (248,231,180)),
    ("Fibrovascular PED", (255,53,94)), ("HRF", (240,120,240)),
    ("IRF", (170,223,235)), ("PH", (51,221,255)),
    ("SHRM", (204,153,51)), ("SRF", (42,125,209)),
]
FLUID_CLASSES = ["IRF", "SRF", "Drosenoid PED", "Fibrovascular PED"]
CLASS_COLORS = np.array([c for _, c in LABEL_MAP], dtype=np.uint8)

# ==========================================================
# SECTION 3: DEEP LEARNING ENGINE (U-NET)
# ==========================================================
@st.cache_resource
def load_model():
    """Initializes the AI model and loads pre-trained weights."""
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=len(LABEL_MAP))
    try:
        with open("unet_oct_best_v2.pth", "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["weights"])
    except Exception as e:
        st.error(f"Critical Error: Failed to load weights. {e}")
    model.eval()
    return model

def analyze_scan(img, model):
    """Predicts segmentation mask and calculates clinical fluid metrics."""
    orig_size = img.size
    resized = img.resize((256, 256), Image.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(resized), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
    
    with torch.no_grad():
        mask_idx = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
    
    stats = {name: int((mask_idx == i).sum()) for i, (name, _) in enumerate(LABEL_MAP)}
    fluid_px = sum(stats[cls] for cls in FLUID_CLASSES)
    fluid_idx = (fluid_px / (256 * 256)) * 100
    
    mask_rgb = Image.fromarray(CLASS_COLORS[mask_idx].astype(np.uint8)).resize(orig_size, Image.NEAREST)
    return mask_rgb, stats, fluid_idx

# ==========================================================
# SECTION 4: CLINICAL AI & PDF LOGIC
# ==========================================================
def get_groq_ai_response(prompt):
    """Fetches medical reasoning from Groq LPU."""
    client = OpenAI(api_key=st.secrets["GROK_API_KEY"], base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "You are a senior retinal specialist assistant. Keep reports professional and very concise to fit one page."},
                  {"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

def create_medical_pdf(p_info, dr_name, report_text):
    """Generates a STRICT 1-PAGE official branded PDF."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False) # Disable auto break to force 1 page
    pdf.add_page()
    
    # Header Branding (AIU)
    try: pdf.image("uni_logo.png", x=10, y=8, w=25)
    except: pass
    
    pdf.set_font("Arial", 'B', 15)
    pdf.set_x(40)
    pdf.cell(0, 10, "ALAMEIN INTERNATIONAL UNIVERSITY - AIU", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.set_x(40)
    pdf.cell(0, 5, "VisionOCT Advanced AI Diagnostic Laboratory", ln=True)
    pdf.ln(10)
    
    # Patient Demographic Box
    pdf.set_fill_color(245, 247, 250)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 7, " PATIENT INFORMATION", ln=True, fill=True)
    pdf.set_font("Arial", size=9)
    pdf.cell(95, 7, f" Name: {p_info['name']}", border=1)
    pdf.cell(95, 7, f" Patient ID: {p_info['id']}", border=1, ln=True)
    pdf.cell(95, 7, f" Age: {p_info['age']}", border=1)
    pdf.cell(95, 7, f" Gender: {p_info['gender']}", border=1, ln=True)
    pdf.ln(8)
    
    # Findings Body
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 7, " CLINICAL FINDINGS & AI ANALYSIS", ln=True)
    pdf.set_font("Arial", size=9)
    # Truncate text if it's too long to ensure it fits page 1
    safe_text = report_text[:2800].replace("**", "")
    pdf.multi_cell(0, 5, safe_text)
    
    # Signature Block (Fixed at the bottom of Page 1)
    pdf.set_y(255) 
    pdf.line(10, 255, 200, 255)
    pdf.ln(2)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 5, "Digitally Verified by:", ln=True, align='R')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 6, f"Dr. {dr_name}", ln=True, align='R')
    
    return bytes(pdf.output())

# ==========================================================
# SECTION 5: MAIN UI LOGIC
# ==========================================================
st.markdown('<div class="main-header"><h1>🏥 VisionOCT Diagnostic Suite</h1><p>Alamein International University | Neural Segmentation & AI Copilot</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("👨‍⚕️ Clinician Setup")
    dr_input = st.text_input("Consultant Physician Name", value="Ahmed Younis")
    st.divider()
    st.info("System Ready 🟢")

uploaded_files = st.file_uploader("Upload OCT Scans", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    MODEL = load_model()
    clinical_history = []
    
    with st.status("Analyzing Scans...", expanded=True) as status:
        for f in uploaded_files:
            img = Image.open(f).convert("RGB")
            mask, stats, f_idx = analyze_scan(img, MODEL)
            clinical_history.append({"Filename": f.name, "Original": img, "Mask": mask, "Fluid Index (%)": round(f_idx, 2), **stats})
        status.update(label="Analysis Complete", state="complete", expanded=False)

    df = pd.DataFrame(clinical_history)
    tab1, tab2, tab3 = st.tabs(["📉 Trends", "🖼️ Scan Gallery", "📑 Official Report"])

    with tab1:
        st.subheader("Temporal Progression of Retinal Fluid")
        fig = go.Figure(go.Scatter(x=df['Filename'], y=df['Fluid Index (%)'], mode='lines+markers', line=dict(color='#3b82f6', width=4)))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        for data in clinical_history:
            with st.expander(f"👁️ Detail: {data['Filename']}"):
                c1, c2, c3 = st.columns([2, 2, 1.2])
                c1.image(data['Original'], caption="Input", use_container_width=True)
                c2.image(data['Mask'], caption="AI Mask", use_container_width=True)
                with c3:
                    st.write(f"**Fluid Index: {data['Fluid Index (%)']}%**")
                    st.divider()
                    for cls in FLUID_CLASSES:
                        if data.get(cls, 0) > 0: st.write(f"• {cls}: {data[cls]:,} px")

    with tab3:
        st.subheader("📝 Official Clinical Documentation")
        
        # Demographic Input Form
        with st.expander("Step 1: Patient Information", expanded=True):
            col1, col2 = st.columns(2)
            p_name = col1.text_input("Patient Full Name")
            p_id = col1.text_input("Patient ID / MRN")
            p_age = col2.number_input("Age", 1, 120, 45)
            p_gender = col2.selectbox("Gender", ["Male", "Female", "Other"])

        if st.button("🚀 Generate AI Clinical Draft", type="primary"):
            if not p_name or not p_id: st.warning("Fill Patient Name and ID.")
            else:
                with st.spinner("Synthesizing clinical findings..."):
                    cols = ["Filename", "Fluid Index (%)"] + [c for c in FLUID_CLASSES if c in df.columns]
                    st.session_state['report_text'] = get_groq_ai_response(f"Draft a concise 1-page report for this data:\n{df[cols].to_string(index=False)}")

        # Review and Interactive Section
        if 'report_text' in st.session_state:
            st.markdown("---")
            st.markdown("### ✍️ Step 2: Review & AI Refinement")
            
            # Manual Text Editor
            final_report = st.text_area("Clinical Summary (Edit manually)", value=st.session_state['report_text'], height=350)
            st.session_state['report_text'] = final_report 

            # AI Chat Input for refinement
            user_instruction = st.chat_input("Ask AI to update the report (e.g. 'Translate to Arabic', 'Focus more on SRF')")
            if user_instruction:
                with st.spinner("AI Assistant is refining..."):
                    refine_prompt = f"Original Report: {st.session_state['report_text']}\n\nDoctor Instruction: {user_instruction}\n\nUpdate report accordingly. Return ONLY the new text."
                    st.session_state['report_text'] = get_groq_ai_response(refine_prompt)
                    st.rerun()

            st.markdown("---")
            st.markdown("### 🖨️ Step 3: Finalize Official Document")
            col_pdf, col_prnt = st.columns(2)
            
            with col_pdf:
                # ── FILENAME LOGIC ([NAME]_[ID].pdf) ──
                clean_name = p_name.replace(" ", "_")
                clean_id = p_id.replace(" ", "_")
                download_filename = f"{clean_name}_{clean_id}.pdf"
                
                p_info = {"name": p_name, "age": p_age, "gender": p_gender, "id": p_id}
                pdf_data = create_medical_pdf(p_info, dr_input, st.session_state['report_text'])
                
                st.download_button(
                    label="📥 Download Signed One-Page PDF", 
                    data=io.BytesIO(pdf_data), 
                    file_name=download_filename, 
                    mime="application/pdf"
                )
            
            with col_prnt:
                if st.button("🖨️ Open Print Preview"):
                    st.markdown(f'<div style="background:white; color:black; padding:40px; border:2px solid #3b82f6; border-radius:15px; font-family:serif; text-align:left;">'
                                f'<h2 style="text-align:center;">Medical Assessment Report</h2>'
                                f'<h4 style="text-align:center; color:#666;">Alamein International University</h4><hr>'
                                f'<p><b>Physician:</b> Dr. {dr_input} | <b>Date:</b> {time.strftime("%Y-%m-%d")}</p>'
                                f'<p><b>Patient:</b> {p_name} | <b>ID:</b> {p_id}</p><hr>'
                                f'<p style="white-space:pre-wrap;">{st.session_state["report_text"]}</p>'
                                f'<br><br><p style="text-align:right;"><b>Digitally Verified by Dr. {dr_input}</b></p></div>', unsafe_allow_html=True)
                    st.components.v1.html("<script>window.print();</script>", height=0)

else:
    st.info("🏥 Welcome, Dr. Abdo. Please upload the patient's OCT scans to initiate analysis.")

# System Footer
st.markdown("<br><hr><center style='color:#64748b;'>VisionOCT Pro Suite | Alamein International University | Developed by Abdo Lasheen | 2026</center>", unsafe_allow_html=True)
