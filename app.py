import streamlit as st
import numpy as np
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
from PIL import Image
import time

# ── 1. Page Configuration & Styling
st.set_page_config(page_title="VisionOCT | Advanced Lesion Segmentation", layout="wide", page_icon="🔬")

# Professional CSS Injection
st.markdown("""
    <style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Custom Cards */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-5px); border-color: #3b82f6; }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5rem;
        background-color: #2563eb;
        color: white;
        font-weight: bold;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover { background-color: #1d4ed8; border: none; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ── 2. Constants & Model Loading
LABEL_MAP = [
    ("Background",        (0,   0,   0  )), ("Drosenoid PED",     (248, 231, 180)),
    ("Fibrovascular PED", (255, 53,  94 )), ("HRF",               (240, 120, 240)),
    ("IRF",               (170, 223, 235)), ("PH",                (51,  221, 255)),
    ("SHRM",              (204, 153, 51 )), ("SRF",               (42,  125, 209)),
]
CLASS_COLORS = np.array([c for _, c in LABEL_MAP], dtype=np.uint8)
IMG_SIZE, MEAN, STD = 256, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

@st.cache_resource
def load_model():
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=len(LABEL_MAP))
    ckpt = torch.load("unet_oct_best_v2.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["weights"])
    model.eval()
    return model

def preprocess(image):
    orig_size = image.size
    resized = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(resized), MEAN, STD)
    return tensor.unsqueeze(0), orig_size

# ── 3. Application Interface
st.markdown('<div class="main-header"><h1>🔬 VisionOCT Analysis Pro</h1><p>Next-Gen Retinal Lesion Segmentation Engine</p></div>', unsafe_allow_html=True)

# Tabs for Organization
tab1, tab2 = st.tabs(["🚀 Analysis Dashboard", "📖 Documentation"])

with tab1:
    with st.sidebar:
        st.header("📋 Legend & Settings")
        st.markdown("---")
        for name, (r, g, b) in LABEL_MAP:
            if name == "Background": continue
            st.markdown(f'<div style="display:flex; align-items:center; margin-bottom:5px;">'
                        f'<div style="width:20px; height:20px; background-color:rgb({r},{g},{b}); border-radius:4px; margin-right:10px;"></div>'
                        f'<b>{name}</b></div>', unsafe_allow_html=True)
        st.sidebar.divider()
        st.info("System Status: **Optimal** 🟢")

    # Main Layout
    col_up, col_info = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader("Drop OCT Scan Here", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🖼️ Input Scan")
            st.image(input_image, use_container_width=True)
            
        if st.button("Analyze Scan ✨"):
            with st.spinner("Decoding Retinal Layers..."):
                MODEL = load_model()
                tensor, orig_size = preprocess(input_image)
                time.sleep(1) # Visual effect
                
                with torch.no_grad():
                    logits = MODEL(tensor)
                    idx_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

                mask_rgb = Image.fromarray(CLASS_COLORS[idx_mask].astype(np.uint8)).resize(orig_size, Image.NEAREST)
                
                with c2:
                    st.markdown("### 🎭 Generated Mask")
                    st.image(mask_rgb, use_container_width=True)

                # Quantitative Results
                st.divider()
                st.subheader("📊 Lesion Quantification (Pixel Count)")
                m_cols = st.columns(4)
                found = False
                idx = 0
                for i, (name, _) in enumerate(LABEL_MAP):
                    if name == "Background": continue
                    px_count = int((idx_mask == i).sum())
                    if px_count > 0:
                        found = True
                        with m_cols[idx % 4]:
                            st.markdown(f'<div class="metric-card"><h4>{name}</h4><h2>{px_count:,}</h2><p>Pixels</p></div>', unsafe_allow_html=True)
                        idx += 1
                if not found:
                    st.success("✅ Analysis Complete: No significant lesions detected.")

with tab2:
    st.header("About the Model")
    st.write("This system utilizes a **U-Net** architecture with an **EfficientNet-B0** backbone.")
    
    st.markdown("""
    - **Dataset**: Trained on multicenter OCT scans (AMD/DME).
    - **Mean Dice Score**: 64% 
    - **Classes**: 8 distinct retinal lesion types.
    """)

st.markdown("<br><center style='color:#64748b;'>© 2026 VisionOCT AI Development Team</center>", unsafe_allow_html=True)
