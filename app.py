import streamlit as st
import numpy as np
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
from PIL import Image

# ── 1. Configuration & Constants (Matching your training exactly)
LABEL_MAP = [
    ("Background",        (0,   0,   0  )),
    ("Drosenoid PED",     (248, 231, 180)),
    ("Fibrovascular PED", (255, 53,  94 )),
    ("HRF",               (240, 120, 240)),
    ("IRF",               (170, 223, 235)),
    ("PH",                (51,  221, 255)),
    ("SHRM",              (204, 153, 51 )),
    ("SRF",               (42,  125, 209)),
]

CLASS_COLORS = np.array([c for _, c in LABEL_MAP], dtype=np.uint8)
NUM_CLASSES  = len(LABEL_MAP)
IMG_SIZE     = 256
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]
DEVICE       = "cpu" 

# ── 2. Helper Functions
def index_to_rgb(idx_mask: np.ndarray) -> np.ndarray:
    """Converts a class-index mask to an RGB image."""
    return CLASS_COLORS[idx_mask]

@st.cache_resource # Caches the model in memory to prevent re-loading on every interaction
def load_model():
    """Initializes the U-Net with EfficientNet-B0 backbone and loads weights."""
    model = smp.Unet(
        encoder_name    = "efficientnet-b0",
        encoder_weights = None,
        in_channels     = 3,
        classes         = NUM_CLASSES,
    )
    # Ensure 'unet_oct_best_v2.pth' is in the same folder on GitHub
    ckpt = torch.load("unet_oct_best_v2.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["weights"])
    model.eval()
    return model

def preprocess(image: Image.Image):
    """Resizes and normalizes the input image for the model."""
    orig_size = image.size                                    
    resized   = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    tensor    = TF.normalize(TF.to_tensor(resized), MEAN, STD)
    return tensor.unsqueeze(0), orig_size                    

# ── 3. Streamlit UI Layout
st.set_page_config(page_title="OCT Segmentation AI", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { text-align: center; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 OCT Retinal Lesion Segmentation")
st.markdown("AI-powered segmentation of **AMD & DME** lesions from OCT scans using **U-Net + EfficientNet-B0**.")

# ── 4. Sidebar Legend
with st.sidebar:
    st.header("🎨 Color Legend")
    st.markdown("Class colors used in the mask:")
    for name, (r, g, b) in LABEL_MAP:
        if name == "Background": continue
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        st.markdown(f'<p><span style="color:{hex_color}; font-size:20px;">■</span> <b>{name}</b></p>', unsafe_allow_html=True)
    
    st.divider()
    st.info("Note: For research use only. Not a clinical diagnostic tool.")

# ── 5. Main Logic: Upload and Prediction
MODEL = load_model()

uploaded_file = st.file_uploader("Upload OCT Scan (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Split screen into two columns
    col1, col2 = st.columns(2)
    
    # Load and display original image
    input_image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("🖼️ Original OCT Scan")
        st.image(input_image, use_container_width=True)

    # Prediction Trigger
    if st.button("▶ Run Segmentation"):
        with st.spinner("Analyzing scan... Please wait."):
            # Preprocess
            tensor, orig_size = preprocess(input_image)
            
            # Inference
            with torch.no_grad():
                logits   = MODEL(tensor)
                idx_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

            # Post-process mask
            mask_rgb_array = index_to_rgb(idx_mask).astype(np.uint8)
            mask_pil = Image.fromarray(mask_rgb_array).resize(orig_size, Image.NEAREST)
            
            with col2:
                st.subheader("🎭 Segmentation Mask")
                st.image(mask_pil, use_container_width=True)

            # ── 6. Quantitative Summary
            st.divider()
            st.subheader("📊 Lesion Area Analysis")
            
            has_lesions = False
            # Use columns for metrics to look professional
            metric_cols = st.columns(3)
            col_idx = 0
            
            for i, (name, _) in enumerate(LABEL_MAP):
                if name == "Background": continue
                count = int((idx_mask == i).sum())
                
                if count > 0:
                    has_lesions = True
                    with metric_cols[col_idx % 3]:
                        st.metric(label=name, value=f"{count:,} px")
                    col_idx += 1
            
            if not has_lesions:
                st.success("✅ No lesions detected — The retina appears normal.")

else:
    st.warning("Please upload an image to begin.")

# Footer
st.markdown("<br><hr><center>Developed by Abdallah Lasheen | OCT-Segmentation v2.0</center>", unsafe_allow_html=True)
