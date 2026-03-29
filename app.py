import gradio as gr
import numpy as np
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
from PIL import Image
from huggingface_hub import hf_hub_download

# ── Label Map (must match training exactly)
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
CLASS_NAMES  = [n for n, _ in LABEL_MAP]
CLASS_COLORS = np.array([c for _, c in LABEL_MAP], dtype=np.uint8)
NUM_CLASSES  = len(LABEL_MAP)
IMG_SIZE     = 256
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]
DEVICE       = "cpu"   # HF Spaces free tier uses CPU


def index_to_rgb(idx_mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[idx_mask]


# ── Load model once at startup
def load_model():
    model = smp.Unet(
        encoder_name    = "efficientnet-b0",
        encoder_weights = None,
        in_channels     = 3,
        classes         = NUM_CLASSES,
    )
    ckpt = torch.load("unet_oct_best_v2.pth",
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["weights"])
    model.eval()
    return model

MODEL = load_model()


def preprocess(image: Image.Image):
    orig_size = image.size                                    # (W, H)
    resized   = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    tensor    = TF.normalize(TF.to_tensor(resized), MEAN, STD)
    return tensor.unsqueeze(0), orig_size                    # [1,3,256,256]


def predict_single(image: Image.Image):
    """Run inference on one PIL Image. Returns (mask_rgb PIL, areas dict)."""
    tensor, orig_size = preprocess(image)
    with torch.no_grad():
        logits   = MODEL(tensor)
        idx_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    mask_rgb = Image.fromarray(
        index_to_rgb(idx_mask).astype(np.uint8)
    ).resize(orig_size, Image.NEAREST)

    # Compute lesion areas (pixels per class, excluding background)
    areas = {}
    for i, (name, _) in enumerate(LABEL_MAP):
        if name == "Background":
            continue
        count = int((idx_mask == i).sum())
        if count > 0:
            areas[name] = count

    return mask_rgb, areas


def build_legend() -> str:
    lines = ["**Legend**\n"]
    for name, (r, g, b) in LABEL_MAP:
        if name == "Background":
            continue
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        lines.append(f"🟦 `{hex_color}` &nbsp; **{name}**")
    return "\n\n".join(lines)


def build_summary(all_areas: list[dict], filenames: list[str]) -> str:
    if not all_areas:
        return ""
    lines = ["## 📊 Lesion Summary\n"]
    for fname, areas in zip(filenames, all_areas):
        lines.append(f"### 🖼️ {fname}")
        if not areas:
            lines.append("_No lesions detected — retina appears normal._\n")
        else:
            for name, px in sorted(areas.items(), key=lambda x: -x[1]):
                bar_len = min(30, px // 500 + 1)
                bar     = "█" * bar_len
                lines.append(f"- **{name}**: {px:,} px &nbsp; `{bar}`")
        lines.append("")
    return "\n".join(lines)


# ── Main inference function called by Gradio
def run_segmentation(images):
    if images is None or len(images) == 0:
        return [], [], "⚠️ Please upload at least one image."

    result_pairs = []   # list of [original, mask] for the gallery
    all_areas    = []
    filenames    = []

    for img in images:
        pil_img  = Image.fromarray(img).convert("RGB")
        mask_pil, areas = predict_single(pil_img)
        result_pairs.append(pil_img)
        result_pairs.append(mask_pil)
        all_areas.append(areas)
        filenames.append("OCT Scan")   # Gradio doesn't expose filenames here

    summary = build_summary(all_areas, filenames)
    return result_pairs, summary


# ── Gradio UI
LEGEND_MD = build_legend()

with gr.Blocks(
    title="OCT Retinal Lesion Segmentation",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css="""
        #header { text-align: center; padding: 1.5rem 0 0.5rem; }
        #header h1 { font-size: 2rem; font-weight: 700; }
        #header p  { color: #64748b; font-size: 1rem; margin-top: 0.25rem; }
        .gallery-item img { border-radius: 8px; }
        #footer { text-align: center; color: #94a3b8;
                  font-size: 0.8rem; padding: 1rem 0; }
    """,
) as demo:

    # ── Header
    gr.HTML("""
        <div id="header">
          <h1>🔬 OCT Retinal Lesion Segmentation</h1>
          <p>AI-powered segmentation of AMD &amp; DME lesions from OCT scans
             &nbsp;·&nbsp; U-Net + EfficientNet-B0 &nbsp;·&nbsp; 8 classes</p>
        </div>
    """)

    with gr.Row():
        # Left column — upload + controls
        with gr.Column(scale=1):
            image_input = gr.Image(
                type        = "numpy",
                label       = "Upload OCT Scan(s)",
                sources     = ["upload"],
            )
            run_btn = gr.Button(
                "▶  Run Segmentation",
                variant = "primary",
                size    = "lg",
            )
            gr.Markdown(LEGEND_MD)

        # Right column — results
        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label        = "Results  (original → mask)",
                columns      = 2,
                object_fit   = "contain",
                height       = 500,
                show_label   = True,
            )
            summary_md = gr.Markdown(label="Lesion Summary")

    # ── Examples
    gr.Examples(
        examples    = [],          # add sample image paths here if available
        inputs      = [image_input],
        label       = "Example Images",
    )

    # ── Footer
    gr.HTML("""
        <div id="footer">
            Trained on AMD &amp; DME OCT datasets &nbsp;·&nbsp;
            Mean Dice 64% &nbsp;·&nbsp;
            For research use only — not a clinical diagnostic tool.
        </div>
    """)

    # ── Event
    run_btn.click(
        fn      = run_segmentation,
        inputs  = [image_input],
        outputs = [gallery, summary_md],
    )

demo.launch()
