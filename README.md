# 🔬 OCT Retinal Lesion Segmentation

> AI-powered segmentation of retinal lesions from Optical Coherence Tomography (OCT) scans — supporting AMD & DME clinical analysis.

---

## 🧠 Overview

This application uses a **U-Net + EfficientNet-B0** deep learning model to automatically segment retinal lesions in OCT images. Upload one or multiple scans and receive instant color-coded segmentation masks with per-lesion pixel analysis across **8 retinal pathology classes**.

Developed as part of a graduation project at **Al-Alamein International University (AIU)**.

---

## ✨ Features

- 📤 **Multi-image upload** — process multiple OCT scans in one run
- 🎨 **Color-coded masks** — each lesion class has a distinct color for easy interpretation
- 📊 **Lesion summary** — pixel-level quantification for each detected pathology
- ⚡ **Fast inference** — results in seconds per image
- 🖥️ **Clean UI** — designed for clinical and research use

---

## 🏥 Detected Lesion Classes

| # | Class | Color |
|---|-------|-------|
| 0 | Background | ⬛ Black |
| 1 | Drosenoid PED | 🟨 Light Yellow |
| 2 | Fibrovascular PED | 🔴 Red |
| 3 | HRF (Hyperreflective Foci) | 🟣 Pink/Magenta |
| 4 | IRF (Intraretinal Fluid) | 🔵 Light Blue |
| 5 | PH (Posterior Hyaloid) | 🩵 Cyan |
| 6 | SHRM (Subretinal Hyperreflective Material) | 🟫 Brown |
| 7 | SRF (Subretinal Fluid) | 💙 Blue |

---

## 🏗️ Model Architecture

```
Input OCT Scan (256×256)
        ↓
EfficientNet-B0 Encoder (ImageNet pretrained)
        ↓
U-Net Decoder with Skip Connections
        ↓
8-Class Segmentation Map (256×256)
```

| Component | Detail |
|-----------|--------|
| Architecture | U-Net |
| Encoder | EfficientNet-B0 (ImageNet pretrained) |
| Parameters | 6.25M |
| Input size | 256 × 256 |
| Output | 8-class segmentation mask |
| Loss | Focal Loss + Dice Loss |
| Sampling | WeightedRandomSampler |

---

## 📊 Performance

| Metric | Score |
|--------|-------|
| Overall Pixel Accuracy | 97.0% |
| Mean Dice Score | 64.2% |
| Mean IoU | 52.4% |

**Per-class Dice:**

| Class | Dice | Status |
|-------|------|--------|
| Background | 98.5% | ✅ Strong |
| IRF | 78.4% | ✅ Strong |
| PH | 73.1% | ✅ Strong |
| Fibrovascular PED | 77.3% | ✅ Strong |
| Drosenoid PED | 72.8% | ✅ Strong |
| SRF | 67.9% | ⚡ Moderate |
| HRF | 41.6% | ⚡ Moderate |
| SHRM | 3.6% | ⚠️ Rare class |

---

## 🗂️ Dataset

| Split | Images |
|-------|--------|
| Train | 345 |
| Validation | 74 |
| Test | 74 |
| **Total** | **493** |

- **Diseases:** AMD (Age-related Macular Degeneration) · DME (Diabetic Macular Edema)
- **Augmentation:** Albumentations — HorizontalFlip, VerticalFlip, Rotate, ElasticTransform, GridDistortion, ColorJitter, GaussianBlur

---

## 🚀 Running Locally

```bash
git clone https://github.com/Abdallah-12/oct-segmentation
cd oct-segmentation
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
oct-segmentation/
├── app.py                  ← Streamlit web application
├── requirements.txt        ← Python dependencies
├── unet_oct_best_v2.pth    ← Trained model weights
└── README.md
```

---

## ⚠️ Disclaimer

> This tool is intended for **research and educational purposes only**.
> It is **not a certified medical device** and should not be used as a substitute
> for professional clinical diagnosis.

---

## 👨‍💻 Author

**Abdallah** — AI & Computer Vision  
Al-Alamein International University · Graduation Project 2026

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.
