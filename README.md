---
title: realtime-face-detection
emoji: 🌖
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: Realtime face detection from video feed using Haar Cascades
---

# 🎯 Realtime Face Detection — Haar Cascade × Hugging Face Spaces

> A lightweight, browser-based **realtime face detection app** powered by **OpenCV Haar Cascade classifiers**, deployed on **Hugging Face Spaces** via **Gradio**, and continuously delivered through a **GitHub Actions CI/CD pipeline**.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/arafmustavi/realtime-face-detection)
[![Made with OpenCV](https://img.shields.io/badge/Made%20with-OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)
[![Deploy](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

---

## 🚀 Live Demo

👉 **Try it now:** [https://huggingface.co/spaces/arafmustavi/realtime-face-detection](https://huggingface.co/spaces/arafmustavi/realtime-face-detection)

Point your webcam at any face — the app draws bounding boxes in realtime, entirely in your browser session.

---

## 📌 Project Overview

This project demonstrates **classical computer vision** in a modern, production-ready deployment context. While deep learning models like YOLO dominate the current CV landscape, **Haar Cascade classifiers** remain incredibly relevant for:

- ⚡ **Ultra-low latency** detection (CPU-only, no GPU required)
- 📱 **Edge-friendly** deployment on constrained devices
- 🎓 **Educational value** — understanding CV fundamentals before diving into neural networks
- 🧩 **Composable pipelines** — often used as a fast pre-filter before heavier models

---

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Webcam Feed   │ ───▶ │  Gradio Frontend │ ───▶ │  OpenCV Haar     │
│   (Browser)     │      │  (HF Spaces)     │      │  Cascade Engine  │
└─────────────────┘      └──────────────────┘      └────────┬─────────┘
                                                            │
                                                            ▼
                                                  ┌──────────────────┐
                                                  │ Bounding Boxes + │
                                                  │ Annotated Output │
                                                  └──────────────────┘

                         ▲
                         │  CI/CD Sync
                         │
                ┌────────┴──────────┐
                │  GitHub Actions   │
                │  (auto-push to HF)│
                └───────────────────┘
```

---

## 🧠 How Haar Cascades Work (In 60 Seconds)

Haar Cascade is a **machine learning-based** object detection algorithm proposed by Viola & Jones (2001).

1. **Haar-like features** — simple rectangular filters that measure contrast differences between adjacent regions (e.g., eyes are darker than cheeks).
2. **Integral image** — a clever data structure that computes feature values in constant time.
3. **AdaBoost cascade** — chains hundreds of weak classifiers into a strong one, rejecting non-face regions early for speed.
4. **Sliding window** — scans the image at multiple scales to catch faces of any size.

The result: **face detection in milliseconds**, even on CPU.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Computer Vision** | OpenCV (Haar Cascade classifiers) |
| **UI Framework** | Gradio 5.35.0 |
| **Deployment** | Hugging Face Spaces |
| **CI/CD** | GitHub Actions |
| **Language** | Python 3.10+ |
| **License** | Apache 2.0 |

---

## 📂 Repository Structure

```
haarcascade-detection-hfspace/
├── app.py                          # Gradio app + Haar Cascade inference
├── requirements.txt                # Python dependencies
├── haarcascade_frontalface.xml     # Pre-trained cascade classifier (optional)
├── .github/
│   └── workflows/
│       └── sync-to-hf.yml          # CI/CD pipeline
├── README.md                       # You are here
└── LICENSE
```

---

## 🖥️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/arafmustavi/haarcascade-detection-hfspace.git
cd haarcascade-detection-hfspace

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Gradio app
python app.py
```

The app will open at `http://localhost:7860`.

---

## 🔄 CI/CD: Sync GitHub → Hugging Face Spaces

This repo uses a **GitHub Actions workflow** to automatically push every commit on `main` to the connected Hugging Face Space. Here's how to set it up:

### Step 1: Generate a Hugging Face Access Token

1. Go to [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**, choose the **`write`** role, name it `HF_TOKEN`
3. Copy the token (you won't see it again!)

### Step 2: Add the Token to GitHub Secrets

1. In your GitHub repo → **Settings → Secrets and variables → Actions**
2. Click **"New repository secret"**
3. Name: `HF_TOKEN` | Value: *(paste the token)*
4. Save

### Step 3: Create the GitHub Actions Workflow

Create the file `.github/workflows/sync-to-hf.yml`:

```yaml
name: Sync to Hugging Face Hub

on:
  push:
    branches: [main]
  workflow_dispatch:    # Allows manual trigger from the Actions tab

jobs:
  sync-to-hub:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true

      - name: Push to Hugging Face Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git push https://arafmustavi:$HF_TOKEN@huggingface.co/spaces/arafmustavi/realtime-face-detection main --force
```

> 🔁 Replace `arafmustavi/realtime-face-detection` with your actual HF Space path if different.

### Step 4: Commit & Push

```bash
git add .github/workflows/sync-to-hf.yml
git commit -m "ci: add GitHub Actions workflow to sync with Hugging Face"
git push origin main
```

### Step 5: Verify

- Go to your GitHub repo → **Actions** tab → confirm the workflow ran successfully ✅
- Check your Hugging Face Space → the app should rebuild automatically 🚀

### 🧯 Troubleshooting

| Issue | Fix |
|-------|-----|
| `403 Forbidden` on push | Regenerate `HF_TOKEN` with **write** access |
| `Space not building` | Check `requirements.txt` — pin `gradio==5.35.0` |
| `LFS files missing` | Ensure `lfs: true` in checkout step |
| `Force push rejected` | Verify branch protection rules on HF Space |

---

## 🎓 What I Learned

- **Classical CV still has a place** — Haar Cascades outperform deep learning models on latency-critical edge scenarios.
- **Gradio + HF Spaces** is the fastest zero-DevOps path from prototype to live demo.
- **GitHub Actions as a CD pipeline** — reusable pattern for syncing any ML repo to HF Hub.
- **Separation of concerns** — GitHub for source-of-truth + collaboration; HF Spaces for hosting + community discovery.

---

## 🗺️ Roadmap

- [ ] Add **eye + smile detection** (multi-cascade pipeline)
- [ ] Benchmark **Haar Cascade vs. YOLOv8-face** on latency + accuracy
- [ ] Add **face blurring mode** for privacy demos
- [ ] Export to **ONNX** for edge deployment (Raspberry Pi / Jetson Nano)

---

## 🤝 Related Projects

Part of my broader **Computer Vision portfolio**:

- 🎯 **YOLO on HoloLens/Quest** — modern deep-learning-based detection on XR devices
- 🏭 **Supply Chain Vision AI** — enterprise CV applications at BAT

---

## 👤 Author

**Araf Mustavi**
Global IT Business Analyst — Supply Chain Technology, AI/ML & Digital Transformation
🌐 Portfolio: [arafmustavi.netlify.app](https://arafmustavi.netlify.app)
💼 LinkedIn: [linkedin.com/in/arafmustavi](https://linkedin.com/in/arafmustavi)

---

## 📄 License

Licensed under the [Apache License 2.0](LICENSE). Free for personal, academic, and commercial use with attribution.

---

<p align="center">
  Built with ❤️ using OpenCV, Gradio & Hugging Face Spaces
</p>
