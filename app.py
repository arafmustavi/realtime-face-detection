"""
Realtime Face Detection — Haar Cascade × Hugging Face Spaces
------------------------------------------------------------
A low-FPS streaming face detection app using OpenCV Haar Cascade
classifiers, served via Gradio on Hugging Face Spaces.

Author : Araf Mustavi
License: Apache 2.0
"""

import os
import sys
import urllib.request
import numpy as np
import gradio as gr
from PIL import Image

# ---------------------------------------------------------------------
# 1. Import OpenCV with diagnostics
# ---------------------------------------------------------------------
try:
    import cv2
    print(f"✅ OpenCV imported successfully — version: {cv2.__version__}")
    print(f"   cv2 module path: {cv2.__file__}")
    print(f"   Has CascadeClassifier: {hasattr(cv2, 'CascadeClassifier')}")
except ImportError as e:
    print(f"❌ Failed to import cv2: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------
# 2. Locate / download the Haar Cascade XML files
# ---------------------------------------------------------------------
# Strategy:
#   1. Try cv2.data.haarcascades (bundled with OpenCV)
#   2. Fall back to local ./haarcascades/ directory
#   3. Fall back to downloading from OpenCV's official GitHub repo
# ---------------------------------------------------------------------

HAARCASCADE_URLS = {
    "haarcascade_frontalface_default.xml":
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
    "haarcascade_eye.xml":
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_eye.xml",
}

LOCAL_CASCADE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascades")
os.makedirs(LOCAL_CASCADE_DIR, exist_ok=True)


def resolve_cascade(filename: str) -> str:
    """Locate a Haar Cascade XML file, downloading it if necessary."""
    # 1️⃣ Try OpenCV's bundled path
    try:
        bundled = os.path.join(cv2.data.haarcascades, filename)
        if os.path.exists(bundled):
            print(f"✅ Using bundled cascade: {bundled}")
            return bundled
    except AttributeError:
        print("⚠️  cv2.data.haarcascades not available — using fallback.")

    # 2️⃣ Try local ./haarcascades/ folder
    local = os.path.join(LOCAL_CASCADE_DIR, filename)
    if os.path.exists(local):
        print(f"✅ Using local cascade: {local}")
        return local

    # 3️⃣ Download from OpenCV's official GitHub
    url = HAARCASCADE_URLS.get(filename)
    if url:
        print(f"⬇️  Downloading {filename} from GitHub…")
        urllib.request.urlretrieve(url, local)
        print(f"✅ Downloaded to: {local}")
        return local

    raise FileNotFoundError(f"Cannot locate cascade file: {filename}")


FACE_CASCADE_PATH = resolve_cascade("haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH  = resolve_cascade("haarcascade_eye.xml")

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE_PATH)

if face_cascade.empty():
    raise IOError(f"❌ Failed to load face cascade from {FACE_CASCADE_PATH}")
print("✅ Haar Cascade classifiers loaded successfully.")


# ---------------------------------------------------------------------
# 3. Face Detection Function
# ---------------------------------------------------------------------
def detect_faces(
    image: Image.Image,
    scale_factor: float = 1.2,
    min_neighbors: int = 5,
    detect_eyes: bool = False,
):
    """Detect faces (and optionally eyes) in a webcam frame."""
    if image is None:
        return None

    frame = np.array(image)

    # Downscale for faster inference on HF free CPU tier
    h, w = frame.shape[:2]
    max_width = 480
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(40, 40),
    )

    for (x, y, fw, fh) in faces:
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        cv2.putText(
            frame, "Face", (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )

        if detect_eyes:
            roi_gray = gray[y:y + fh, x:x + fw]
            roi_color = frame[y:y + fh, x:x + fw]
            eyes = eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15)
            )
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    roi_color, (ex, ey), (ex + ew, ey + eh), (255, 200, 0), 1
                )

    cv2.putText(
        frame, f"Faces detected: {len(faces)}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 2, cv2.LINE_AA,
    )

    return Image.fromarray(frame)


# ---------------------------------------------------------------------
# 4. Gradio Interface (Low-FPS streaming)
# ---------------------------------------------------------------------
LOW_FPS_INTERVAL = 0.5   # seconds between frames  →  ~2 FPS

with gr.Blocks(title="Realtime Face Detection — Haar Cascade") as demo:
    gr.Markdown(
        """
        # 🎯 Realtime Face Detection (Haar Cascade)
        A lightweight **classical computer vision** demo running on
        **Hugging Face Spaces** via Gradio.

        > ⚙️ Streaming is throttled to **~2 FPS** to stay within the free CPU tier.
        > Allow webcam access when prompted.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            webcam_input = gr.Image(
                type="pil",
                label="📷 Your Webcam Feed",
                sources=["webcam"],
                streaming=True,
                mirror_webcam=True,
            )
            scale_factor_slider = gr.Slider(
                minimum=1.05, maximum=1.5, value=1.2, step=0.05,
                label="Scale Factor (detection sensitivity)",
            )
            min_neighbors_slider = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Min Neighbors (detection quality)",
            )
            detect_eyes_checkbox = gr.Checkbox(
                value=False, label="👁️ Also detect eyes (slower)",
            )

        with gr.Column(scale=1):
            output_image = gr.Image(
                type="pil",
                label="🟩 Detection Output",
                streaming=True,
            )

    webcam_input.stream(
        fn=detect_faces,
        inputs=[
            webcam_input,
            scale_factor_slider,
            min_neighbors_slider,
            detect_eyes_checkbox,
        ],
        outputs=output_image,
        stream_every=LOW_FPS_INTERVAL,
        show_progress="hidden",
        concurrency_limit=1,
    )

    gr.Markdown(
        """
        ---
        **Built with** OpenCV · Gradio · Hugging Face Spaces  
        **Author:** [Araf Mustavi](https://arafmustavi.netlify.app)
        """
    )

# ---------------------------------------------------------------------
# 5. Launch
# ---------------------------------------------------------------------
if __name__ == "__main__":
    demo.queue(max_size=5).launch()
