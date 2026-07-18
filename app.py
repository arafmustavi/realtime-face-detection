"""
Realtime Face Detection — Haar Cascade × Hugging Face Spaces
------------------------------------------------------------
A low-FPS streaming face detection app using OpenCV Haar Cascade
classifiers, served via Gradio on Hugging Face Spaces.

Author : Araf Mustavi
License: Apache 2.0
"""

import cv2
import numpy as np
import gradio as gr
from PIL import Image

# ---------------------------------------------------------------------
# 1. Load the Haar Cascade classifier
# ---------------------------------------------------------------------
# OpenCV ships pre-trained Haar cascades in `cv2.data.haarcascades`.
# Using the bundled path means we DON'T need to commit the XML file
# to the repo — perfect for Hugging Face Spaces.
# ---------------------------------------------------------------------

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE_PATH)

if face_cascade.empty():
    raise IOError(f"❌ Failed to load face cascade from {FACE_CASCADE_PATH}")
print("✅ Haar Cascade classifiers loaded successfully.")


# ---------------------------------------------------------------------
# 2. Face Detection Function
# ---------------------------------------------------------------------
def detect_faces(
    image: Image.Image,
    scale_factor: float = 1.2,
    min_neighbors: int = 5,
    detect_eyes: bool = False,
):
    """
    Detects faces (and optionally eyes) in a webcam frame using
    Haar Cascade classifiers, then draws bounding boxes.

    Parameters
    ----------
    image        : PIL.Image from the Gradio webcam stream
    scale_factor : How much the image size is reduced at each scale
    min_neighbors: How many neighbors each candidate rectangle should have
    detect_eyes  : Whether to also draw eye bounding boxes

    Returns
    -------
    Annotated PIL.Image with bounding boxes.
    """
    if image is None:
        return None

    # PIL -> NumPy (RGB)
    frame = np.array(image)

    # Downscale for faster inference on HF Spaces free CPU tier.
    # Keeps latency low and helps enforce the low-FPS target.
    h, w = frame.shape[:2]
    max_width = 480
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))

    # Convert to grayscale (Haar Cascade requirement)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # ---- Face detection ----
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(40, 40),
    )

    # Draw green bounding boxes on detected faces
    for (x, y, fw, fh) in faces:
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Face",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # ---- Optional eye detection inside each face ROI ----
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

    # Overlay a small HUD with the detection count
    cv2.putText(
        frame,
        f"Faces detected: {len(faces)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return Image.fromarray(frame)


# ---------------------------------------------------------------------
# 3. Gradio Interface (Low-FPS streaming)
# ---------------------------------------------------------------------
# `stream_every=0.5` throttles the pipeline to ~2 FPS, which is
# gentle on the Hugging Face free CPU tier and prevents queue buildup.
# Increase to 1.0 for ~1 FPS, decrease to 0.25 for ~4 FPS.
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
        # ---- Left column: webcam input + controls ----
        with gr.Column(scale=1):
            webcam_input = gr.Image(
                type="pil",
                label="📷 Your Webcam Feed",
                sources=["webcam"],
                streaming=True,
                mirror_webcam=True,
            )
            scale_factor_slider = gr.Slider(
                minimum=1.05,
                maximum=1.5,
                value=1.2,
                step=0.05,
                label="Scale Factor (detection sensitivity)",
            )
            min_neighbors_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Min Neighbors (detection quality)",
            )
            detect_eyes_checkbox = gr.Checkbox(
                value=False,
                label="👁️ Also detect eyes (slower)",
            )

        # ---- Right column: annotated output ----
        with gr.Column(scale=1):
            output_image = gr.Image(
                type="pil",
                label="🟩 Detection Output",
                streaming=True,
            )

    # ---- Wire the streaming pipeline (low FPS) ----
    webcam_input.stream(
        fn=detect_faces,
        inputs=[
            webcam_input,
            scale_factor_slider,
            min_neighbors_slider,
            detect_eyes_checkbox,
        ],
        outputs=output_image,
        stream_every=LOW_FPS_INTERVAL,   # 🔑 throttles FPS
        show_progress="hidden",
        concurrency_limit=1,             # avoid queue pile-up on HF free tier
    )

    gr.Markdown(
        """
        ---
        **Built with** OpenCV · Gradio · Hugging Face Spaces  
        **Author:** [Araf Mustavi](https://arafmustavi.netlify.app)
        """
    )

# ---------------------------------------------------------------------
# 4. Launch (HF Spaces auto-detects `demo`)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    demo.queue(max_size=5).launch()
