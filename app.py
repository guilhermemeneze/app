# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 13:08:19 2025

@author: Badger
"""

# app.py
import os
import threading
from datetime import datetime
from io import BytesIO

import av
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer, WebRtcMode, RTCConfiguration, VideoHTMLAttributes
)

st.set_page_config(page_title="Crop + Save + Count", layout="centered")
st.title("Crop + Save + Count")

# -------- Sidebar (shared controls) --------
st.sidebar.header("Circle & Save Settings")
filename_in = st.sidebar.text_input("Filename (without extension)", value="")
radius      = st.sidebar.slider("Circle radius (px)", 50, 2000, 400, step=5)
thickness   = st.sidebar.slider("Circle thickness", 1, 30, 6)
x_frac      = st.sidebar.slider("Center X (0–1)", 0.0, 1.0, 0.5, step=0.005)
y_frac      = st.sidebar.slider("Center Y (0–1)", 0.0, 1.0, 0.5, step=0.005)

# Crop cleanup
margin      = st.sidebar.number_input("Crop margin (px)", 0, 200, 10)
open_iters  = st.sidebar.number_input("Mask open iterations", 0, 5, 1)
blur_ksize  = st.sidebar.selectbox("Blur kernel (odd)", [0, 3, 5, 7], index=2)

# Saving
save_png    = st.sidebar.checkbox("Also save transparent PNG", True)
save_plot   = st.sidebar.checkbox("Save plot with counting", True)

st.caption(
    "Use live camera or upload from gallery. The circle guides cropping; "
    "we then count colonies and show a plot."
)

DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ---------- Helpers ----------
def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 1
    while True:
        cand = f"{root}_{i}{ext}"
        if not os.path.exists(cand):
            return cand
        i += 1

def apply_circle_mask_and_crop(rgb: np.ndarray, cx: int, cy: int, r: int,
                               margin: int, open_iters: int, blur_ksize: int):
    h, w = rgb.shape[:2]
    Y, X = np.ogrid[:h, :w]
    mask = (((X - cx) ** 2 + (Y - cy) ** 2) <= (r ** 2)).astype(np.uint8) * 255

    if open_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(open_iters))

    if blur_ksize and blur_ksize % 2 == 1 and blur_ksize >= 3:
        mask_blur = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    else:
        mask_blur = mask

    coords = cv2.findNonZero(mask_blur)
    if coords is None:
        return None, None, None

    x, y, w_box, h_box = cv2.boundingRect(coords)
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    x2 = min(x + w_box + 2 * margin, w)
    y2 = min(y + h_box + 2 * margin, h)

    masked_rgb  = cv2.bitwise_and(rgb, rgb, mask=mask)
    cropped_rgb = masked_rgb[y:y2, x:x2]
    cropped_mask = mask[y:y2, x:x2]
    return cropped_rgb, cropped_mask, (x, y, x2, y2)

def count_colonies(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(gray_enhanced, 150, 255, cv2.THRESH_BINARY_INV)

    h, w = binary.shape[:2]
    cx, cy = w // 2, h // 2
    radius_local = max(min(cx, cy) - 52, 5)
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), radius_local - 2, 255, thickness=-1)

    bw = cv2.bitwise_and(binary, binary, mask=circle_mask)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis_rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

    colony_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        per  = cv2.arcLength(cnt, True)
        if area > 0.1 and per > 0:
            circ = 4 * np.pi * area / (per ** 2)
            if circ > 0.10:
                colony_count += 1
                cv2.drawContours(vis_rgb, [cnt], -1, (255, 0, 0), 1)

    cv2.circle(vis_rgb, (cx, cy), radius_local, (200, 200, 200), 2)
    return colony_count, vis_rgb

def save_bytes_and_buttons(base: str, cropped_rgb: np.ndarray, cropped_mask: np.ndarray,
                           vis_rgb: np.ndarray, count: int):
    """Save to server Downloads and also expose mobile-friendly download buttons."""
    # 1) Filesystem saves (server side)
    jpg_path = unique_path(os.path.join(DOWNLOADS, f"{base}.jpg"))
    Image.fromarray(cropped_rgb).save(jpg_path, quality=95)

    png_path = None
    if save_png:
        rgba = np.dstack([cropped_rgb, cropped_mask])
        png_path = unique_path(os.path.join(DOWNLOADS, f"{base}.png"))
        Image.fromarray(rgba).save(png_path)

    processed_path = unique_path(os.path.join(DOWNLOADS, f"{base}_processed.png"))
    cv2.imwrite(processed_path, cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))

    # Plot
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(vis_rgb)
    plt.axis('off')
    plt.title(f"Detected Colonies: {count}")
    plt.tight_layout()

    plot_path = None
    if save_plot:
        plot_path = unique_path(os.path.join(DOWNLOADS, f"{base}_colonies.png"))
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")

    # 2) Download buttons (client/mobile friendly)
    # JPG bytes
    jpg_buf = BytesIO()
    Image.fromarray(cropped_rgb).save(jpg_buf, format="JPEG", quality=95)
    st.download_button("⬇️ Download cropped JPG", jpg_buf.getvalue(),
                       file_name=f"{base}.jpg", mime="image/jpeg")

    # PNG bytes
    if save_png:
        png_buf = BytesIO()
        rgba = np.dstack([cropped_rgb, cropped_mask])
        Image.fromarray(rgba).save(png_buf, format="PNG")
        st.download_button("⬇️ Download transparent PNG", png_buf.getvalue(),
                           file_name=f"{base}.png", mime="image/png")

    # Plot bytes
    if save_plot:
        plot_buf = BytesIO()
        fig.savefig(plot_buf, format="PNG", dpi=200, bbox_inches="tight")
        st.download_button("⬇️ Download plot (PNG)", plot_buf.getvalue(),
                           file_name=f"{base}_colonies.png", mime="image/png")

    # Show in app
    st.image(cropped_rgb, caption="Cropped JPG (black outside circle)", use_column_width=True)
    st.pyplot(fig, use_container_width=True)
    st.metric("Total colonies detected", int(count))

# ---------- TABS: Live Camera | Upload from Gallery ----------
tab_cam, tab_gallery = st.tabs(["📷 Live camera", "🖼️ Upload from gallery"])

# ===== Tab 1: Live camera with overlay =====
with tab_cam:
    st.write("Use the live preview with a red circle. Rear camera is requested on phones.")

    class CircleOverlay:
        def __init__(self):
            self.radius = 400
            self.thickness = 6
            self.x_frac = 0.5
            self.y_frac = 0.5
            self.last_frame_rgb = None
            self.frame_lock = threading.Lock()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")
            h, w = img_bgr.shape[:2]
            cx = int(self.x_frac * w)
            cy = int(self.y_frac * h)
            cv2.circle(img_bgr, (cx, cy), int(self.radius), (0, 0, 255), int(self.thickness), cv2.LINE_AA)

            rgb = frame.to_ndarray(format="rgb24")
            with self.frame_lock:
                self.last_frame_rgb = rgb

            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    ctx = webrtc_streamer(
        key="camera_live_circle",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": "environment"},  # rear camera on phones
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
            },
            "audio": False,
        },
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False),
        video_processor_factory=CircleOverlay,
    )

    if ctx and ctx.video_processor:
        ctx.video_processor.radius    = radius
        ctx.video_processor.thickness = thickness
        ctx.video_processor.x_frac    = x_frac
        ctx.video_processor.y_frac    = y_frac

    def get_current_frame_rgb():
        # Try to pull a fresh frame from WebRTC
        try:
            if ctx and hasattr(ctx, "video_receiver") and ctx.video_receiver:
                av_frame = ctx.video_receiver.get_frame(timeout=1.0)
                if av_frame is not None:
                    return av_frame.to_ndarray(format="rgb24")
        except Exception:
            pass
        # Fallback: last stored frame
        if ctx and ctx.video_processor:
            with ctx.video_processor.frame_lock:
                if ctx.video_processor.last_frame_rgb is not None:
                    return ctx.video_processor.last_frame_rgb.copy()
        return None

    st.divider()
    if st.button("📸 Capture & Save (live camera)"):
        rgb = get_current_frame_rgb()
        if rgb is None:
            st.info("Waiting for camera. Please allow access and try again.")
        else:
            base = (filename_in or "").strip() or f"petri_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            H, W = rgb.shape[:2]
            cx = int(x_frac * W)
            cy = int(y_frac * H)
            r  = int(radius)

            cropped_rgb, cropped_mask, _ = apply_circle_mask_and_crop(
                rgb, cx, cy, r, margin=int(margin), open_iters=int(open_iters), blur_ksize=int(blur_ksize)
            )
            if cropped_rgb is None:
                st.error("Nothing inside the circle. Adjust the circle or recenter.")
            else:
                # Count + save + show + download buttons
                count, vis_rgb = count_colonies(cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))
                try:
                    save_bytes_and_buttons(base, cropped_rgb, cropped_mask, vis_rgb, count)
                except Exception as e:
                    st.warning(f"Could not write to {DOWNLOADS} ({e}). Using download buttons only.")
                    save_bytes_and_buttons(base, cropped_rgb, cropped_mask, vis_rgb, count)

# ===== Tab 2: Upload from gallery =====
with tab_gallery:
    st.write("Pick a photo from your phone’s gallery.")
    uploaded = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        rgb = np.array(image)
        H, W = rgb.shape[:2]
        cx = int(x_frac * W)
        cy = int(y_frac * H)
        r  = int(radius)

        # Preview with circle on the uploaded image
        preview = rgb.copy()
        cv2.circle(preview, (cx, cy), r, (255, 0, 0), int(thickness), cv2.LINE_AA)
        st.image(preview, caption="Preview with circle (uploaded image)", use_column_width=True)

        if st.button("✅ Process uploaded image"):
            cropped_rgb, cropped_mask, _ = apply_circle_mask_and_crop(
                rgb, cx, cy, r, margin=int(margin), open_iters=int(open_iters), blur_ksize=int(blur_ksize)
            )
            if cropped_rgb is None:
                st.error("Nothing inside the circle at this position. Adjust radius/center and try again.")
            else:
                base = (filename_in or os.path.splitext(uploaded.name)[0]).strip() or \
                       f"petri_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                count, vis_rgb = count_colonies(cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))
                try:
                    save_bytes_and_buttons(base, cropped_rgb, cropped_mask, vis_rgb, count)
                except Exception as e:
                    st.warning(f"Could not write to {DOWNLOADS} ({e}). Using download buttons only.")
                    save_bytes_and_buttons(base, cropped_rgb, cropped_mask, vis_rgb, count)

st.caption(
    "Tip for phones: camera access usually requires **HTTPS**. "
    "If you’re opening this from another device on your LAN, deploy with HTTPS (e.g., Streamlit Cloud) "
    "or use the **Upload from gallery** tab which works everywhere."
)

