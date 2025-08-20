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
st.title("Crop + Save + Count – Continuous")

# ---------- Session state ----------
st.session_state.setdefault("overlay_on", True)     # show/hide red circle
st.session_state.setdefault("started_once", False)  # to show tips only once

# ---------- Sidebar controls ----------
st.sidebar.header("Circle & Save Settings")
radius      = st.sidebar.slider("Circle radius (px)", 50, 2000, 400, step=5)
thickness   = st.sidebar.slider("Circle thickness", 1, 30, 6)
x_frac      = st.sidebar.slider("Center X (0–1)", 0.0, 1.0, 0.5, step=0.005)
y_frac      = st.sidebar.slider("Center Y (0–1)", 0.0, 1.0, 0.5, step=0.005)

margin      = st.sidebar.number_input("Crop margin (px)", 0, 200, 10)
open_iters  = st.sidebar.number_input("Mask open iterations", 0, 5, 1)
blur_ksize  = st.sidebar.selectbox("Mask blur kernel (odd)", [0, 3, 5, 7], index=2)

save_png    = st.sidebar.checkbox("Also save transparent PNG", True)
save_plot   = st.sidebar.checkbox("Save plot with counting", True)
low_res     = st.sidebar.checkbox("Low-res mode (640×480 @ 15fps)", False)

st.caption(
    "Tap **Start camera** once, align the plate, then press **Capture & Save** as many times as you need. "
    "Use **Adjust camera (show circle)** to show/hide the red circle any time. "
    "When finished, press **Finish** to release the camera."
)

DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")

# ---------- STUN + optional TURN (add in st.secrets['turn']) ----------
def build_rtc_config():
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]
    turn = st.secrets.get("turn") if hasattr(st, "secrets") else None
    if isinstance(turn, dict):
        urls = turn.get("urls"); user = turn.get("username"); cred = turn.get("credential")
        if urls and user and cred:
            if isinstance(urls, str):
                urls = [urls]
            ice_servers.append({"urls": urls, "username": user, "credential": cred})
    return RTCConfiguration({"iceServers": ice_servers})

rtc_config = build_rtc_config()

# ---------- Top controls ----------
col_name, col_h = st.columns([3, 1])
with col_name:
    base_name = st.text_input("Base image name (optional)", placeholder="e.g., plate_A01")
with col_h:
    preview_h = st.slider("Preview height (px)", 280, 900, 520, step=10)

# Keep the video inline and sized by slider
st.markdown(f"""
<style>
video[playsinline] {{
  height: {int(preview_h)}px !important;
  width: 100% !important;
  object-fit: contain !important;
  background: #000 !important;
  border-radius: 12px !important;
}}
video[playsinline]::-webkit-media-controls-fullscreen-button {{ display:none !important; }}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path); i = 1
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
    mask_blur = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0) if (blur_ksize and blur_ksize % 2 == 1 and blur_ksize >= 3) else mask
    coords = cv2.findNonZero(mask_blur)
    if coords is None:
        return None, None, None
    x, y, w_box, h_box = cv2.boundingRect(coords)
    x = max(x - margin, 0); y = max(y - margin, 0)
    x2 = min(x + w_box + 2 * margin, w); y2 = min(y + h_box + 2 * margin, h)
    masked_rgb  = cv2.bitwise_and(rgb, rgb, mask=mask)
    return masked_rgb[y:y2, x:x2], mask[y:y2, x:x2], (x, y, x2, y2)

def count_colonies(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(gray_enhanced, 150, 255, cv2.THRESH_BINARY_INV)
    h, w = binary.shape[:2]; cx, cy = w // 2, h // 2
    radius_local = max(min(cx, cy) - 52, 5)
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), radius_local - 2, 255, thickness=-1)
    bw = cv2.bitwise_and(binary, binary, mask=circle_mask)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis_rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt); per = cv2.arcLength(cnt, True)
        if area > 0.1 and per > 0:
            circ = 4 * np.pi * area / (per ** 2)
            if circ > 0.10:
                count += 1
                cv2.drawContours(vis_rgb, [cnt], -1, (255, 0, 0), 1)
    cv2.circle(vis_rgb, (cx, cy), radius_local, (200, 200, 200), 2)
    return count, vis_rgb

def save_all(base: str, cropped_rgb: np.ndarray, cropped_mask: np.ndarray, vis_rgb: np.ndarray, count: int):
    # server save (best-effort) + download buttons
    try:
        Image.fromarray(cropped_rgb).save(unique_path(os.path.join(DOWNLOADS, f"{base}.jpg")), quality=95)
        if save_png:
            rgba = np.dstack([cropped_rgb, cropped_mask])
            Image.fromarray(rgba).save(unique_path(os.path.join(DOWNLOADS, f"{base}.png")))
        cv2.imwrite(unique_path(os.path.join(DOWNLOADS, f"{base}_processed.png")),
                    cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
    except Exception:
        pass
    # plot for preview & optional download
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(vis_rgb); plt.axis('off'); plt.title(f"Detected Colonies: {count}"); plt.tight_layout()
    if save_plot:
        try:
            fig.savefig(unique_path(os.path.join(DOWNLOADS, f"{base}_colonies.png")), dpi=200, bbox_inches="tight")
        except Exception:
            pass
    # in-app quick downloads
    jpg_buf = BytesIO(); Image.fromarray(cropped_rgb).save(jpg_buf, format="JPEG", quality=95)
    st.download_button("⬇️ Download cropped JPG", jpg_buf.getvalue(), file_name=f"{base}.jpg", mime="image/jpeg")
    if save_png:
        png_buf = BytesIO(); Image.fromarray(np.dstack([cropped_rgb, cropped_mask])).save(png_buf, format="PNG")
        st.download_button("⬇️ Download transparent PNG", png_buf.getvalue(), file_name=f"{base}.png", mime="image/png")
    if save_plot:
        plot_buf = BytesIO(); fig.savefig(plot_buf, format="PNG", dpi=200, bbox_inches="tight")
        st.download_button("⬇️ Download plot (PNG)", plot_buf.getvalue(), file_name=f"{base}_colonies.png", mime="image/png")
    # show results
    st.image(cropped_rgb, caption="Cropped JPG (black outside circle)", use_column_width=True)
    st.pyplot(fig, use_container_width=True)
    st.metric("Total colonies detected", int(count))

# ---------- Always render the WebRTC component once (stable key) ----------
class CircleOverlay:
    """Draw the live circle only if overlay_on is True; always keep last RGB frame."""
    def __init__(self):
        self.radius = 400; self.thickness = 6
        self.x_frac = 0.5; self.y_frac = 0.5
        self.last_frame_rgb = None
        self.frame_lock = threading.Lock()
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        h, w = img_bgr.shape[:2]
        cx = int(self.x_frac * w); cy = int(self.y_frac * h)
        if st.session_state.get("overlay_on", True):
            cv2.circle(img_bgr, (cx, cy), int(self.radius), (0, 0, 255), int(self.thickness), cv2.LINE_AA)
        # keep last frame (RGB) for capture
        rgb = frame.to_ndarray(format="rgb24")
        with self.frame_lock:
            self.last_frame_rgb = rgb
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# Camera constraints
if low_res:
    video_constraints = {
        "facingMode": {"exact": "environment"},
        "advanced": [{"facingMode": "environment"}],
        "width": {"ideal": 640, "max": 640},
        "height": {"ideal": 480, "max": 480},
        "frameRate": {"ideal": 15, "max": 15},
    }
else:
    video_constraints = {
        "facingMode": {"exact": "environment"},
        "advanced": [{"facingMode": "environment"}],
        "width": {"ideal": 1280},
        "height": {"ideal": 720},
        "frameRate": {"ideal": 30, "max": 30},
    }

ctx = webrtc_streamer(
    key="webrtc_persistent",                # <- keep the same key to avoid re-permission
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": video_constraints, "audio": False},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, playsinline=True, muted=True),
    video_processor_factory=CircleOverlay,
    async_transform=True,
)

# keep overlay & circle params synced live
if ctx and ctx.video_processor:
    ctx.video_processor.radius    = radius
    ctx.video_processor.thickness = thickness
    ctx.video_processor.x_frac    = x_frac
    ctx.video_processor.y_frac    = y_frac

# Tip once
if ctx and not st.session_state.started_once:
    st.info("Tap **START** once to grant camera access. Then you can capture many images without extra prompts.")
    st.session_state.started_once = True

# ---------- Control row ----------
b1, b2, b3 = st.columns([1.4, 1.6, 1])
with b1:
    if st.button(("Hide circle" if st.session_state.overlay_on else "Adjust camera (show circle)")):
        st.session_state.overlay_on = not st.session_state.overlay_on
with b2:
    capture_clicked = st.button("📸 Capture & Save", use_container_width=True)
with b3:
    finish_clicked = st.button("⏹️ Finish")

def get_current_frame_rgb():
    try:
        if ctx and hasattr(ctx, "video_receiver") and ctx.video_receiver:
            av_frame = ctx.video_receiver.get_frame(timeout=1.0)
            if av_frame is not None:
                return av_frame.to_ndarray(format="rgb24")
    except Exception:
        pass
    if ctx and ctx.video_processor:
        with ctx.video_processor.frame_lock:
            if ctx.video_processor.last_frame_rgb is not None:
                return ctx.video_processor.last_frame_rgb.copy()
    return None

# ---------- Capture handler (camera stays ON) ----------
if capture_clicked:
    rgb = get_current_frame_rgb()
    if rgb is None:
        st.warning("No camera frames yet. Press **START** (browser prompt) and try again.")
    else:
        # name: use base_name + timestamp to avoid overwriting
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = (base_name.strip() if base_name else f"petri") + f"_{ts}"

        H, W = rgb.shape[:2]
        cx = int(x_frac * W); cy = int(y_frac * H); r = int(radius)
        cropped_rgb, cropped_mask, _ = apply_circle_mask_and_crop(
            rgb, cx, cy, r, margin=int(margin), open_iters=int(open_iters), blur_ksize=int(blur_ksize)
        )
        if cropped_rgb is None:
            st.error("Nothing inside the circle. Adjust the camera or radius and try again.")
        else:
            count, vis_rgb = count_colonies(cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))
            save_all(base, cropped_rgb, cropped_mask, vis_rgb, count)

# ---------- Finish (stop camera) ----------
if finish_clicked and ctx:
    try:
        # Try the official stop if available
        if hasattr(ctx, "stop") and callable(ctx.stop):
            ctx.stop()
    except Exception:
        pass
    st.success("Camera stopped. You can close the page safely.")
