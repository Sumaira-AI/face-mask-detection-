"""Face Mask Detection - Computer Vision Lab Assignment 2
Run:  streamlit run app.py
"""
import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import processing as P
import detector as D

st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="wide")

MODEL_PATH = os.path.join("model", "best.pt")


@st.cache_resource
def get_model(path):
    return D.load_model(path)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("😷 Face Mask Detection App")
file = st.sidebar.file_uploader("1. Upload image", type=["jpg", "jpeg", "png"])

st.sidebar.header("2. Processing (applied in order)")
ops = []  # list of (name, function)

with st.sidebar.expander("Zoom / Resize"):
    if st.checkbox("Resize"):
        s = st.slider("Scale", 0.2, 2.0, 1.0, 0.1)
        ops.append(("Resize", lambda im, s=s: P.resize_image(im, s)))
    if st.checkbox("Zoom (centre)"):
        z = st.slider("Zoom factor", 1.0, 4.0, 1.5, 0.1)
        ops.append(("Zoom", lambda im, z=z: P.zoom_image(im, z)))

with st.sidebar.expander("Blur / Noise removal"):
    if st.checkbox("Gaussian blur"):
        k = st.slider("Gaussian kernel", 3, 31, 5, 2)
        ops.append(("Gaussian blur", lambda im, k=k: P.gaussian_blur(im, k)))
    if st.checkbox("Average blur"):
        k2 = st.slider("Average kernel", 3, 31, 5, 2)
        ops.append(("Average blur", lambda im, k=k2: P.average_blur(im, k)))
    if st.checkbox("Median blur"):
        k3 = st.slider("Median kernel", 3, 31, 5, 2)
        ops.append(("Median blur", lambda im, k=k3: P.median_blur(im, k)))
    if st.checkbox("Bilateral filter"):
        ops.append(("Bilateral", P.bilateral_filter))

with st.sidebar.expander("Contrast / Histogram"):
    if st.checkbox("Histogram equalization"):
        ops.append(("Hist. equalization", P.hist_equalization))
    if st.checkbox("CLAHE"):
        cl = st.slider("Clip limit", 1.0, 8.0, 2.0, 0.5)
        ops.append(("CLAHE", lambda im, c=cl: P.clahe(im, c)))
    if st.checkbox("Brightness / Contrast"):
        a = st.slider("Contrast (alpha)", 0.5, 3.0, 1.0, 0.1)
        b = st.slider("Brightness (beta)", -100, 100, 0, 5)
        ops.append(("Brightness/Contrast", lambda im, a=a, b=b: P.brightness_contrast(im, a, b)))
    if st.checkbox("Gamma correction"):
        g = st.slider("Gamma", 0.2, 3.0, 1.0, 0.1)
        ops.append(("Gamma", lambda im, g=g: P.gamma_correction(im, g)))

with st.sidebar.expander("Edge detection / Filters"):
    if st.checkbox("Grayscale"):
        ops.append(("Grayscale", P.to_grayscale))
    if st.checkbox("Sharpen"):
        am = st.slider("Sharpen amount", 0.5, 3.0, 1.0, 0.1)
        ops.append(("Sharpen", lambda im, a=am: P.sharpen(im, a)))
    if st.checkbox("Canny edges"):
        lo = st.slider("Canny low", 0, 255, 100)
        hi = st.slider("Canny high", 0, 255, 200)
        ops.append(("Canny", lambda im, l=lo, h=hi: P.canny_edges(im, l, h)))
    if st.checkbox("Sobel edges"):
        ops.append(("Sobel", P.sobel_edges))
    if st.checkbox("Laplacian edges"):
        ops.append(("Laplacian", P.laplacian_edges))
    if st.checkbox("Negative"):
        ops.append(("Negative", P.negative))

st.sidebar.header("3. Detection")
conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
source = st.sidebar.radio("Run detection on", ["Original image", "Processed image"])
run = st.sidebar.button("🔍 Detect Mask", type="primary", use_container_width=True)

# --------------------------------------------------------------------- main
st.title("😷 Face Mask Detection & Image Processing")
st.caption("Model: YOLO11n (trained on Kaggle Face Mask Detection dataset) · 3 classes")

if file is None:
    st.info("⬅️ Upload an image from the sidebar to begin.")
    st.stop()

original = np.array(ImageOps.exif_transpose(Image.open(file)).convert("RGB"))

processed = original.copy()
for name, fn in ops:
    processed = fn(processed)

c1, c2 = st.columns(2)
c1.subheader("Original")
c1.image(original, use_container_width=True)
c1.caption(f"Size: {original.shape[1]}x{original.shape[0]}")
c2.subheader("Processed")
c2.image(processed, use_container_width=True)
c2.caption("Steps: " + (" → ".join(n for n, _ in ops) if ops else "none (same as original)"))

tab_h, tab_d = st.tabs(["📊 Histograms", "🎯 Detection result"])

with tab_h:
    h1, h2 = st.columns(2)
    h1.pyplot(P.histogram_figure(original, "Original"))
    h2.pyplot(P.histogram_figure(processed, "Processed"))

with tab_d:
    if not run:
        st.info("Press **Detect Mask** in the sidebar.")
    elif not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at `{MODEL_PATH}`. Put your trained `best.pt` there.")
    else:
        model = get_model(MODEL_PATH)
        target = original if source == "Original image" else processed
        out, dets, counts = D.detect(model, target, conf)
        st.image(out, caption=f"Detection on: {source}", use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("🟢 With Mask", counts["With Mask"])
        m2.metric("🟡 Mask Worn Incorrectly", counts["Mask Worn Incorrectly"])
        m3.metric("🔴 Without Mask", counts["Without Mask"])
        if dets:
            st.dataframe(dets, use_container_width=True)
        else:
            st.warning("No faces detected. Try lowering the confidence threshold.")
