import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Photo Editor", layout="wide")

st.title("📸 AI Photo Editor")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.subheader("Original Image")
    st.image(img, use_column_width=True)

    st.sidebar.header("Adjustments")

    scale = st.sidebar.slider("Resize (%)", 10, 200, 100)
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    resized = cv2.resize(img, (width, height))

    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", 1.0, 3.0, 1.0)

    adjusted = cv2.convertScaleAbs(resized, alpha=contrast, beta=brightness)

    st.sidebar.header("Filters")

    grayscale = st.sidebar.checkbox("Grayscale")
    blur = st.sidebar.checkbox("Blur")
    sharpen = st.sidebar.checkbox("Sharpen")
    warm = st.sidebar.checkbox("Warm Filter")
    portrait = st.sidebar.checkbox("Portrait Blur")

    output = adjusted.copy()

    if grayscale:
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    if blur:
        output = cv2.GaussianBlur(output, (15, 15), 0)

    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        output = cv2.filter2D(output, -1, kernel)

    if warm and len(output.shape) == 3:  # only for color images
        increase = np.array([10, 10, 30])
        output = cv2.add(output, increase)

    if portrait:
        mask = np.zeros(output.shape[:2], np.uint8)
        h, w = mask.shape
        cv2.circle(mask, (w // 2, h // 2), min(w, h) // 3, 255, -1)

        blurred = cv2.GaussianBlur(output, (25, 25), 0)

        if len(output.shape) == 2:
            # grayscale case
            fg = cv2.bitwise_and(output, output, mask=mask)
            bg = cv2.bitwise_and(blurred, blurred, mask=cv2.bitwise_not(mask))
            output = cv2.add(fg, bg)
        else:
            # color case
            mask_3 = cv2.merge([mask, mask, mask])
            inv_mask = cv2.bitwise_not(mask_3)

            fg = cv2.bitwise_and(output, mask_3)
            bg = cv2.bitwise_and(blurred, inv_mask)

            output = cv2.add(fg, bg)

    st.subheader("Edited Image")
    st.image(output, use_column_width=True)

    result = Image.fromarray(output)
    buf = io.BytesIO()
    result.save(buf, format="PNG")

    st.download_button(
        label="Download Image",
        data=buf.getvalue(),
        file_name="edited_image.png",
        mime="image/png"
    )