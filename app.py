import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Watermelon Vine Coverage App", layout="wide")

st.title("Watermelon Vine Coverage and Petiole Sap Tracker")

st.sidebar.header("Image Settings")

uploaded_file = st.file_uploader(
    "Upload a drone image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.subheader("Original Drone Image")
    st.image(img, use_container_width=True)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_h = st.sidebar.slider("Lower green hue", 20, 90, 35)
    upper_h = st.sidebar.slider("Upper green hue", 40, 120, 85)
    lower_s = st.sidebar.slider("Minimum saturation", 0, 255, 40)
    lower_v = st.sidebar.slider("Minimum brightness", 0, 255, 40)

    lower_green = np.array([lower_h, lower_s, lower_v])
    upper_green = np.array([upper_h, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    green_pixels = np.count_nonzero(mask_clean)
    total_pixels = mask_clean.size
    vine_coverage = green_pixels / total_pixels * 100

    st.subheader("Predicted Vine Coverage")
    st.metric("Vine Coverage", f"{vine_coverage:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.image(mask_clean, caption="Detected Vine Mask", use_container_width=True)

    overlay = img.copy()
    overlay[mask_clean > 0] = [0, 255, 0]
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    with col2:
        st.image(blended, caption="Vine Coverage Overlay", use_container_width=True)

else:
    vine_coverage = None
    st.info("Upload a drone image to estimate vine coverage.")

st.header("Weekly Petiole Sap Measurements")

week = st.number_input("Week after transplanting", min_value=1, max_value=12, value=5)
no3 = st.number_input("Petiole sap NO₃-N ppm", min_value=0, value=600)
k = st.number_input("Petiole sap K ppm", min_value=0, value=3000)

if no3 < 400:
    no3_status = "Low"
elif no3 <= 900:
    no3_status = "Adequate"
else:
    no3_status = "High"

if k < 2500:
    k_status = "Low"
elif k <= 4500:
    k_status = "Adequate"
else:
    k_status = "High"

st.subheader("Crop Condition Summary")

st.write(f"NO₃-N status: **{no3_status}**")
st.write(f"K status: **{k_status}**")

if vine_coverage is not None:
    if vine_coverage < 50:
        canopy_status = "Low"
    elif vine_coverage < 75:
        canopy_status = "Moderate"
    else:
        canopy_status = "Good"

    st.write(f"Canopy status: **{canopy_status}**")

    results = pd.DataFrame([{
        "week_after_transplanting": week,
        "vine_coverage_percent": round(vine_coverage, 1),
        "no3_ppm": no3,
        "k_ppm": k,
        "no3_status": no3_status,
        "k_status": k_status,
        "canopy_status": canopy_status
    }])

    st.subheader("Results Table")
    st.dataframe(results)

    csv = results.to_csv(index=False)

    st.download_button(
        "Download Results as CSV",
        csv,
        "watermelon_vine_coverage_results.csv",
        "text/csv"
    )