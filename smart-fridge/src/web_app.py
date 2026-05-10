"""
Streamlit web UI for Smart Fridge detection, inventory, and calorie estimates.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from calories import estimate_detections_calories
from detect import MODEL_PATH, run_detection
from inventory import build_inventory, get_grouped_counts
from report import draw_boxes


st.set_page_config(page_title="Smart Fridge", layout="wide")

st.title("Smart Fridge Inventory Detection")
st.caption(f"Model: `{MODEL_PATH}`")

uploaded_file = st.file_uploader("Upload a fridge image", type=["jpg", "jpeg", "png", "webp"])
confidence = st.slider("Confidence threshold", min_value=0.05, max_value=0.90, value=0.25, step=0.05)

if "detections" not in st.session_state:
    st.session_state.detections = []
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "annotated_path" not in st.session_state:
    st.session_state.annotated_path = None

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_image:
        temp_image.write(uploaded_file.getbuffer())
        image_path = temp_image.name

    st.image(image_path, caption="Uploaded image", use_container_width=True)

    if st.button("Run detection", type="primary"):
        with st.spinner("Detecting fridge items..."):
            detections = run_detection(image_path, conf=confidence)
            annotated_path = draw_boxes(image_path, detections, tempfile.gettempdir()) if detections else None

        st.session_state.detections = detections
        st.session_state.image_path = image_path
        st.session_state.annotated_path = annotated_path

if st.session_state.detections:
    detections = st.session_state.detections
    inventory = build_inventory(detections)
    grouped_counts = get_grouped_counts(inventory)

    st.subheader("Detection Results")
    if st.session_state.annotated_path:
        st.image(st.session_state.annotated_path, caption="Annotated image", use_container_width=True)

    st.metric("Total items detected", inventory["total"])

    count_rows = []
    for group, items in grouped_counts.items():
        for class_name, count in items.items():
            if count > 0:
                count_rows.append({"group": group, "class": class_name, "count": count})

    if count_rows:
        st.dataframe(pd.DataFrame(count_rows), use_container_width=True, hide_index=True)

    if st.button("Calculate calories"):
        calorie_report = estimate_detections_calories(detections)
        st.subheader("Calorie Estimate")
        st.metric("Total estimated calories", f"{calorie_report['total_calories']} kcal")
        st.dataframe(
            pd.DataFrame(calorie_report["items"]).drop(columns=["bbox"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Calories are approximate and based on detected class plus bounding-box size.")
elif uploaded_file is not None:
    st.info("Run detection first, then calculate calories.")
