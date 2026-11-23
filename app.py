
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd

MODEL_PATH = str(Path(__file__).parent / "best.pt")

st.set_page_config(page_title="Industrial Defect Detection", page_icon="🔍", layout="wide")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.title("Industrial Surface Defect Detection")
st.write("Upload a steel surface image to detect and classify defects.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Detecting defects..."):
        result = model.predict(image, conf=0.25, device="cpu", verbose=False)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        annotated = result.plot()[..., ::-1]
        st.image(annotated, use_container_width=True)

    st.subheader("Detection Summary")
    st.metric("Defect regions detected", len(result.boxes))

    if len(result.boxes):
        rows = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            rows.append({
                "Defect": model.names[class_id],
                "Confidence": round(float(box.conf.item()) * 100, 2)
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No defects detected at the selected confidence threshold.")
