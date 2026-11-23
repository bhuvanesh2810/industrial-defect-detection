
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import pandas as pd

MODEL_PATH = str(Path(__file__).parent / "best.pt")

st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

st.title("Industrial Surface Defect Detection")
st.write(
    "Upload a steel surface image to detect and classify "
    "multiple industrial defects."
)

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    with st.spinner("Detecting defects..."):

        result = model.predict(
            source=image,
            conf=0.10,
            iou=0.70,
            imgsz=640,
            max_det=300,
            device="cpu",
            verbose=False
        )[0]

    st.subheader("Detection Result")

    annotated_image = result.plot()[..., ::-1]

    st.image(
        annotated_image,
        caption="Detected Steel Surface Defects",
        use_container_width=True
    )

    total_defects = len(result.boxes)

    st.subheader("Detection Summary")

    st.metric(
        label="Total Defect Regions Detected",
        value=total_defects
    )

    if total_defects > 0:

        rows = []

        for box in result.boxes:

            class_id = int(box.cls.item())
            confidence = float(box.conf.item()) * 100

            rows.append({
                "Defect": model.names[class_id],
                "Confidence (%)": round(confidence, 2)
            })

        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Defect Class Summary")

        class_counts = (
            df["Defect"]
            .value_counts()
            .reset_index()
        )

        class_counts.columns = ["Defect Class", "Count"]

        st.dataframe(
            class_counts,
            use_container_width=True,
            hide_index=True
        )

        st.success(
            f"Successfully detected {total_defects} defect regions."
        )

    else:

        st.warning(
            "No defects detected. Try another steel surface image."
        )
