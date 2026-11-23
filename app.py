
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from io import BytesIO
import pandas as pd

MODEL_PATH = Path(__file__).parent / "best.pt"

st.set_page_config(
    page_title="STEELVISION AI | Inspection",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5edf7;
}
[data-testid="stSidebar"] {
    background: #111b2d;
    border-right: 1px solid #25344a;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
.hero {
    background: linear-gradient(120deg, #14283b, #102b35);
    border: 1px solid #245166;
    padding: 27px;
    border-radius: 16px;
    margin-bottom: 24px;
}
.hero h1 {
    color: #f1f7ff;
    font-size: 32px;
    margin-bottom: 5px;
}
.hero p {
    color: #9db5c9;
    margin-bottom: 0;
}
.section-title {
    color: #e6f2ff;
    font-size: 21px;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 12px;
}
.metric-card {
    background: #142136;
    border: 1px solid #293d56;
    border-radius: 13px;
    padding: 17px;
    min-height: 112px;
}
.metric-label {
    color: #93a9c1;
    font-size: 13px;
}
.metric-value {
    color: #4dd8df;
    font-size: 29px;
    font-weight: 750;
    margin-top: 9px;
}
.metric-note {
    color: #8298ae;
    font-size: 12px;
}
.panel {
    background: #111d30;
    border: 1px solid #293d56;
    border-radius: 14px;
    padding: 17px;
}
.small-note {
    color: #91a8bf;
    font-size: 13px;
}
div.stButton > button, div.stDownloadButton > button {
    border-radius: 9px;
    border: 1px solid #2a8495;
    background: #123c4b;
    color: white;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    background: #1b6070;
    color: white;
    border-color: #4dd8df;
}
[data-testid="stFileUploader"] {
    background: #111d30;
    border: 1px dashed #34748a;
    border-radius: 12px;
    padding: 12px;
}
h2, h3 {
    color: #e6f2ff;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )
    return YOLO(str(MODEL_PATH))


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the detection model: {error}")
    st.stop()


# Sidebar
st.sidebar.markdown("## 🔍 STEELVISION AI")
st.sidebar.caption("Industrial Inspection Console")
st.sidebar.divider()

st.sidebar.markdown("### Inspection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.01,
    max_value=0.90,
    value=0.10,
    step=0.01,
    help="Lower values show more possible detections, including weaker predictions."
)

iou = st.sidebar.slider(
    "NMS IoU Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.70,
    step=0.05
)

image_size = st.sidebar.select_slider(
    "Inference Image Size",
    options=[416, 512, 640, 768, 960],
    value=640
)

max_detections = st.sidebar.select_slider(
    "Maximum Detections",
    options=[50, 100, 200, 300, 500],
    value=300
)

st.sidebar.divider()
st.sidebar.markdown("### Supported Defect Classes")

class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

for name in class_names:
    st.sidebar.markdown(f"• `{name}`")

st.sidebar.divider()
st.sidebar.caption("Model: Custom-trained YOLO11n")
st.sidebar.caption("Inference: CPU")


# Main header
st.markdown("""
<div class="hero">
    <h1>STEELVISION AI</h1>
    <p>Intelligent Steel Surface Inspection & Defect Classification</p>
    <p>Automated visual inspection powered by deep learning and YOLO.</p>
</div>
""", unsafe_allow_html=True)


# Upload section
st.markdown('<div class="section-title">01 / Image Inspection</div>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG or PNG image for defect inspection."
)

if uploaded_file is None:
    st.markdown("""
    <div class="panel">
        <h3>Ready for Inspection</h3>
        <p class="small-note">
        Upload a steel surface image to start automated defect detection.
        The system will identify possible defect regions, classify them
        and generate an inspection summary.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Inspection workflow")
    c1, c2, c3 = st.columns(3)

    c1.info("**01 — Upload**\n\nSelect a steel surface image.")
    c2.info("**02 — Detect**\n\nYOLO identifies potential defect regions.")
    c3.info("**03 — Analyze**\n\nReview detections and export results.")

    st.stop()


# Load image
try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception:
    st.error("The uploaded file could not be opened as an image.")
    st.stop()

width, height = image.size

st.markdown(
    f'<p class="small-note">Image: {uploaded_file.name} | '
    f'Resolution: {width} × {height} pixels</p>',
    unsafe_allow_html=True
)


# Prediction
with st.spinner("Analyzing steel surface and detecting defects..."):
    try:
        result = model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            max_det=max_detections,
            device="cpu",
            verbose=False
        )[0]
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()


# Process predictions
rows = []

for box in result.boxes:
    class_id = int(box.cls.item())
    score = float(box.conf.item())
    xyxy = box.xyxy[0].tolist()

    rows.append({
        "Defect Class": model.names[class_id],
        "Confidence (%)": round(score * 100, 2),
        "X1": round(xyxy[0], 1),
        "Y1": round(xyxy[1], 1),
        "X2": round(xyxy[2], 1),
        "Y2": round(xyxy[3], 1)
    })

df = pd.DataFrame(rows)

total = len(df)
unique_classes = df["Defect Class"].nunique() if total else 0
average_confidence = (
    df["Confidence (%)"].mean() if total else 0
)

annotated_array = result.plot()[..., ::-1]
annotated_image = Image.fromarray(annotated_array)


# Metrics
st.markdown(
    '<div class="section-title">02 / Inspection Overview</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

metrics = [
    (m1, "TOTAL DEFECTS", total, "Detected regions"),
    (m2, "DEFECT CLASSES", unique_classes, "Unique categories"),
    (m3, "AVG CONFIDENCE", f"{average_confidence:.1f}%", "Mean model confidence"),
    (m4, "IMAGE RESOLUTION", f"{width}×{height}", "Width × height")
]

for column, label, value, note in metrics:
    with column:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """, unsafe_allow_html=True)


# Image comparison
st.markdown(
    '<div class="section-title">03 / Visual Inspection</div>',
    unsafe_allow_html=True
)

original_col, detected_col = st.columns(2)

with original_col:
    st.markdown("#### Original Image")
    st.image(image, use_container_width=True)

with detected_col:
    st.markdown("#### AI Detection Output")
    st.image(annotated_image, use_container_width=True)


# Analytics
st.markdown(
    '<div class="section-title">04 / Defect Analytics</div>',
    unsafe_allow_html=True
)

if total > 0:
    chart_col, detail_col = st.columns([1, 1])

    class_summary = (
        df.groupby("Defect Class")
        .size()
        .reset_index(name="Detections")
        .sort_values("Detections", ascending=False)
    )

    with chart_col:
        st.markdown("#### Defects by Category")
        st.bar_chart(
            class_summary.set_index("Defect Class"),
            y="Detections"
        )

    with detail_col:
        st.markdown("#### Class-wise Summary")
        st.dataframe(
            class_summary,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("#### Individual Detection Confidence")
    st.bar_chart(
        df[["Confidence (%)"]]
    )

else:
    st.warning(
        "No defects were detected at the selected confidence threshold. "
        "Try adjusting the inspection settings or uploading another image."
    )


# Detailed report
st.markdown(
    '<div class="section-title">05 / Detection Report</div>',
    unsafe_allow_html=True
)

if total > 0:
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    buffer = BytesIO()
    annotated_image.save(buffer, format="PNG")

    st.download_button(
        label="⬇ Download Annotated Inspection Image",
        data=buffer.getvalue(),
        file_name="steel_defect_inspection.png",
        mime="image/png",
        use_container_width=True
    )

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Detection Report (CSV)",
        data=csv_data,
        file_name="steel_defect_report.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info("No detection report is available because no boxes were returned.")


# Footer
st.divider()

st.markdown("""
<div style="text-align:center; color:#8298ae; padding:12px;">
    <b>STEELVISION AI</b><br>
    Intelligent Industrial Surface Inspection<br>
    <small>YOLO11n | Deep Learning | Computer Vision</small>
</div>
""", unsafe_allow_html=True)
