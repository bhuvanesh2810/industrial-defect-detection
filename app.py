
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from io import BytesIO
from datetime import datetime
import pandas as pd
import zipfile
import time
import json

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "best.pt"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

st.set_page_config(
    page_title="STEELVISION AI | Industrial Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: #09111f;
    color: #e6edf7;
}
[data-testid="stSidebar"] {
    background: #101b2c;
    border-right: 1px solid #263b52;
}
.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #122b40, #123c48);
    border: 1px solid #286176;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 34px;
    color: #f3f8ff;
    margin: 0;
    letter-spacing: 1px;
}
.hero p {
    color: #a7c1d3;
    margin: 7px 0 0 0;
}
.section-title {
    color: #eaf3ff;
    font-size: 21px;
    font-weight: 700;
    margin: 25px 0 14px 0;
}
.metric-card {
    background: #142238;
    border: 1px solid #2a405b;
    border-radius: 13px;
    padding: 17px;
    min-height: 112px;
}
.metric-label {
    color: #9aafc6;
    font-size: 12px;
    font-weight: 600;
}
.metric-value {
    color: #54d8df;
    font-size: 29px;
    font-weight: 750;
    margin-top: 8px;
}
.metric-note {
    color: #8ca3b9;
    font-size: 11px;
    margin-top: 4px;
}
.panel {
    background: #111e31;
    border: 1px solid #293f59;
    border-radius: 14px;
    padding: 17px;
}
.info-text {
    color: #9aafc6;
    font-size: 13px;
}
.status {
    display: inline-block;
    background: #123e36;
    color: #70e5b1;
    border: 1px solid #26765c;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
}
div.stButton > button,
div.stDownloadButton > button {
    background: #124455;
    color: white;
    border: 1px solid #347b8d;
    border-radius: 9px;
    font-weight: 600;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: #1b6070;
    color: white;
    border-color: #54d8df;
}
[data-testid="stFileUploader"] {
    background: #111e31;
    border: 1px dashed #347b8d;
    border-radius: 12px;
    padding: 12px;
}
h2, h3 {
    color: #e6f3ff;
}
hr {
    border-color: #263b52;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

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
    st.error(f"Model loading failed: {error}")
    st.stop()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "inspection_results" not in st.session_state:
    st.session_state.inspection_results = []

if "inspection_timestamp" not in st.session_state:
    st.session_state.inspection_timestamp = None

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def create_detection_dataframe(result, model_names):
    rows = []

    for index, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls.item())
        confidence = float(box.conf.item()) * 100
        coordinates = box.xyxy[0].tolist()

        rows.append({
            "Detection ID": index,
            "Defect Class": model_names[class_id],
            "Confidence (%)": round(confidence, 2),
            "X1": round(coordinates[0], 1),
            "Y1": round(coordinates[1], 1),
            "X2": round(coordinates[2], 1),
            "Y2": round(coordinates[3], 1)
        })

    columns = [
        "Detection ID",
        "Defect Class",
        "Confidence (%)",
        "X1", "Y1", "X2", "Y2"
    ]

    return pd.DataFrame(rows, columns=columns)


def image_to_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_combined_dataframe(results):
    combined = []

    for item in results:
        if not item["dataframe"].empty:
            temp = item["dataframe"].copy()
            temp.insert(0, "Image", item["filename"])
            combined.append(temp)

    columns = [
        "Image",
        "Detection ID",
        "Defect Class",
        "Confidence (%)",
        "X1", "Y1", "X2", "Y2"
    ]

    if combined:
        return pd.concat(combined, ignore_index=True)

    return pd.DataFrame(columns=columns)


def create_report_zip(results):
    buffer = BytesIO()

    with zipfile.ZipFile(
        buffer, "w", zipfile.ZIP_DEFLATED
    ) as archive:

        for index, item in enumerate(results, start=1):
            prefix = f"inspection_{index}"

            archive.writestr(
                f"{prefix}/original.png",
                image_to_bytes(item["original"])
            )

            archive.writestr(
                f"{prefix}/annotated.png",
                image_to_bytes(item["annotated"])
            )

            archive.writestr(
                f"{prefix}/detections.csv",
                item["dataframe"].to_csv(index=False)
            )

            for crop_index, crop in enumerate(
                item["crops"], start=1
            ):
                archive.writestr(
                    f"{prefix}/crops/defect_{crop_index}.png",
                    image_to_bytes(crop["image"])
                )

        combined = create_combined_dataframe(results)

        archive.writestr(
            "complete_detection_report.csv",
            combined.to_csv(index=False)
        )

        summary_rows = []

        for item in results:
            summary_rows.append({
                "Image": item["filename"],
                "Defect Regions": len(item["dataframe"]),
                "Inference Time (s)": round(item["time"], 3),
                "Resolution": (
                    f'{item["width"]} × {item["height"]}'
                )
            })

        archive.writestr(
            "inspection_summary.csv",
            pd.DataFrame(summary_rows).to_csv(index=False)
        )

    buffer.seek(0)
    return buffer.getvalue()


def run_inspection(
    uploaded_files,
    confidence,
    iou,
    image_size,
    max_det
):
    results = []

    progress = st.progress(0)
    status = st.empty()

    for index, uploaded_file in enumerate(uploaded_files):

        status.write(
            f"Inspecting image {index + 1} of "
            f"{len(uploaded_files)}: {uploaded_file.name}"
        )

        try:
            image = Image.open(uploaded_file).convert("RGB")
            width, height = image.size

            start_time = time.perf_counter()

            prediction = model.predict(
                source=image,
                conf=confidence,
                iou=iou,
                imgsz=image_size,
                max_det=max_det,
                device="cpu",
                verbose=False
            )[0]

            elapsed = time.perf_counter() - start_time

            annotated = Image.fromarray(
                prediction.plot()[..., ::-1]
            )

            dataframe = create_detection_dataframe(
                prediction, model.names
            )

            crops = []

            for box in prediction.boxes:
                coords = box.xyxy[0].tolist()

                x1 = max(0, int(coords[0]))
                y1 = max(0, int(coords[1]))
                x2 = min(width, int(coords[2]))
                y2 = min(height, int(coords[3]))

                if x2 > x1 and y2 > y1:
                    crop = image.crop((x1, y1, x2, y2))

                    class_id = int(box.cls.item())
                    score = float(box.conf.item()) * 100

                    crops.append({
                        "image": crop,
                        "class": model.names[class_id],
                        "confidence": round(score, 2)
                    })

            results.append({
                "filename": uploaded_file.name,
                "original": image,
                "annotated": annotated,
                "dataframe": dataframe,
                "crops": crops,
                "time": elapsed,
                "width": width,
                "height": height
            })

        except Exception as error:
            st.error(
                f"Could not process {uploaded_file.name}: {error}"
            )

        progress.progress((index + 1) / len(uploaded_files))

    status.success(
        f"Inspection completed: {len(results)} of "
        f"{len(uploaded_files)} image(s) processed."
    )

    return results


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("## 🔍 STEELVISION AI")
st.sidebar.caption("INDUSTRIAL INTELLIGENCE PLATFORM")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Live Inspection",
        "Analytics",
        "Inspection Reports",
        "Model Evaluation",
        "About Model"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### MODEL INFORMATION")
st.sidebar.markdown("**Architecture:** YOLO11n")
st.sidebar.markdown("**Framework:** Ultralytics")
st.sidebar.markdown("**Inference:** CPU")
st.sidebar.markdown("**Classes:** 6")

st.sidebar.markdown(
    '<span class="status">● MODEL LOADED</span>',
    unsafe_allow_html=True
)

st.sidebar.divider()
st.sidebar.caption("STEELVISION AI | Deep Learning Inspection")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>STEELVISION AI</h1>
    <p>INTELLIGENT STEEL SURFACE INSPECTION SYSTEM</p>
    <p>
        Detect. Classify. Analyze. Improve industrial quality inspection
        through deep learning and computer vision.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# GLOBAL METRICS
# --------------------------------------------------

all_results = st.session_state.inspection_results

total_images = len(all_results)

total_defects = sum(
    len(item["dataframe"]) for item in all_results
)

images_with_detections = sum(
    1 for item in all_results
    if not item["dataframe"].empty
)

images_without_detections = (
    total_images - images_with_detections
)

all_data = create_combined_dataframe(all_results)

unique_classes = (
    all_data["Defect Class"].nunique()
    if not all_data.empty else 0
)

average_confidence = (
    all_data["Confidence (%)"].mean()
    if not all_data.empty else 0
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">01 / SYSTEM OVERVIEW</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (c1, "IMAGES INSPECTED", total_images),
        (c2, "DEFECT REGIONS", total_defects),
        (c3, "DEFECT CLASSES", unique_classes),
        (c4, "AVG CONFIDENCE", f"{average_confidence:.1f}%")
    ]

    for col, label, value in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">Current session</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">02 / INSPECTION WORKSPACE</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("""
        <div class="panel">
            <h3>AI Inspection Engine</h3>
            <p class="info-text">
            Analyze steel surface images using a custom-trained
            YOLO11n object detection model.
            </p>
            <p>✓ Multiple defect region detection</p>
            <p>✓ Six defect categories</p>
            <p>✓ Confidence-based predictions</p>
            <p>✓ Annotated inspection results</p>
            </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="panel">
            <h3>Supported Defect Categories</h3>
            <p>01 · Crazing</p>
            <p>02 · Inclusion</p>
            <p>03 · Patches</p>
            <p>04 · Pitted Surface</p>
            <p>05 · Rolled-in Scale</p>
            <p>06 · Scratches</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">03 / RECENT INSPECTIONS</div>',
        unsafe_allow_html=True
    )

    if all_results:
        recent = []

        for item in all_results:
            recent.append({
                "Image": item["filename"],
                "Defect Regions": len(item["dataframe"]),
                "Inference Time (s)": round(item["time"], 3),
                "Resolution": (
                    f'{item["width"]} × {item["height"]}'
                )
            })

        st.dataframe(
            pd.DataFrame(recent),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "No inspections yet. Open Live Inspection to begin."
        )


# --------------------------------------------------
# LIVE INSPECTION
# --------------------------------------------------

elif page == "Live Inspection":

    st.markdown(
        '<div class="section-title">01 / IMAGE ACQUISITION</div>',
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Upload steel surface images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(
            f"{len(uploaded_files)} image(s) selected."
        )

    st.markdown(
        '<div class="section-title">02 / DETECTION PARAMETERS</div>',
        unsafe_allow_html=True
    )

    p1, p2 = st.columns(2)

    with p1:
        confidence = st.slider(
            "Confidence Threshold",
            0.01, 0.90, 0.10, 0.01,
            help="Lower values may show more detections and false positives."
        )

        image_size = st.select_slider(
            "Inference Image Size",
            options=[416, 512, 640, 768, 960],
            value=640
        )

    with p2:
        iou = st.slider(
            "NMS IoU Threshold",
            0.10, 0.95, 0.70, 0.05,
            help="Controls suppression of overlapping predicted boxes."
        )

        max_det = st.select_slider(
            "Maximum Detections Per Image",
            options=[50, 100, 200, 300, 500],
            value=300
        )

    st.markdown(
        '<div class="section-title">03 / EXECUTE INSPECTION</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "🔍 RUN AI INSPECTION",
        use_container_width=True,
        disabled=not bool(uploaded_files)
    ):
        new_results = run_inspection(
            uploaded_files,
            confidence,
            iou,
            image_size,
            max_det
        )

        st.session_state.inspection_results = new_results
        st.session_state.inspection_timestamp = (
            datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        )

        st.rerun()

    if all_results:

        st.markdown(
            '<div class="section-title">04 / BATCH INSPECTION SUMMARY</div>',
            unsafe_allow_html=True
        )

        s1, s2, s3, s4 = st.columns(4)

        s1.metric("Images Processed", total_images)
        s2.metric("Total Defect Regions", total_defects)
        s3.metric("Images With Defects", images_with_detections)
        s4.metric("Images Without Defects", images_without_detections)

        st.metric(
            "Average Detection Confidence",
            f"{average_confidence:.2f}%"
        )

        if not all_data.empty:
            st.markdown("#### OVERALL DEFECT CLASS COUNTS")

            class_counts = (
                all_data["Defect Class"]
                .value_counts()
                .reindex(CLASS_NAMES, fill_value=0)
                .rename_axis("Defect Class")
                .reset_index(name="Detections")
            )

            st.dataframe(
                class_counts,
                use_container_width=True,
                hide_index=True
            )

        st.markdown(
            '<div class="section-title">05 / ALL IMAGE RESULTS</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Each uploaded image is shown separately below, "
            "including all detections and localization results."
        )

        for index, item in enumerate(all_results, start=1):

            data = item["dataframe"]

            with st.expander(
                f"Image {index}: {item['filename']} — "
                f"{len(data)} defect region(s)",
                expanded=True
            ):

                original_col, result_col = st.columns(2)

                with original_col:
                    st.markdown("#### ORIGINAL IMAGE")
                    st.image(
                        item["original"],
                        use_container_width=True
                    )

                with result_col:
                    st.markdown("#### AI DETECTION OUTPUT")
                    st.image(
                        item["annotated"],
                        use_container_width=True
                    )

                k1, k2, k3 = st.columns(3)

                k1.metric("Defect Regions", len(data))

                k2.metric(
                    "Unique Classes",
                    data["Defect Class"].nunique()
                    if not data.empty else 0
                )

                k3.metric(
                    "Inference Time",
                    f'{item["time"]:.3f} s'
                )

                if not data.empty:

                    st.markdown("#### DETECTION DETAILS")

                    st.dataframe(
                        data,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.markdown("#### INDIVIDUAL DEFECT CROPS")

                    crop_cols = st.columns(3)

                    for crop_index, crop in enumerate(item["crops"]):
                        with crop_cols[crop_index % 3]:
                            st.image(
                                crop["image"],
                                caption=(
                                    f'{crop["class"]} | '
                                    f'{crop["confidence"]}% confidence'
                                ),
                                use_container_width=True
                            )

                else:
                    st.warning(
                        "No defect regions detected at the selected "
                        "confidence threshold."
                    )

        st.markdown(
            '<div class="section-title">06 / BATCH EXPORT</div>',
            unsafe_allow_html=True
        )

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            st.download_button(
                "⬇ DOWNLOAD COMBINED CSV",
                data=all_data.to_csv(index=False).encode("utf-8"),
                file_name="steelvision_all_detections.csv",
                mime="text/csv",
                use_container_width=True
            )

        with export_col2:
            st.download_button(
                "⬇ DOWNLOAD COMPLETE INSPECTION ZIP",
                data=create_report_zip(all_results),
                file_name="steelvision_inspection_reports.zip",
                mime="application/zip",
                use_container_width=True
            )


# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------

elif page == "Analytics":

    st.markdown(
        '<div class="section-title">01 / DETECTION ANALYTICS</div>',
        unsafe_allow_html=True
    )

    if all_data.empty:
        st.info(
            "Run an inspection first to generate analytics."
        )

    else:

        a1, a2, a3 = st.columns(3)

        a1.metric("Total Detections", len(all_data))

        a2.metric(
            "Classes Found",
            all_data["Defect Class"].nunique()
        )

        a3.metric(
            "Mean Confidence",
            f'{all_data["Confidence (%)"].mean():.2f}%'
        )

        st.markdown("### Defect Distribution")

        distribution = (
            all_data["Defect Class"]
            .value_counts()
            .reindex(CLASS_NAMES, fill_value=0)
            .rename_axis("Defect Class")
            .reset_index(name="Detections")
        )

        chart_col, table_col = st.columns(2)

        with chart_col:
            st.bar_chart(
                distribution.set_index("Defect Class")
            )

        with table_col:
            st.dataframe(
                distribution,
                use_container_width=True,
                hide_index=True
            )

        st.markdown("### Confidence Analysis")

        st.bar_chart(
            all_data["Confidence (%)"]
        )

        st.markdown("### Confidence Statistics")

        st.dataframe(
            all_data.groupby("Defect Class")[
                "Confidence (%)"
            ].agg(
                ["count", "mean", "min", "max"]
            ).round(2),
            use_container_width=True
        )

        st.caption(
            "These statistics describe model predictions, not "
            "ground-truth precision or accuracy."
        )


# --------------------------------------------------
# REPORTS
# --------------------------------------------------

elif page == "Inspection Reports":

    st.markdown(
        '<div class="section-title">01 / REPORT CENTER</div>',
        unsafe_allow_html=True
    )

    if not all_results:
        st.info(
            "No inspection reports are available. Run an inspection first."
        )

    else:

        st.write(
            f"Inspection session: "
            f"{st.session_state.inspection_timestamp or 'Available'}"
        )

        report_rows = []

        for item in all_results:
            report_rows.append({
                "Image": item["filename"],
                "Defect Regions": len(item["dataframe"]),
                "Inference Time (s)": round(item["time"], 3),
                "Resolution": (
                    f'{item["width"]} × {item["height"]}'
                )
            })

        report_df = pd.DataFrame(report_rows)

        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Export Inspection Results")

        st.download_button(
            "⬇ DOWNLOAD COMPLETE INSPECTION ZIP",
            data=create_report_zip(all_results),
            file_name="steelvision_inspection_reports.zip",
            mime="application/zip",
            use_container_width=True
        )

        st.download_button(
            "⬇ DOWNLOAD ALL DETECTIONS CSV",
            data=all_data.to_csv(index=False).encode("utf-8"),
            file_name="steelvision_all_detections.csv",
            mime="text/csv",
            use_container_width=True
        )


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":

    st.markdown(
        '<div class="section-title">01 / MODEL PERFORMANCE EVALUATION</div>',
        unsafe_allow_html=True
    )

    METRICS_PATH = Path(__file__).parent / "evaluation_metrics.json"
    CONFUSION_MATRIX_PATH = Path(__file__).parent / "confusion_matrix.png"

    if not METRICS_PATH.exists():
        st.warning(
            "evaluation_metrics.json was not found. Upload this file "
            "to the same GitHub repository folder as app.py and best.pt."
        )
    else:
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as metrics_file:
                evaluation = json.load(metrics_file)

            st.caption(
                f"Evaluation split: {evaluation.get('evaluation_split', 'Not specified')} "
                "| These are offline validation metrics, not live-session confidence scores."
            )

            st.markdown("### Overall Detection Metrics")

            precision = evaluation.get("precision")
            recall = evaluation.get("recall")
            f1_score = evaluation.get("f1_score")

            if f1_score is None and precision is not None and recall is not None:
                denominator = precision + recall
                f1_score = (
                    2 * precision * recall / denominator
                    if denominator else 0
                )

            m1, m2, m3, m4, m5 = st.columns(5)

            m1.metric("Precision", f"{precision * 100:.2f}%" if precision is not None else "N/A")
            m2.metric("Recall", f"{recall * 100:.2f}%" if recall is not None else "N/A")
            m3.metric("F1-score", f"{f1_score * 100:.2f}%" if f1_score is not None else "N/A")
            m4.metric("mAP@50", f"{evaluation['mAP50'] * 100:.2f}%" if evaluation.get("mAP50") is not None else "N/A")
            m5.metric("mAP@50–95", f"{evaluation['mAP50_95'] * 100:.2f}%" if evaluation.get("mAP50_95") is not None else "N/A")

            st.markdown("### Model and Runtime Information")

            info1, info2, info3 = st.columns(3)

            if MODEL_PATH.exists():
                model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
                info1.metric("Model Size", f"{model_size_mb:.2f} MB")
            else:
                info1.metric("Model Size", "N/A")

            if all_results:
                average_inference_ms = (
                    sum(item["time"] for item in all_results)
                    / len(all_results)
                ) * 1000
                info2.metric("Avg. Inference Time", f"{average_inference_ms:.2f} ms/image")
                info3.metric("Images Timed", len(all_results))
            else:
                info2.metric("Avg. Inference Time", "Run inspection first")
                info3.metric("Images Timed", "0")

            training_seconds = evaluation.get("training_time_seconds")
            if training_seconds is not None:
                st.caption(
                    f"Recorded training time: {training_seconds:.2f} seconds "
                    f"({training_seconds / 60:.2f} minutes). This is not inference time."
                )

            st.info(
                "Accuracy is not reported because the supplied YOLO evaluation "
                "results do not include a defined detection-level accuracy value. "
                "It is intentionally shown as unavailable rather than estimated."
            )

            st.markdown("### Per-Class Performance")

            per_class = evaluation.get("per_class", {})
            if per_class:
                class_rows = []
                for class_name, values in per_class.items():
                    p = values.get("precision")
                    r = values.get("recall")
                    class_f1 = (
                        2 * p * r / (p + r)
                        if p is not None and r is not None and (p + r) > 0
                        else None
                    )
                    class_rows.append({
                        "Defect Class": class_name.replace("_", " ").title(),
                        "Precision (%)": round(p * 100, 2) if p is not None else None,
                        "Recall (%)": round(r * 100, 2) if r is not None else None,
                        "F1-score (%)": round(class_f1 * 100, 2) if class_f1 is not None else None,
                        "mAP@50 (%)": round(values["mAP50"] * 100, 2) if values.get("mAP50") is not None else None,
                        "mAP@50–95 (%)": round(values["mAP50_95"] * 100, 2) if values.get("mAP50_95") is not None else None
                    })

                class_df = pd.DataFrame(class_rows)
                st.dataframe(class_df, use_container_width=True, hide_index=True)

                if "mAP@50 (%)" in class_df.columns:
                    st.markdown("#### Per-Class mAP@50")
                    st.bar_chart(class_df.set_index("Defect Class")[["mAP@50 (%)"]])

            st.markdown("### Confusion Matrix")

            if CONFUSION_MATRIX_PATH.exists():
                st.image(
                    str(CONFUSION_MATRIX_PATH),
                    caption="YOLO validation confusion matrix",
                    use_container_width=True
                )
            else:
                st.warning(
                    "confusion_matrix.png is not in the repository yet. "
                    "Upload the validation confusion-matrix image beside app.py "
                    "to display it here."
                )

            st.caption(
                "Metrics shown above come from the saved validation evaluation. "
                "Live inspection confidence and session inference timing are separate measurements."
            )

        except Exception as error:
            st.error(f"Could not load or display evaluation metrics: {error}")


# --------------------------------------------------
# ABOUT MODEL
# --------------------------------------------------

elif page == "About Model":

    st.markdown(
        '<div class="section-title">01 / MODEL INFORMATION</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="panel">
        <h3>YOLO11n — Industrial Surface Defect Detector</h3>
        <p class="info-text">
        This application uses a custom-trained YOLO11n checkpoint
        for object detection and classification of steel surface defects.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Supported Classes")

    for index, class_name in enumerate(CLASS_NAMES, start=1):
        st.markdown(f"**{index:02d}.** {class_name}")

    st.markdown("### Detection Pipeline")

    st.write("""
    1. Input image acquisition.
    2. Image preprocessing by the YOLO inference pipeline.
    3. Object detection and classification.
    4. Confidence threshold filtering.
    5. Non-Maximum Suppression.
    6. Bounding-box visualization and report generation.
    """)

    st.markdown("### Important Evaluation Information")

    st.info("""
    Detection confidence is not equivalent to model accuracy.
    Open the Model Evaluation page to view the saved validation
    precision, recall, F1-score, mAP metrics and runtime information.
    """)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
<div style="text-align:center; color:#8ca3b9; padding:15px;">
    <b>STEELVISION AI</b><br>
    Intelligent Industrial Surface Inspection Platform<br>
    <small>Custom-trained YOLO11n | PyTorch | Streamlit</small>
</div>
""", unsafe_allow_html=True)
