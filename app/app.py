"""VisionAI - Milestone 8: Interactive Streamlit Web Application

A full-featured, portfolio-grade computer vision dashboard providing:
1. System Overview & Architecture Map
2. Live Vision Feed (Webcam + Combined Face & Object Engine + Dynamic Sensitivity + SQLite Logging)
3. Face Registration & Training with Automated Data Augmentation
4. Static Image Analysis with Threshold Tuning
5. Event Database History with CSV Export
6. Real-Time Analytics & BI Charts
7. Privacy & Ethical AI Guidelines
"""

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FACES_DIR,
    TEST_IMAGES_DIR,
    ANALYTICS_DIR,
    FACE_SIMILARITY_THRESHOLD,
    YOLO_CONFIDENCE_THRESHOLD,
    DEFAULT_CAMERA_INDEX,
)
from src.face_recognition import FaceRecognizer
from src.object_detection import ObjectDetector
from src.vision_engine import CombinedVisionEngine
from src.database import EventDatabase
from src.analytics import AnalyticsEngine


# Configure Streamlit page
st.set_page_config(
    page_title="VisionAI — Face & Object Recognition System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_vision_engine() -> CombinedVisionEngine:
    """Caches the heavy Vision Engine so models load only once."""
    return CombinedVisionEngine(enable_faces=True, enable_objects=True)


@st.cache_resource
def get_event_database() -> EventDatabase:
    """Caches database connection."""
    return EventDatabase()


# Initialize database and analytics
db = get_event_database()
analytics = AnalyticsEngine(db=db)
engine = get_vision_engine()


# Sidebar Navigation
st.sidebar.title("👁️ VisionAI System")
st.sidebar.caption("Intelligent Face & Object Recognition Engine")
st.sidebar.markdown("---")

menu_choice = st.sidebar.radio(
    "Select Module:",
    [
        "🏠 System Overview",
        "📷 Live Vision Feed",
        "👤 Face Registration & Training",
        "🔍 Static Image Inspector",
        "🗄️ Event Database",
        "📊 Visual Analytics",
        "📖 Privacy & Ethics",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
st.sidebar.success("✅ OpenCV 5.0 & YuNet (Active)")
st.sidebar.success("✅ SFace 128-D Embeddings (Active)")
st.sidebar.success("✅ YOLOv8 Nano (Active)")

# Show enrolled identities in sidebar
registered_names = list(engine.face_recognizer.database.keys()) if engine.face_recognizer else []
if registered_names:
    st.sidebar.info(f"👤 Enrolled: **{', '.join(registered_names)}**")
else:
    st.sidebar.warning("⚠️ No enrolled faces yet")

st.sidebar.caption(f"📁 Logged Events in SQLite: {db.get_total_count():,}")


# ---------------------------------------------------------
# PAGE 1: SYSTEM OVERVIEW
# ---------------------------------------------------------
if menu_choice == "🏠 System Overview":
    st.title("👁️ VisionAI — Intelligent Face & Object Recognition System")
    st.markdown("### Production-Grade Computer Vision & Deep Learning Engine")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    kpis = analytics.compute_summary_kpis()
    col1.metric("Total Events Logged", f"{kpis['total_detections']:,}")
    col2.metric("Top Detected Entity", str(kpis['top_object']))
    col3.metric("Enrolled Identities", f"{len(registered_names):,}")
    col4.metric("Average Confidence", f"{kpis['average_confidence']*100:.1f}%")

    st.markdown("### 🧩 Pipeline Architecture")
    st.markdown(
        """
        ```text
        [ Camera / Image Input ]
                   │
                   ├───> [ Face Detection (YuNet ONNX) ] ──> [ Face Alignment (5 Landmarks) ]
                   │                                                     │
                   │                                                     ▼
                   │                                      [ SFace 128-D Embedding Generator ]
                   │                                                     │
                   │                                                     ▼
                   │                                    [ Cosine Similarity Matcher vs Enrolled DB ]
                   │                                                     │
                   ├───> [ Object Detection (YOLOv8) ]                   │
                   │              │                                      │
                   ▼              ▼                                      ▼
             [ Spatial Fusion Engine: Associates Person Boxes with Face Identities ]
                                  │
                                  ├───> [ High-Contrast Graphical Annotator (HUD) ]
                                  │
                                  └───> [ SQLite Event Logger (with 4s Cooldown Throttling) ]
                                                  │
                                                  ▼
                                     [ Pandas Analytics & BI Charts ]
        ```
        """
    )

    st.markdown("### 🚀 Core Engineering Highlights")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            * **Deep Metric Learning**: Uses 128-dimensional hypersphere embeddings ($L_2$ normalized) where geometric vector angle equals identity similarity.
            * **Real-time YOLOv8 Inference**: Detects common everyday objects (laptops, phones, cups, chairs) at 30+ FPS.
            * **Cross-Model Spatial Fusion**: Intelligently correlates YOLO 'person' bounding boxes with inner facial landmark clusters to produce clean, unified labels.
            """
        )
    with c2:
        st.markdown(
            """
            * **Automated Data Augmentation**: Enrolls identities with 7+ synthetic augmentations (flips, contrast, lighting, rotations) for robust recognition.
            * **Throttled SQLite Event Logger**: Eliminates database lock contention and thrashing from continuous 30 FPS video feeds.
            * **Interactive Streamlit Web Dashboard**: Real-time webcam streaming, dynamic registration, visual threshold tuning, and BI charts.
            """
        )


# ---------------------------------------------------------
# PAGE 2: LIVE VISION FEED
# ---------------------------------------------------------
elif menu_choice == "📷 Live Vision Feed":
    st.title("📷 Live Vision Feed")
    st.caption("Real-Time Multi-Model Vision Pipeline with Live SQLite Event Logging")
    st.markdown("---")

    # Enrolled identities reminder
    if registered_names:
        st.success(f"Recognizing enrolled identities: **{', '.join(registered_names)}** (All other faces will be labeled as 'Unknown Face')")
    else:
        st.warning("⚠️ No faces currently enrolled! Please go to **'Face Registration & Training'** tab in the sidebar to enroll your face first.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.2, 1, 1])
    with col_ctrl1:
        run_camera = st.toggle("🔴 Start Live Camera Stream", value=False)
    with col_ctrl2:
        log_to_db = st.checkbox("💾 Log to SQLite", value=True)
    with col_ctrl3:
        camera_idx = st.number_input("Camera Index", min_value=0, max_value=5, value=DEFAULT_CAMERA_INDEX)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        live_thresh = st.slider("Face Recognition Sensitivity", 0.20, 0.60, float(FACE_SIMILARITY_THRESHOLD), 0.02,
                               help="Lower = more sensitive. Higher = stricter.")
    with col_s2:
        yolo_live_conf = st.slider("Object & Animal Detection Sensitivity", 0.10, 0.70, float(YOLO_CONFIDENCE_THRESHOLD), 0.05,
                                   help="Controls sensitivity for detecting animals, laptops, phones, cups, etc. (Default: 0.25)")

    engine.set_similarity_threshold(live_thresh)
    engine.set_yolo_confidence(yolo_live_conf)

    col_view, col_stats = st.columns([3, 1])

    with col_stats:
        st.markdown("#### Live Telemetry")
        fps_placeholder = st.empty()
        face_count_placeholder = st.empty()
        obj_count_placeholder = st.empty()
        st.markdown("---")
        st.markdown("##### Detected Entities")
        recent_list_placeholder = st.empty()

    with col_view:
        frame_placeholder = st.empty()

    if run_camera:
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)

        if not cap.isOpened():
            st.error(f"Could not connect to camera at index {camera_idx}. Check camera privacy settings.")
        else:
            prev_time = time.time()
            frame_count = 0
            fps = 0.0

            while run_camera:
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.warning("Video stream ended or frame could not be grabbed.")
                    break

                frame_count += 1
                cur_time = time.time()
                if cur_time - prev_time > 0.5:
                    fps = frame_count / (cur_time - prev_time)
                    frame_count = 0
                    prev_time = cur_time

                # Run unified vision engine
                result = engine.process_frame(frame)
                result.fps = fps
                annotated = engine.draw_results(frame, result)

                # Optional database logging
                if log_to_db and result.fused_entities:
                    db.log_vision_result(result, source="live_webcam")

                # Convert BGR to RGB for Streamlit display
                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                # Update telemetry
                fps_placeholder.metric("Inference Rate", f"{fps:.1f} FPS")
                face_count_placeholder.metric("Faces Detected", f"{len(result.faces)}")
                obj_count_placeholder.metric("Objects Detected", f"{len(result.objects)}")

                # Recent entities snippet
                if result.fused_entities:
                    summary_text = "\n".join([f"- **{e['label']}** ({e['confidence']*100:.0f}%)" for e in result.fused_entities[:6]])
                    recent_list_placeholder.markdown(summary_text)

            cap.release()
    else:
        frame_placeholder.info("Toggle **'Start Live Camera Stream'** above to start real-time computer vision inference.")


# ---------------------------------------------------------
# PAGE 3: FACE REGISTRATION & TRAINING
# ---------------------------------------------------------
elif menu_choice == "👤 Face Registration & Training":
    st.title("👤 Face Registration & Training")
    st.markdown("Train the 128-dimensional SFace metric learning model on your face with automated multi-shot data augmentation.")
    st.info("ℹ️ **Human Biometrics Note:** This module registers and trains **human face identities** into the SFace 128-D embedding database. If you want to detect **cats, dogs, animals, or general objects**, visit the **🔍 Static Image Inspector** or **📷 Live Vision Feed**!")
    st.markdown("---")

    recognizer = engine.face_recognizer

    tab_reg, tab_browse = st.tabs(["➕ Enroll & Train New Face", "📋 Enrolled Identities Registry"])

    with tab_reg:
        reg_col1, reg_col2 = st.columns(2)
        with reg_col1:
            person_name = st.text_input("Person Name to Enroll:", placeholder="e.g. Priyanka")
            use_aug = st.checkbox("Apply Automated Data Augmentation (Recommended)", value=True,
                                  help="Generates 7+ variations (flips, lighting, rotations) to train a robust embedding cluster.")
            upload_method = st.radio("Photo Input Method:", ["Take Snapshot from Camera", "Upload Photo"])
            input_image = None

            if upload_method == "Upload Photo":
                uploaded_file = st.file_uploader("Upload a clear portrait photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    bytes_data = uploaded_file.read()
                    input_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            else:
                cam_photo = st.camera_input("Capture Face Snapshot")
                if cam_photo is not None:
                    bytes_data = cam_photo.read()
                    input_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            register_button = st.button("🚀 Train & Enroll Identity", type="primary")

        with reg_col2:
            if input_image is not None:
                st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), caption="Captured Preview", width=350)
                if register_button:
                    clean_name = person_name.strip()
                    if not clean_name:
                        st.error("Please enter a valid person name.")
                    else:
                        success = recognizer.register_face(clean_name, input_image, use_augmentation=use_aug)
                        if success:
                            st.success(f"🎉 Successfully trained and enrolled '{clean_name}'! The model can now recognize you in the Live Vision Feed.")
                            st.rerun()
                        else:
                            st.error("Could not detect any face in the photo. Please ensure good lighting and face camera directly.")

    with tab_browse:
        st.markdown("#### Currently Enrolled Identities")
        db_entries = recognizer.database
        if not db_entries:
            st.info("No identities currently registered.")
        else:
            for name, embeddings in db_entries.items():
                st.markdown(f"**Identity:** `{name}` | **Embedding Vectors in Cluster:** `{len(embeddings)}` | **Feature Dimensions:** `128-D`")


# ---------------------------------------------------------
# PAGE 4: STATIC IMAGE INSPECTOR
# ---------------------------------------------------------
elif menu_choice == "🔍 Static Image Inspector":
    st.title("🔍 Static Image Inspector & Threshold Playground")
    st.markdown("Upload any photo to inspect Face Recognition and YOLOv8 detections with interactive threshold sliders.")
    st.markdown("---")

    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        sim_thresh = st.slider("Face Recognition Threshold", 0.1, 0.9, float(FACE_SIMILARITY_THRESHOLD), 0.02,
                               help="Cosine similarity cutoff for matching a human face against enrolled identities.")
    with ctrl_col2:
        yolo_conf = st.slider("YOLO Object & Animal Confidence", 0.05, 0.95, float(YOLO_CONFIDENCE_THRESHOLD), 0.05,
                              help="Minimum confidence for detecting animals (cat, dog, bird, etc.) and COCO objects.")

    input_source = st.radio(
        "Select Image Input Mode:",
        ["📁 Upload Image from Device (Cat, Dog, Animals, People, Objects)", "🖼️ Select Benchmark Sample from Gallery"],
        horizontal=True,
    )

    image_to_process = None
    image_name = "Uploaded Image"

    if "Upload Image" in input_source:
        uploaded = st.file_uploader(
            "Upload any image (JPG, JPEG, PNG, WEBP, BMP):",
            type=["jpg", "jpeg", "png", "webp", "bmp", "jfif"],
            help="Upload any picture of a cat, dog, animal, person, or household item.",
        )
        if uploaded is not None:
            bytes_data = uploaded.read()
            image_to_process = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            image_name = uploaded.name
    else:
        test_img_files = [f.name for f in sorted(TEST_IMAGES_DIR.glob("*.jpg"))]
        available_samples = test_img_files.copy()
        for person_dir in sorted(FACES_DIR.iterdir()):
            if person_dir.is_dir() and person_dir.name != "Lena":
                sample_file = next(person_dir.glob("*.jpg"), None)
                if sample_file:
                    available_samples.append(f"Enrolled Face: {person_dir.name}")

        selected_sample = st.selectbox("Choose pre-loaded test image:", available_samples)
        if selected_sample:
            if selected_sample.startswith("Enrolled Face: "):
                p_name = selected_sample.split("Enrolled Face: ")[1].strip()
                sample_path = FACES_DIR / p_name / "pose_01.jpg"
            else:
                sample_path = TEST_IMAGES_DIR / selected_sample
            if sample_path.exists():
                image_to_process = cv2.imread(str(sample_path))
                image_name = selected_sample

    if image_to_process is not None:
        # Reset temporal tracks for static image analysis
        engine.tracks.clear()

        # Update sensitivity thresholds
        engine.set_similarity_threshold(sim_thresh)
        engine.set_yolo_confidence(yolo_conf)

        res = engine.process_frame(image_to_process)
        annotated = engine.draw_results(image_to_process, res)

        # Telemetry metrics
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        stat_c1.metric("Image Resolution", f"{image_to_process.shape[1]}x{image_to_process.shape[0]}")
        stat_c2.metric("Animals & Objects", f"{len(res.objects)}")
        stat_c3.metric("Human Faces", f"{len(res.faces)}")
        stat_c4.metric("Total Identified", f"{len(res.fused_entities)}")

        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown(f"##### Original Image: `{image_name}`")
            st.image(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB), use_container_width=True)
        with v_col2:
            st.markdown("##### Annotated Detections (All 80 COCO Classes & Faces)")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("#### Detected Entities Breakdown")
        if res.fused_entities:
            table_data = []
            for e in res.fused_entities:
                is_animal = e["label"].lower() in [
                    "cat", "dog", "bird", "horse", "sheep", "cow",
                    "elephant", "bear", "zebra", "giraffe", "teddy bear"
                ]
                entity_kind = "🐾 Animal" if is_animal else ("👤 Person (Face)" if e.get("is_recognized_face") else f"📦 {e['type'].capitalize()}")
                table_data.append({
                    "Category": entity_kind,
                    "Detected Label / Identity": e["label"],
                    "Confidence / Similarity": f"{e['confidence']*100:.1f}%",
                    "Bounding Box (x, y, w, h)": str(e["box"]),
                    "Recognized Face": "Yes" if e.get("is_recognized_face") else "No",
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        else:
            st.warning(f"No entities detected under current thresholds (YOLO: {yolo_conf*100:.0f}%). Try lowering the 'YOLO Object & Animal Confidence' slider above!")


# ---------------------------------------------------------
# PAGE 5: EVENT DATABASE
# ---------------------------------------------------------
elif menu_choice == "🗄️ Event Database":
    st.title("🗄️ Event Database History")
    st.markdown("Browse, search, filter, and export computer vision detection events from SQLite.")
    st.markdown("---")

    df = db.get_all_events_df()

    if df.empty:
        st.info("Database currently has 0 logged events. Run the Live Vision Feed to populate events!")
    else:
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            type_filter = st.multiselect("Filter by Type:", options=df["detection_type"].unique(), default=df["detection_type"].unique())
        with f_col2:
            search_query = st.text_input("Search Label:", placeholder="e.g. laptop, Lena")
        with f_col3:
            if st.button("🗑️ Clear Entire Database History", type="secondary"):
                db.clear_database()
                st.rerun()

        filtered_df = df[df["detection_type"].isin(type_filter)]
        if search_query.strip():
            filtered_df = filtered_df[filtered_df["label"].str.contains(search_query.strip(), case=False)]

        st.markdown(f"Displaying **{len(filtered_df):,}** of **{len(df):,}** total events:")
        st.dataframe(filtered_df.sort_values("id", ascending=False), use_container_width=True)

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export Events to CSV", csv_data, "visionai_detection_events.csv", "text/csv")


# ---------------------------------------------------------
# PAGE 6: VISUAL ANALYTICS
# ---------------------------------------------------------
elif menu_choice == "📊 Visual Analytics":
    st.title("📊 Business Intelligence & Analytics Dashboard")
    st.markdown("Quantitative insights and statistical distributions from detection history.")
    st.markdown("---")

    kpis = analytics.compute_summary_kpis()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Events", f"{kpis['total_detections']:,}")
    m2.metric("Top Object", f"{kpis['top_object']}")
    m3.metric("Recognized Faces", f"{kpis['recognized_faces_count']:,}")
    m4.metric("Unknown Faces", f"{kpis['unknown_faces_count']:,}")
    m5.metric("Avg Confidence", f"{kpis['average_confidence']*100:.1f}%")

    if kpis["total_detections"] == 0:
        st.info("No data available to plot charts. Run live vision feed to record detection events.")
    else:
        chart_paths = analytics.generate_all_charts()

        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            if "class_distribution" in chart_paths:
                st.image(str(chart_paths["class_distribution"]), use_container_width=True)
            if "confidence_distribution" in chart_paths:
                st.image(str(chart_paths["confidence_distribution"]), use_container_width=True)

        with ch_col2:
            if "face_breakdown" in chart_paths:
                st.image(str(chart_paths["face_breakdown"]), use_container_width=True)
            if "timeline" in chart_paths:
                st.image(str(chart_paths["timeline"]), use_container_width=True)


# ---------------------------------------------------------
# PAGE 7: PRIVACY & ETHICS
# ---------------------------------------------------------
elif menu_choice == "📖 Privacy & Ethics":
    st.title("📖 Privacy, Security & Ethical AI Guidelines")
    st.markdown("---")

    st.markdown(
        """
        ### 🔒 Privacy Considerations in Facial Recognition
        * **Facial Data is Sensitive Biometric Information**: Face images and their numerical embedding representations are unique identifiers.
        * **Local Processing Guarantee**: VisionAI processes all camera frames, facial crops, and embeddings **entirely locally on-device**. No images or embeddings are transmitted to third-party cloud APIs.
        * **Consent-First Principle**: Individuals must explicitly consent before their photos or face embeddings are registered into the identity database.
        * **No Covert Surveillance**: This application is built as an educational and portfolio demonstration. It does not include covert tracking, background persistence, or unauthorized identification capabilities.

        ### ⚠️ System Limitations & Realistic Expectations
        * **Probabilistic Nature**: Neural networks compute probability distributions, not absolute certainties. Lighting changes, extreme facial angles, occlusion, and low resolution can cause false matches or false rejections.
        * **Do Not Use for High-Stakes Decisions**: This system is not designed or certified for life-critical, legal, biometric authentication, or access-control decisions.
        * **Threshold Tuning Tradeoff**:
          * High threshold $\\rightarrow$ Low False Acceptance Rate (FAR), but higher False Rejection Rate (FRR).
          * Low threshold $\\rightarrow$ Higher False Acceptance Rate (FAR), lower False Rejection Rate (FRR).
        """
    )
