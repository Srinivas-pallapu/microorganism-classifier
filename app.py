import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import tempfile
from collections import Counter
import plotly.graph_objects as go

st.set_page_config(
    page_title="Microorganism Image Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
* { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }

[data-testid="stMainBlockContainer"] {
    padding-top: 2rem;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 3rem 2rem;
    border-radius: 15px;
    color: white;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    margin: 0;
    font-size: 2.8rem;
    font-weight: 700;
}

.main-header p {
    margin: 0.8rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.95;
}

.prediction-result {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 8px 32px rgba(17, 153, 142, 0.25);
}

.confidence-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.2);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 1rem;
}

.warning-box {
    background: #fff3cd;
    color: #856404;
    padding: 1.2rem;
    border-radius: 12px;
    border-left: 6px solid #ffc107;
    margin: 1rem 0;
    font-weight: 500;
}

.info-box {
    background: #d4edda;
    color: #155724;
    padding: 1.2rem;
    border-radius: 12px;
    border-left: 6px solid #28a745;
    margin: 1rem 0;
    font-weight: 500;
}

.organism-card {
    background: #ffffff;
    color: #222222;
    border: 2px solid #4e73df;
    padding: 1.4rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.organism-card h3 {
    color: #4e73df;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🔬 Microorganism Image Classifier</h1>
    <p>Advanced microscopic image analysis powered by deep learning</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("micro_model.h5")

model = load_model()

class_names = [
    "Amoeba",
    "Euglena",
    "Hydra",
    "Paramecium",
    "Rod_bacteria",
    "Spherical_bacteria",
    "Spiral_bacteria",
    "Yeast"
]

info = {
    "Amoeba": "🦠 A single-celled protozoan that moves using pseudopodia. Commonly found in freshwater, soil, and moist environments.",
    "Euglena": "🌿 A unicellular organism with both plant-like and animal-like features. It moves using a flagellum and can perform photosynthesis.",
    "Hydra": "🪸 A freshwater organism known for regeneration ability. It belongs to phylum Cnidaria.",
    "Paramecium": "👾 A unicellular protozoan that moves using cilia. Commonly found in freshwater environments.",
    "Rod_bacteria": "📏 Cylindrical-shaped bacteria, also called bacilli. They are important in medical and industrial fields.",
    "Spherical_bacteria": "⭕ Round-shaped bacteria, also called cocci. They may occur singly, in chains, or clusters.",
    "Spiral_bacteria": "🌀 Spiral-shaped bacteria, also called spirilla. They often move using flagella.",
    "Yeast": "🍞 A unicellular fungus used in baking, brewing, fermentation, and biotechnology."
}

colors = {
    "Amoeba": "#FF6B6B",
    "Euglena": "#4ECDC4",
    "Hydra": "#45B7D1",
    "Paramecium": "#FFA07A",
    "Rod_bacteria": "#98D8C8",
    "Spherical_bacteria": "#F7DC6F",
    "Spiral_bacteria": "#BB8FCE",
    "Yeast": "#85C1E2"
}

def preprocess_image(image):
    image = image.convert("RGB")
    image = ImageOps.pad(image, (224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0)[0]
    pred_index = np.argmax(prediction)
    pred_class = class_names[pred_index]
    confidence = float(prediction[pred_index]) * 100
    return pred_class, confidence, prediction

def create_confidence_chart(probabilities):
    organisms = [name.replace("_", " ") for name in class_names]
    confidence = probabilities * 100
    bar_colors = [colors[name] for name in class_names]

    fig = go.Figure(data=[
        go.Bar(
            y=organisms,
            x=confidence,
            orientation="h",
            marker=dict(color=bar_colors),
            text=[f"{v:.1f}%" for v in confidence],
            textposition="auto"
        )
    ])

    fig.update_layout(
        title="Prediction Confidence Scores",
        xaxis_title="Confidence (%)",
        yaxis_title="Microorganism",
        height=400,
        showlegend=False,
        template="plotly_white",
        margin=dict(l=150)
    )
    return fig

def create_gauge(confidence, pred_class):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={"text": f"Prediction: {pred_class.replace('_', ' ')}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": colors.get(pred_class, "#667eea")},
            "steps": [
                {"range": [0, 30], "color": "#f5f7fa"},
                {"range": [30, 60], "color": "#e8f4f8"},
                {"range": [60, 100], "color": "#d4f4e6"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=400)
    return fig

def show_prediction_result(pred_class, confidence, probabilities):
    st.markdown(f"""
    <div class="prediction-result">
        <h2>{pred_class.replace('_', ' ')}</h2>
        <div class="confidence-badge">
            Confidence: {confidence:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    if confidence < 30:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Low Confidence</strong><br>
            The model is uncertain. Upload a clearer microscope image.
        </div>
        """, unsafe_allow_html=True)
    elif confidence < 50:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Moderate Confidence</strong><br>
            Check the top predictions before accepting the result.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <strong>✅ High Confidence Prediction</strong><br>
            The model is confident in this classification.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="organism-card">
        <h3>About {pred_class.replace('_', ' ')}</h3>
        <p>{info[pred_class]}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_gauge(confidence, pred_class), use_container_width=True)

    with col2:
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        fig_top3 = go.Figure(data=[
            go.Bar(
                x=[class_names[i].replace("_", " ") for i in top3_idx],
                y=[probabilities[i] * 100 for i in top3_idx],
                marker_color=["#11998e", "#38ef7d", "#6ecccc"],
                text=[f"{probabilities[i] * 100:.1f}%" for i in top3_idx],
                textposition="auto"
            )
        ])
        fig_top3.update_layout(
            title="Top 3 Predictions",
            xaxis_title="Microorganism",
            yaxis_title="Confidence (%)",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_top3, use_container_width=True)

    st.subheader("📊 Complete Analysis")
    st.plotly_chart(create_confidence_chart(probabilities), use_container_width=True)

with st.sidebar:
    st.markdown("### 📖 About This App")
    st.write("""
    This image classifier identifies 8 microorganisms:

    **Protozoans**
    - Amoeba
    - Euglena
    - Hydra
    - Paramecium

    **Bacteria**
    - Rod-shaped bacteria
    - Spherical bacteria
    - Spiral bacteria

    **Fungi**
    - Yeast
    """)

    st.divider()
    st.caption("🔬 Powered by TensorFlow Deep Learning")

tab1, tab2, tab3 = st.tabs(["🖼️ IMAGE ANALYSIS", "🎥 VIDEO ANALYSIS", "📷 LIVE WEBCAM"])

with tab1:
    st.subheader("Upload Microorganism Image")

    uploaded_image = st.file_uploader(
        "Select an image file",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear microscopic image of a microorganism"
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Microscopic Image", width=450)

        if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
            with st.spinner("🧬 Analyzing microorganism..."):
                pred_class, confidence, probabilities = predict_image(image)
                show_prediction_result(pred_class, confidence, probabilities)

with tab2:
    st.subheader("Upload Microorganism Video")

    uploaded_video = st.file_uploader(
        "Select a video file",
        type=["mp4", "avi", "mov"],
        help="Upload a video containing microorganism footage"
    )

    if uploaded_video is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_video.read())

        st.video(temp_file.name)

        col1, col2 = st.columns(2)

        with col1:
            frame_gap = st.slider(
                "Process every Nth frame",
                min_value=1,
                max_value=30,
                value=15
            )

        with col2:
            confidence_threshold = st.slider(
                "Confidence threshold (%)",
                min_value=10,
                max_value=100,
                value=50
            )

        if st.button("🎬 Analyze Video", use_container_width=True, type="primary"):
            cap = cv2.VideoCapture(temp_file.name)

            frame_predictions = []
            frame_confidences = []
            frame_count = 0
            processed_frames = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_gap == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)

                    pred_class, confidence, _ = predict_image(pil_frame)

                    if confidence >= confidence_threshold:
                        frame_predictions.append(pred_class)
                        frame_confidences.append(confidence)

                    processed_frames += 1
                    progress_bar.progress(min(processed_frames / 100, 1.0))
                    status_text.text(f"Processing frame {processed_frames}...")

                frame_count += 1

            cap.release()
            progress_bar.empty()
            status_text.empty()

            if frame_predictions:
                final_class = Counter(frame_predictions).most_common(1)[0][0]

                related_conf = [
                    frame_confidences[i]
                    for i in range(len(frame_predictions))
                    if frame_predictions[i] == final_class
                ]

                avg_confidence = sum(related_conf) / len(related_conf)

                st.markdown(f"""
                <div class="prediction-result">
                    <h2>{final_class.replace('_', ' ')}</h2>
                    <div class="confidence-badge">
                        Average Confidence: {avg_confidence:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="organism-card">
                    <h3>About {final_class.replace('_', ' ')}</h3>
                    <p>{info[final_class]}</p>
                </div>
                """, unsafe_allow_html=True)

                counts = Counter(frame_predictions)

                col1, col2 = st.columns(2)

                with col1:
                    fig_frames = go.Figure(data=[
                        go.Pie(
                            labels=[k.replace("_", " ") for k in counts.keys()],
                            values=list(counts.values()),
                            marker=dict(colors=[colors.get(k, "#667eea") for k in counts.keys()])
                        )
                    ])
                    fig_frames.update_layout(title="Frame Distribution", height=400)
                    st.plotly_chart(fig_frames, use_container_width=True)

                with col2:
                    st.markdown("### 📈 Frame-wise Summary")
                    for cls, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / len(frame_predictions)) * 100
                        st.write(f"**{cls.replace('_', ' ')}**: {count} frames ({percentage:.1f}%)")
            else:
                st.error("❌ No frames could be processed. Try lowering the confidence threshold.")

with tab3:
    st.subheader("Real-time Microscope Capture")
    st.write("Capture images directly from webcam or connected microscope camera.")

    camera_image = st.camera_input("📷 Capture microscope image")

    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", width=450)

        if st.button("🔍 Analyze Captured Image", use_container_width=True, type="primary"):
            with st.spinner("🧬 Analyzing microorganism..."):
                pred_class, confidence, probabilities = predict_image(image)
                show_prediction_result(pred_class, confidence, probabilities)

st.divider()
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem 0; font-size: 0.9rem;">
    <p>🔬 <strong>Microorganism Image Classifier</strong></p>
</div>
""", unsafe_allow_html=True)