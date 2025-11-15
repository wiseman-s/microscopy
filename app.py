import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import plotly.graph_objects as go

# ---------------------- PAGE SETTINGS ----------------------
st.set_page_config(
    page_title="AI Microscopy System",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
<style>
/* Dark background & light text */
body, .stApp, .block-container {
    background-color: #121212;
    color: #ffffff;
}

/* Gradient Title */
h1 {
    text-align: center;
    font-weight: 900 !important;
    background: -webkit-linear-gradient(45deg, #00bfff, #00ffcc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 10px;
}

/* Subheaders */
h2, h3 {
    color: #00ffff;
}

/* Card styling */
.card {
    background-color: #2a2a2a;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.3);
    margin-top: 20px;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 14px;
    color: #aaaaaa;
}

/* Buttons */
.stButton>button {
    background-color: #00bfff;
    color: #000000;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 24px;
}
.stButton>button:hover {
    background-color: #00ffff;
    color: #000000;
}

/* File uploader */
.stFileUploader>div>div>input {
    color: #000000 !important;
    background-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADING ----------------------
st.title("AI Microscopy System")
st.markdown("""
Upload a blood smear image or select a sample from the repository to detect antimicrobial resistance indicators.
The system provides real-time detection with confidence scores, bounding boxes, and an interactive summary table.
""")

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()
if model is None:
    st.stop()

# ---------------------- SIDEBAR: ABOUT ----------------------
with st.sidebar.expander("ℹ️ About This Project"):
    st.markdown("""
    ### About This Project
    
    **AI Microscopy System** is a cutting-edge platform for **real-time analysis of blood smear images**. 
    Using **YOLOv8s FP16**, it detects potential **antimicrobial resistance (AMR) indicators** quickly and accurately.

    #### Key Features:
    - **Fast and Accurate Detection:** YOLOv8 optimized for FP16 for high-speed inference.
    - **Interactive Visualization:** Bounding boxes colored by confidence, with hover tooltips and summary tables.
    - **Flexible Input:** Upload custom images or choose from sample repository images.
    - **Dark Mode UI:** Optimized for both phones and laptops for professional appearance.
    - **Research & Demo Utility:** Designed for academic research, diagnostics demonstration, and AI-assisted microscopy.
    """)

# ---------------------- FILE INPUTS ----------------------
st.subheader("Upload Sample Image")
uploaded_file = st.file_uploader("Upload blood smear image (.jpg, .png)", type=["jpg", "png"])

st.subheader("Or select a sample from the repository")
sample_images = [f for f in os.listdir(".") 
                 if f.lower().endswith(".png") and f[0].isdigit()]
selected_sample = st.selectbox("Upload Sample Image:", [""] + sample_images)

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif selected_sample:
    image = Image.open(selected_sample).convert("RGB")

if image is not None:
    st.image(image, caption="Selected Image", use_container_width=True)

# ---------------------- RUN DETECTION ----------------------
if st.button("🔍 Run Detection"):
    if image is None:
        st.warning("Please upload or select an image first.")
    else:
        with st.spinner("Running detection..."):
            img_np = np.array(image)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            try:
                results = model(img_cv)
            except Exception as e:
                st.error(f"Error during inference: {e}")
                st.stop()

            # Prepare Plotly figure for hover
            height, width, _ = img_np.shape
            fig = go.Figure()
            fig.add_trace(go.Image(z=img_np))

            detections = []

            for r in results:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    label = model.names[cls] if model.names else str(cls)

                    detections.append({"Class": label, "Confidence": conf})

                    # Bounding box color based on confidence
                    green_intensity = int(conf * 255)
                    color_hex = f'rgb(0,{green_intensity},{255-green_intensity})'

                    # Add rectangle to Plotly
                    fig.add_shape(
                        type="rect",
                        x0=xyxy[0], y0=xyxy[1], x1=xyxy[2], y1=xyxy[3],
                        line=dict(color=color_hex, width=3)
                    )
                    # Add hover text at top-left corner
                    fig.add_trace(go.Scatter(
                        x=[xyxy[0]], y=[xyxy[1]],
                        text=[f"{label}: {conf:.2f}"],
                        mode="markers+text",
                        marker=dict(size=1, color=color_hex),
                        textposition="top left",
                        showlegend=False
                    ))

            fig.update_layout(
                xaxis=dict(visible=False, range=[0, width]),
                yaxis=dict(visible=False, range=[height, 0]),
                margin=dict(l=0, r=0, t=0, b=0),
                autosize=True,
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detection summary in card
            if detections:
                df = pd.DataFrame(detections).sort_values(by="Confidence", ascending=False)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Detection Summary")
                st.dataframe(df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No AMR indicators detected.")

# ---------------------- FOOTER ----------------------
st.markdown("""
<div class="footer">
    System by <strong>Simon</strong> — contact: <a href="mailto:allinmer57@gmail.com">allinmer57@gmail.com</a>
</div>
""", unsafe_allow_html=True)
