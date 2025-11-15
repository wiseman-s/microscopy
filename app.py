import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd

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

/* Sample image box */
.sample-box {
    background-color: #1f1f1f;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #333;
    margin-bottom: 20px;
}

/* Detection summary card */
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
    color: #000000;
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADING ----------------------
st.title("AI Microscopy System")
st.markdown("""
Upload a microscope image or select a sample from the repository to detect antimicrobial resistance indicators.
The system provides real-time detection with confidence scores, bounding boxes, and a summary table for analysis.
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
with st.sidebar.expander("ℹ️ About this Project"):
    st.markdown("""
    ### About AI Microscopy System
    
    **AI Microscopy System** is a cutting-edge platform designed for **real-time analysis of blood smear images**. 
    Using **YOLOv8s FP16**, it detects potential indicators of **antimicrobial resistance (AMR)** with high precision.

    #### Key Features:
    - **Fast and Accurate Detection:** Real-time inference using YOLOv8 optimized for FP16.
    - **Professional Visualization:** Bounding boxes colored based on confidence, with interactive summary tables.
    - **Flexible Input:** Accepts custom uploads or sample images stored in the repository.
    - **Dark Mode UI:** Polished dark theme optimized for phones and laptops.
    - **Research Utility:** Designed for academic research, diagnostics demonstration, and AI-assisted microscopy studies.
    
    *Accelerating the workflow of AMR research through intuitive AI-assisted microscopy.*
    """)

# ---------------------- FILE INPUT ----------------------
st.subheader("Input Image")
uploaded_file = st.file_uploader("Upload microscope image (.jpg, .png)", type=["jpg", "png"])

# ---------------------- SAMPLE IMAGES FROM ROOT ----------------------
st.markdown("Or select a sample image:")

sample_images = [f for f in os.listdir(".") 
                 if f.lower().endswith(".png") and f[0].isdigit()]

selected_sample = st.selectbox("Sample Images in Repo Root:", [""] + sample_images)

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif selected_sample:
    image = Image.open(selected_sample).convert("RGB")

# Display image
if image is not None:
    st.image(image, caption="Selected Image", use_container_width=True)

# ---------------------- BUTTON: RUN DETECTION ----------------------
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

            # Process detection results
            img_out = img_cv.copy()
            detections = []

            for r in results:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    label = model.names[cls] if model.names else str(cls)

                    detections.append({"Class": label, "Confidence": conf})

                    # Color based on confidence (green → yellow → cyan)
                    green_intensity = int(conf * 255)
                    color = (0, green_intensity, 255 - green_intensity)

                    # Draw bounding box
                    cv2.rectangle(img_out, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)

                    # Label text
                    txt = f"{label} {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(img_out, (xyxy[0], xyxy[1]-20), (xyxy[0]+w, xyxy[1]), color, -1)
                    cv2.putText(img_out, txt, (xyxy[0], xyxy[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

            # Display result
            img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
            st.image(img_out_rgb, caption="Detection Result", use_container_width=True)

            # Detection table in card
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
