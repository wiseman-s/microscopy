import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd

# ---------------------- PAGE SETTINGS ----------------------
st.set_page_config(
    page_title="AI Microscopy AMR Detector",
    layout="centered",
)

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
<style>
h1 { text-align: center; font-weight: 800 !important; color: #2e6edf; padding-bottom: 10px; }
.block-container { padding-top: 2rem; }
.footer { text-align: center; margin-top: 40px; font-size: 14px; color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADING ----------------------
st.title("AI Microscopy System for Detecting Antimicrobial Resistance (AMR) in Blood Smears")
st.markdown("""
This system uses a **YOLOv8 deep learning model** to detect AMR indicators directly from microscope blood smear images.
Upload an image or pick a sample to begin inference.
""")

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()
if model is None:
    st.stop()

# ---------------------- SIDEBAR ----------------------
with st.sidebar.expander("ℹ️ About the Project"):
    st.markdown("""
    - **Domain:** AI-assisted microscopy  
    - **Target:** Detect AMR indicators  
    - **Model:** YOLOv8s FP16  
    - **Format:** Web-based inference system  
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

                # Draw bounding box
                cv2.rectangle(img_out, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

                # Label text
                txt = f"{label} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img_out, (xyxy[0], xyxy[1]-20), (xyxy[0]+w, xyxy[1]), (0, 255, 0), -1)
                cv2.putText(img_out, txt, (xyxy[0], xyxy[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        # Display result
        img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        st.image(img_out_rgb, caption="Detection Result", use_container_width=True)

        # Detection table
        if detections:
            df = pd.DataFrame(detections).sort_values(by="Confidence", ascending=False)
            st.markdown("### Detection Summary")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No AMR indicators detected.")

# ---------------------- FOOTER ----------------------
st.markdown("""
<div class="footer">
    System by <strong>Simon</strong> — contact: <a href="mailto:allinmer57@gmail.com">allinmer57@gmail.com</a>
</div>
""", unsafe_allow_html=True)
