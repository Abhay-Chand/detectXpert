import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# Load trained YOLO model
model = YOLO("best.pt")

# App Title & Description
st.title("🔬 DetectXpert - Blood Cell Detector")
st.markdown("Upload an image to detect **RBC, WBC, and Platelets** using YOLOv10.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Confidence slider
conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# Run detection if image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform object detection
    with st.spinner("Detecting..."):
        results = model(image, conf=conf_threshold)
        result_img = results[0].plot()

    # Display results
    st.image(result_img, caption="Detection Result", use_column_width=True)
    
    # Convert NumPy array to PIL image
    result_pil = Image.fromarray(result_img)

    # Convert to BytesIO format for Streamlit download
    img_byte_arr = io.BytesIO()
    result_pil.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()  # Get binary content

    # Update the download button
    st.download_button(">>Download Detection", img_byte_arr, file_name="detection_result.png", mime="image/png")

st.markdown(" >> Created by **Abhay** - Powered by YOLOv10 & Streamlit")
