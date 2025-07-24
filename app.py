import streamlit as st
from utils import preprocess_image, download_model_from_url, load_model
import os

# Put your direct download link here (Google Drive example):
MODEL_URL = 'https://drive.google.com/uc?id=YOUR_FILE_ID&export=download'
MODEL_PATH = 'resnet18_brain_tumor.pth'

@st.cache_resource
def initialize_model():
    download_model_from_url(MODEL_URL, MODEL_PATH)
    return load_model(MODEL_PATH)

model = initialize_model()

st.title("Brain Tumor Classification")
uploaded_file = st.file_uploader("Upload brain MRI image", type=["jpg", "jpeg", "png"])

classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

if uploaded_file is not None:
    image_tensor = preprocess_image(uploaded_file)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        pred_class = classes[predicted.item()]

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.success(f"Prediction: **{pred_class}**")
