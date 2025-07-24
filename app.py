import streamlit as st
from PIL import Image
from utils import preprocess_image, download_model_from_url, load_model
import torch

MODEL_URL = "https://drive.google.com/file/d/1AhlepsKoDDDzxU-ROKmQvOKld-7NXNEK/view?usp=drive_link"  # replace with your direct link
MODEL_PATH = "resnet18_brain_tumor.pth"
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classifier")
st.write("Upload a brain MRI scan. The model will predict the tumor type.")

download_model_from_url(MODEL_URL, MODEL_PATH)
model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            st.success(f"**Prediction:** {CLASS_NAMES[pred.item()]}")
