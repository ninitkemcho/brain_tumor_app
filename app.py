import streamlit as st
from PIL import Image
from utils import load_model, preprocess_image
import torch

st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

st.title("🧠 Brain Tumor Classification App")
st.write("Upload a brain MRI image. The model will predict the tumor type.")

CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
model = load_model("resnet18_brain_tumor.pth")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            st.success(f"**Prediction:** {CLASS_NAMES[predicted.item()]}")
