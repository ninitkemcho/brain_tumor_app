import streamlit as st
import torch
from PIL import Image
from utils import preprocess_image, download_model_from_url, load_model
from model import get_model

MODEL_PATH = "resnet18_brain_tumor.pth"
MODEL_URL = "https://drive.google.com/uc?id=1WSLzGt6yejXThEjdZbz9F4kf29mTmVzt&export=download"

@st.cache_resource
def initialize_model():
    download_model_from_url(MODEL_URL, MODEL_PATH)
    model = get_model()
    model = load_model(model, MODEL_PATH)
    return model

st.title("Brain Tumor Classification")
st.write("Upload a brain MRI scan image:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = initialize_model()
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    st.write(f"### Prediction: {class_names[prediction]}")
