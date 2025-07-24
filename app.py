import streamlit as st
import torch
from PIL import Image
from utils import preprocess_image, download_model_from_url, load_model
from model import get_model

MODEL_PATH = "resnet18_brain_tumor.pth"
MODEL_URL = "https://drive.google.com/uc?id=1cevX7wHrDpWUtV051nCd-UG2htvpJAiO&export=download"

@st.cache_resource
def initialize_model():
    try:
        download_model_from_url(MODEL_URL, MODEL_PATH)
        model = get_model()
        model = load_model(model, MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def main():
    st.title("🧠 Brain Tumor Classification")
    st.write("Upload a brain MRI scan image to classify tumor type:")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("Loading model..."):
                model = initialize_model()
            
            if model is None:
                st.error("Failed to load model. Please try again.")
                return
            
            with st.spinner("Processing image..."):
                input_tensor = preprocess_image(image)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = probabilities[0][prediction].item()
            
            class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
            
            st.write("### Results:")
            st.write(f"**Prediction:** {class_names[prediction]}")
            st.write(f"**Confidence:** {confidence:.2%}")
            
            # Show all probabilities
            st.write("### All Class Probabilities:")
            for i, (class_name, prob) in enumerate(zip(class_names, probabilities[0])):
                st.write(f"{class_name}: {prob.item():.2%}")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
