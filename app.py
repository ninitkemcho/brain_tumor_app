import streamlit as st
import torch
from PIL import Image
import sys
import os

# Add current directory to Python path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import preprocess_image, download_model_from_url, load_model
from model import get_model, BrainTumorResNet

MODEL_PATH = "model.pkl"
MODEL_URL = "https://drive.google.com/uc?id=1oXRJHDblcAjD3yBDCW9DvIW9QQ_jqVKt&export=download"

@st.cache_resource
def initialize_model():
    try:
        # Download model if needed
        download_model_from_url(MODEL_URL, MODEL_PATH)
        
        # Create model instance
        model = get_model()
        
        # Load the trained weights
        model = load_model(model, MODEL_PATH)
        return model
        
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        st.error("This usually happens when the model file format doesn't match the expected format.")
        st.info("Please ensure your model was saved properly during training.")
        return None

def main():
    st.title("🧠 Brain Tumor Classification")
    st.write("Upload a brain MRI scan image to classify tumor type:")
    
    # Add some debug information
    with st.expander("Debug Information"):
        st.write(f"Model file exists: {os.path.exists(MODEL_PATH)}")
        if os.path.exists(MODEL_PATH):
            st.write(f"Model file size: {os.path.getsize(MODEL_PATH)} bytes")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with st.spinner("Loading model..."):
                model = initialize_model()
            
            if model is None:
                st.error("Failed to load model. Please check the debug information above.")
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
            st.error("Please check if the uploaded file is a valid image.")

if __name__ == "__main__":
    main()
