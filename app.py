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

def get_survival_rate_and_treatment(tumor_type, age, sex, tumor_size, symptoms_duration, karnofsky_score, previous_surgery):
    """Calculate survival rate and treatment recommendations based on patient data and tumor type"""
    
    # Base survival rates by tumor type (5-year survival rates)
    base_survival_rates = {
        "Glioma": 35,
        "Meningioma": 85,
        "No Tumor": 100,
        "Pituitary": 95
    }
    
    survival_rate = base_survival_rates.get(tumor_type, 50)
    treatment_recommendations = []
    
    if tumor_type == "No Tumor":
        return 100, ["Regular monitoring", "Healthy lifestyle maintenance", "Annual MRI screening if symptoms persist"]
    
    # Age adjustments
    if age < 40:
        survival_rate += 15
    elif age > 65:
        survival_rate -= 20
    
    # Size adjustments
    if tumor_size == "Small (<3cm)":
        survival_rate += 10
    elif tumor_size == "Large (>5cm)":
        survival_rate -= 15
    
    # Karnofsky score adjustments
    if karnofsky_score >= 80:
        survival_rate += 10
    elif karnofsky_score < 60:
        survival_rate -= 15
    
    # Ensure survival rate stays within reasonable bounds
    survival_rate = max(5, min(95, survival_rate))
    
    # Treatment recommendations based on tumor type
    if tumor_type == "Glioma":
        if tumor_size == "Large (>5cm)" or karnofsky_score < 70:
            treatment_recommendations = [
                "🏥 Surgical resection (maximal safe resection)",
                "⚡ Radiation therapy (60 Gy in 30 fractions)",
                "💊 Temozolomide chemotherapy",
                "🔬 Molecular testing (IDH, MGMT status)",
                "👥 Multidisciplinary team consultation"
            ]
        else:
            treatment_recommendations = [
                "🏥 Surgical biopsy or resection",
                "⚡ Concurrent chemoradiation",
                "💊 Temozolomide maintenance",
                "🔬 Genetic profiling for targeted therapy",
                "🧠 Neurocognitive monitoring"
            ]
    
    elif tumor_type == "Meningioma":
        if tumor_size == "Small (<3cm)" and age < 65:
            treatment_recommendations = [
                "👁️ Active surveillance with serial MRI",
                "🏥 Surgical resection if symptomatic",
                "⚡ Stereotactic radiosurgery option",
                "💊 Symptom management (anti-seizure meds if needed)"
            ]
        else:
            treatment_recommendations = [
                "🏥 Complete surgical resection (Simpson Grade I-II)",
                "⚡ Adjuvant radiation if incomplete resection",
                "🔬 Histological grading assessment",
                "👁️ Long-term follow-up imaging"
            ]
    
    elif tumor_type == "Pituitary":
        if previous_surgery == "Yes":
            treatment_recommendations = [
                "🔬 Comprehensive hormonal evaluation",
                "⚡ Stereotactic radiosurgery",
                "💊 Medical management (dopamine agonists/somatostatin analogs)",
                "👁️ Visual field monitoring",
                "🧪 Endocrine replacement therapy as needed"
            ]
        else:
            treatment_recommendations = [
                "🏥 Transsphenoidal surgical resection",
                "🧪 Pre and post-operative hormonal assessment",
                "👁️ Ophthalmologic evaluation",
                "💊 Medical therapy for hormone-secreting tumors",
                "📅 Regular endocrine follow-up"
            ]
    
    return survival_rate, treatment_recommendations

def main():
    st.set_page_config(page_title="Brain Tumor Classification", layout="wide", page_icon="🧠")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .survival-box {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .treatment-box {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Classification & Treatment Advisor</h1>', unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📊 Image Classification")
        st.write("Upload a brain MRI scan image to classify tumor type:")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        prediction_result = None
        confidence_result = None
        probabilities_result = None
        
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
                prediction_result = class_names[prediction]
                confidence_result = confidence
                probabilities_result = probabilities[0]
                
                # Display results in styled boxes
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>🎯 Prediction: {prediction_result}</h3>
                    <h4>📊 Confidence: {confidence:.2%}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Show all probabilities
                st.write("### 📈 All Class Probabilities:")
                for i, (class_name, prob) in enumerate(zip(class_names, probabilities[0])):
                    st.progress(prob.item(), text=f"{class_name}: {prob.item():.2%}")
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.error("Please check if the uploaded file is a valid image.")
    
    with col2:
        st.header("👤 Patient Information")
        
        # Patient information form
        with st.form("patient_form"):
            sex = st.selectbox("Sex", ["Male", "Female"])
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            tumor_size = st.selectbox("Tumor Size", ["Small (<3cm)", "Medium (3-5cm)", "Large (>5cm)"])
            symptoms_duration = st.selectbox("Symptoms Duration", 
                                           ["< 1 month", "1-6 months", "6-12 months", "> 1 year"])
            karnofsky_score = st.slider("Karnofsky Performance Score", 
                                      min_value=10, max_value=100, value=80, step=10,
                                      help="Functional status scale (100 = normal, no complaints)")
            previous_surgery = st.selectbox("Previous Brain Surgery", ["No", "Yes"])
            family_history = st.selectbox("Family History of Brain Tumors", ["No", "Yes"])
            seizure_history = st.selectbox("History of Seizures", ["No", "Yes"])
            headache_severity = st.slider("Headache Severity (0-10)", 
                                        min_value=0, max_value=10, value=3)
            
            submitted = st.form_submit_button("Calculate Survival Rate & Treatment Plan", 
                                            use_container_width=True)
        
        if submitted and prediction_result:
            # Calculate survival rate and treatment
            survival_rate, treatments = get_survival_rate_and_treatment(
                prediction_result, age, sex, tumor_size, symptoms_duration, 
                karnofsky_score, previous_surgery
            )
            
            # Display survival rate
            st.markdown(f"""
            <div class="survival-box">
                <h3>📊 Estimated 5-Year Survival Rate</h3>
                <h2>{survival_rate}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for survival rate
            st.progress(survival_rate/100, text=f"Survival Rate: {survival_rate}%")
            
            # Treatment recommendations
            st.markdown("""
            <div class="treatment-box">
                <h3>💊 Treatment Recommendations</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for treatment in treatments:
                st.write(f"• {treatment}")
            
            # Additional information box
            st.info("""
            ⚠️ **Important Disclaimer**: 
            This tool provides general information based on medical literature and should not replace professional medical advice. 
            Always consult with qualified healthcare professionals for personalized diagnosis and treatment plans.
            """)
            
            # Risk factors summary
            with st.expander("🔍 Risk Factors Analysis"):
                st.write("**Favorable factors:**")
                if age < 40:
                    st.write("• Young age")
                if karnofsky_score >= 80:
                    st.write("• Good functional status")
                if tumor_size == "Small (<3cm)":
                    st.write("• Small tumor size")
                
                st.write("**Risk factors:**")
                if age > 65:
                    st.write("• Advanced age")
                if karnofsky_score < 70:
                    st.write("• Reduced functional status")
                if tumor_size == "Large (>5cm)":
                    st.write("• Large tumor size")
                if family_history == "Yes":
                    st.write("• Family history of brain tumors")
        
        elif submitted and not prediction_result:
            st.warning("Please upload and analyze an image first to get treatment recommendations.")

if __name__ == "__main__":
    main()
