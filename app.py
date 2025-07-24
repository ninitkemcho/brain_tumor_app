import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# --------------- Google Drive Download ----------------
def download_model_from_drive(file_id, output_path):
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# --------------- Model Definition ----------------
class BrainTumorResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorResNet, self).__init__()
        from torchvision import models
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

@st.cache_resource
def load_model():
    model_path = "resnet18_brain_tumor.pth"
    drive_file_id = "1WSLzGt6yejXThEjdZbz9F4kf29mTmVzt"  # <-- REPLACE with your actual file ID

    if not os.path.exists(model_path):
        st.warning("Model not found locally. Downloading from Google Drive...")
        download_model_from_drive(drive_file_id, model_path)


    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load("resnet18_brain_tumor.pth", weights_only=True))
    model.eval()

    return model

model = load_model()
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# ---------------- Image Preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ---------------- App ----------------
st.title("🧠 Brain Tumor Detection & Survival Estimator")

uploaded_file = st.file_uploader("Upload an MRI Scan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = class_names[predicted.item()]
    
    st.subheader("🧪 Tumor Classification Result")
    st.success(f"Prediction: *{pred_class}*")

    if pred_class != "No Tumor":
        st.subheader("📋 Patient Information for Survival Estimation")
        age = st.slider("Age", 1, 100, 50)
        sex = st.radio("Sex", ["Male", "Female"])
        size_mm = st.slider("Estimated Tumor Size (mm)", 0, 100, 30)
        location = st.selectbox("Tumor Location", ["Frontal", "Temporal", "Parietal", "Occipital", "Other"])

        survival = 0.0
        treatment = []

        if pred_class == "Glioma Tumor":
            survival = 0.6
            treatment = ["Surgery", "Radiation", "Chemotherapy"]
            if age > 60 or size_mm > 40:
                survival -= 0.2
        elif pred_class == "Meningioma Tumor":
            survival = 0.85
            treatment = ["Surgery", "Observation"]
            if size_mm > 50:
                survival -= 0.1
        elif pred_class == "Pituitary Tumor":
            survival = 0.9
            treatment = ["Medication", "Surgery"]
            if age > 70:
                survival -= 0.1

        survival = max(0, min(survival, 1))

        st.subheader("🧬 Estimated Survival Probability")
        st.info(f"Estimated survival rate: *{int(survival * 100)}%*")

        st.subheader("💊 Suggested Treatment Options")
        st.write(", ".join(treatment))

        st.caption("Note: This is a rule-based, educational approximation. Not for clinical use.")
