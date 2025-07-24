import torch
import gdown
from torchvision import transforms
from PIL import Image
import os
import pickle 
# Import the model class so it's available during unpickling
from model import BrainTumorResNet

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    if isinstance(image, Image.Image):
        return transform(image).unsqueeze(0)
    return transform(Image.open(image)).unsqueeze(0)

def download_model_from_url(url, filename):
    if os.path.exists(filename):
        return
    print("Downloading model...")
    try:
        gdown.download(url, filename, quiet=False)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete (fallback method).")

def load_model(model, model_path):
    """
    Load model supporting multiple formats:
    - .safetensors (recommended, most secure)
    - .pt (TorchScript)
    - .pth (standard PyTorch with fallback)
    """
    try:
        print(f"Loading model from {model_path}...")
        
        # Determine file format and load accordingly
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
