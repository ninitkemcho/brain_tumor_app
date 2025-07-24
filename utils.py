import torch
import gdown
from torchvision import transforms
from PIL import Image
import os

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Add the same normalization as training
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
        # Fallback to requests if gdown fails
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete (fallback method).")

def load_model(model, model_path):
    try:
        # Try with weights_only=True first (safer), fallback to weights_only=False if needed
        print("Loading model...")
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            print("Model loaded with weights_only=True (secure mode)")
        except Exception as weights_only_error:
            print(f"weights_only=True failed: {weights_only_error}")
            print("Falling back to weights_only=False...")
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
            print("Model loaded with weights_only=False")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
