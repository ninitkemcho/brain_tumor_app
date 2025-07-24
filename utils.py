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
        # Load with weights_only=True for security and compatibility
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try loading without weights_only if the above fails
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e2:
            print(f"Secondary loading attempt failed: {e2}")
            raise e2
