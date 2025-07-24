import torch
import requests
from torchvision import transforms
from PIL import Image
import os

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if isinstance(image, Image.Image):
        return transform(image).unsqueeze(0)
    return transform(Image.open(image)).unsqueeze(0)

def download_model_from_url(url, filename):
    if os.path.exists(filename):
        return
    print("Downloading model...")
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    print("Download complete.")

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
