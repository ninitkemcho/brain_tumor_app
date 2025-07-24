import torch
from torchvision import transforms
from PIL import Image
import os
import requests

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0)

def download_model_from_url(url, filename):
    if not os.path.exists(filename):
        print("Downloading model...")
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)
        print("Download complete.")

def load_model(model_path):
    from model import BrainTumorResNet
    model = BrainTumorResNet(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))

    model.eval()
    return model
