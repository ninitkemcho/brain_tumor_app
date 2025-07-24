import os
import torch
import gdown  # install with pip if you don’t have it
from torchvision import transforms
from PIL import Image
from model import BrainTumorResNet18

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image).convert("RGB")
    return transform(image).unsqueeze(0)

def download_model_from_url(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading model from {url}...")
        gdown.download(url, output_path, quiet=False)

def load_model(model_path):
    model = BrainTumorResNet18(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
