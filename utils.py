import torch
from torchvision import transforms
from PIL import Image

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = transform(image).unsqueeze(0)  # add batch dimension
    return image

def load_model(model_path):
    from model import BrainTumorResNet
    model = BrainTumorResNet(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
