import torch

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

def preprocess_image(image):
    from torchvision import transforms
    from PIL import Image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if isinstance(image, Image.Image):
        return transform(image).unsqueeze(0)
    return transform(Image.open(image)).unsqueeze(0)
