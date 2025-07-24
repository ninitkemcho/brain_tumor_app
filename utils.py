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
        if model_path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
                model.load_state_dict(state_dict)
                print("Model loaded using SafeTensors format")
            except ImportError:
                raise ImportError("SafeTensors not installed. Run: pip install safetensors")
                
        elif model_path.endswith('.pt'):
            # TorchScript format
            model = torch.jit.load(model_path, map_location='cpu')
            print("Model loaded using TorchScript format")
            
        else:
            # Standard .pth format with fallback
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
