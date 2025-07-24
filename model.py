import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # for grayscale
    model.fc = nn.Linear(model.fc.in_features, 4)  # 4 output classes
    return model
