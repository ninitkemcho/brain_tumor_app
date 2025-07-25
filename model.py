import torch
import torch.nn as nn
import torchvision.models as models


# Define the custom model class (same as used during training)
class BrainTumorResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def get_model():
    return BrainTumorResNet(num_classes=4)
