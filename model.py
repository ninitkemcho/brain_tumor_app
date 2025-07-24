import torch.nn as nn
from torchvision.models import resnet18

class BrainTumorResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorResNet18, self).__init__()
        self.model = resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
