"""ImageNet-pretrained CNN encoders that return a single pooled feature vector per image."""

import torch.nn as nn
from torchvision import models

FEATURE_DIMS = {"resnet18": 512, "mobilenet_v2": 1280}


class CNNEncoder(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        self.backbone = backbone
        if backbone == "resnet18":
            self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        elif backbone == "mobilenet_v2":
            self.cnn = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_dim = self.cnn.classifier[1].in_features
            self.cnn.classifier = nn.Identity()
        else:
            raise ValueError(f"unsupported backbone {backbone!r}; choose from {list(FEATURE_DIMS)}")

    def forward(self, images):
        return self.cnn(images)

    def freeze_all_but_last_block(self):
        for p in self.cnn.parameters():
            p.requires_grad = False
        last = self.cnn.layer4 if self.backbone == "resnet18" else self.cnn.features[-1]
        for p in last.parameters():
            p.requires_grad = True
