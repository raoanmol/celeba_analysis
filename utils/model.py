import torch.nn as nn
import torchvision.models as models

def create_resnet18(num_classes=1, pretrained=True):
    model = models.resnet18(pretrained=pretrained)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
