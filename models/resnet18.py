"""
Models used in classifier.
"""

import torch.nn as nn
from torchvision.models import resnet18


resnet_model = resnet18(pretrained=True)

print(resnet_model)

resnet_model.fc = nn.Linear(512,6)
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
