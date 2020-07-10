import torch
from torch import nn


class ConvModel(nn.Module):
    def __init__(self, in_dim):
        # in_dim = 28
        super(ConvModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * (in_dim // 2) ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
