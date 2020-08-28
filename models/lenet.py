import numpy as np
import torch
from torch import nn



class LeNet_300_100(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        # in_dim = 28
        super(LeNet5, self).__init__()
        self.in_dim = in_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim ** 2, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, out_dim)
        )

    def sample_input(self, device='cpu'):
        """
        input shape: [Batch_Size, num Channel, H, W]
        """
        batch_size = 4
        input_size = [self.in_dim, self.in_dim]
        num_channel = 1

        x = torch.rand([batch_size, num_channel] + input_size).to(device)

        return x

    def forward(self, x):
        x = self.classifier(x)
        return x


class LeNet5(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        # in_dim = 28
        super(LeNet5, self).__init__()
        self.in_dim = in_dim
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * (in_dim // 2) ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def sample_input(self, device='cpu'):
        """
        input shape: [Batch_Size, num Channel, H, W]
        """
        batch_size = 4
        input_size = [self.in_dim, self.in_dim]
        num_channel = 1

        x = torch.rand([batch_size, num_channel] + input_size).to(device)

        return x

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = LeNet5(28, 10)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    x = model.sample_input(device)

    try:
        print(device)
        y = model(x)
        print(x.shape, y.shape)
    except:
        print('[model.forward error]')
        print('features running...')
        x1 = model.features(x)
        print(x1.shape)
        print('features running...done')
        print('classifier running...')
        x2 = model.classifier(x1)
        print(x2.shape)
        print('classifier running...done')
