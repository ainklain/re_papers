
import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, utils



class MaskedConv(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv, self).__init__(*args, **kwargs)
        assert mask_type in ['A', 'B']
        self.register_buffer('mask', torch.ones_like(self.weight))

        _, c, h, w = self.weight.size()
        if mask_type == 'A':
            self.mask[:, :, h// 2, w // 2:] = 0
            self.mask[:, :, h // 2 + 1:, :] = 0
        else:
            self.mask[:, :, h // 2, w // 2 + 1:] = 0
            self.mask[:, :, h // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv, self).forward(x)


class PixelCNN(nn.Module):
    def __init__(self, n_layers=8, kernel=7, hidden_dim=64, device=None):
        super(PixelCNN, self).__init__()
        self.n_layers = n_layers
        self.kernel = kernel
        self.hidden_dim = hidden_dim
        self.device = device

        modules = []
        for i in range(self.n_layers):
            if i == 0:
                mask_type = 'A'
                in_channel = 1
            else:
                mask_type = 'B'
                in_channel = hidden_dim
            out_channel = hidden_dim

            modules.append(
                nn.Sequential(
                    MaskedConv(mask_type, in_channel, out_channel, kernel, stride=1, padding=kernel // 2, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.LeakyReLU(),
                ))
            self.encoder = nn.Sequential(*modules)
            self.out_layer = nn.Conv2d(hidden_dim, 256, 1)

    def forward(self, x):
        x = self.encoder(x)
        return self.out_layer(x)


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=True, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    model = PixelCNN()
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(10):
        print(epoch)
        model.train()
        for data, _ in train_loader:
            data = data.to('cuda')
            pred = model(data)
            loss = loss_func(pred, data.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sample = torch.zeros(10, 1, 28, 28).to('cuda')
        model.eval()
        for i in range(28):
            for j in range(28):
                out = model(sample)
                probs = torch.softmax(out[:, :, i, j], 0).data
                sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.
        utils.save_image(sample, 'sample_{}.png'.format(epoch), nrow=12, padding=0)





