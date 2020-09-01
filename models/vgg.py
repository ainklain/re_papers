
import torch
from torch import nn


def vgg11(in_dim, out_dim):
    return VGG(11, in_dim, out_dim)


def vgg13(in_dim,out_dim):
    return VGG(13, in_dim, out_dim)


def vgg16(in_dim,out_dim):
    return VGG(16, in_dim, out_dim)


def vgg19(in_dim, out_dim):
    return VGG(19, in_dim, out_dim)


class VGG(nn.Module):
    def __init__(self, depth: int, in_dim: int, out_dim: int):
        super(VGG, self).__init__()

        self.vgg_model, last_hidden_dim = self.make_graph(depth)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_hidden_dim, out_dim)

    def sample_input(self, device='cpu'):
        """
        input shape: [Batch_Size, num Channel, H, W]
        """
        batch_size = 64
        input_size = [224, 224]
        num_channel = 3

        x = torch.rand([batch_size, num_channel] + input_size).to(device)

        return x

    def forward(self, x):
        x = self.vgg_model(x)
        x = self.global_average_pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def _get_conv_structure(self, depth):
        conv_structure = {11: [64, 'max', 128, 'max', 256, 256, 'max', 512, 512, 'max', 512, 512],
                                13: [64, 64, 'max', 128, 128, 'max', 256, 256, 'max', 512, 512, 'max', 512, 512],
                                16: [64, 64, 'max', 128, 128, 'max', 256, 256, 256, 'max', 512, 512, 512, 'max', 512, 512, 512],
                                19: [64, 64, 'max', 128, 128, 'max', 256, 256, 256, 256, 'max', 512, 512, 512, 512, 'max', 512, 512, 512, 512],
                            }

        return conv_structure[depth]

    def make_graph(self, depth):
        conv_structure = self._get_conv_structure(depth)
        conv_graph = []
        in_channels = 3
        for l in conv_structure:
            if l == 'max':
                conv_graph.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv_graph.append(nn.Conv2d(in_channels, l, kernel_size=3, padding=1, bias=False))
                in_channels = l
                conv_graph.append(nn.ReLU())

        return nn.Sequential(*conv_graph), conv_structure[-1]


if __name__ == '__main__':
    model = VGG(11, 10)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    x = model.sample_input(device)

    try:
        print(device)
        y = model(x)
        print(x.shape, y.shape)
    except:
        print('[model.forward error]')
        print('vgg_model running...')
        x1 = model.vgg_model(x)
        print(x1.shape)
        print('vgg_model running...done')
        print('gap running...')
        x2 = model.global_average_pooling(x1)
        print(x2.shape)
        print('gap running...done')
        print('classifier running...')
        x3 = model.classifier(x2)
        print(x3.shape)
        print('classifier running...done')