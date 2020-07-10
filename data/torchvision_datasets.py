import torch
import torchvision


def get_datasets(name, train=True, transform=None):
    assert name in ['MNIST', 'FashionMNIST', 'CIFAR10']
    dataset_module = getattr(torchvision.datasets, name)
    return dataset_module('./data/', download=True, train=train, transform=transform)


def get_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == '__main__':
    batch_size = 16
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5,), (0.5,))])

    mnist_train = get_datasets('MNIST', train=True, transform=transform)
    dataloader = get_dataloader(mnist_train, batch_size=batch_size)
    x = next(iter(dataloader))
    print('image_shape: {} / label_shape: {}'.format(x[0].shape, x[1].shape))



