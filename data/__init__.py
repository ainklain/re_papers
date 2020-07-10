from . import torchvision_datasets

def get_datasets(name, train, transform):
    if name in ['MNIST', 'FashionMNIST', 'CIFAR10']:
        return torchvision_datasets.get_datasets(name, train, transform)
    else:
        return False


def get_dataloader(dataset, *args, **kwargs):
    return torchvision_datasets.get_dataloader(dataset, *args, **kwargs)
