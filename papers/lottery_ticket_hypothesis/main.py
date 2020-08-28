
import torch
from torch import nn, optim
from torchvision import transforms

from data import get_datasets, get_dataloader
from models import model_dict
from utils import torch_utils as tu


class Configs:
    def __init__(self, dataset_nm='mnist', model_nm='lenet300_100'):
        # dataset
        self.set_dataset_nm(dataset_nm)

        # model
        self.set_model_nm(model_nm)

    def set_dataset_nm(self, name):
        if name.lower() == 'mnist':
            self.dataset_nm = 'MNIST'
            self.in_dim = 28
            self.out_dim = 10

    def set_model_nm(self, name):
        self.model_nm = name
        if name.lower() == 'lenet300_100':
            # optimizer
            self.optim = 'adam'
            self.lr = 1.2e-3
            self.weight_decay = 1e-4

            # training
            self.num_iters = 50e3
            self.batch_size = 60

            # pruning
            self.features_pruning_r = 0.0
            self.classifier_pruning_r = 0.2

        elif name.lower() == 'conv2':
            # optimizer
            self.optim = 'adam'
            self.optim_cfg = dict(lr=2e-4, weight_decay=1e-4)

            # training
            self.num_iters = 20e3
            self.batch_size = 60

            # pruning
            self.features_pruning_r = 0.1
            self.classifier_pruning_r = 0.2

        elif name.lower() == 'conv4':
            # optimizer
            self.optim = 'adam'
            self.scheduler = None
            self.optim_cfg = dict(lr=3e-4, weight_decay=1e-4)

            # training
            self.num_iters = 25e3
            self.batch_size = 60

            # pruning
            self.features_pruning_r = 0.1
            self.classifier_pruning_r = 0.2

        elif name.lower() == 'conv6':
            # optimizer
            self.optim = 'adam'
            self.scheduler = None
            self.optim_cfg = dict(lr=3e-4, weight_decay=1e-4)

            # training
            self.num_iters = 30e3
            self.batch_size = 60

            # pruning
            self.features_pruning_r = 0.15
            self.classifier_pruning_r = 0.2

        elif name.lower() == 'resnet18':
            # optimizer
            self.optim = 'sgd'
            self.scheduler = [0.1, 0.01, 0.001]
            self.optim_cfg = dict(momentum=0.9)

            # training
            self.num_iters = 30e3
            self.batch_size = 128

            # pruning
            self.features_pruning_r = 0.2
            self.classifier_pruning_r = 0.0

        elif name.lower() == 'vgg19':
            # optimizer
            self.optim = 'sgd'
            self.scheduler = [0.1, 0.01, 0.001]
            self.optim_cfg = dict(momentum=0.9)

            # training
            self.num_iters = 112e3
            self.batch_size = 64

            # pruning
            self.features_pruning_r = 0.2
            self.classifier_pruning_r = 0.0

    def __repr__(self):
            return_str = "[configs]\n"
            for key in self.__dict__.keys():
                return_str += "{}: {}\n".format(key, self.__dict__[key])

            return return_str


def init_weights(model):
    for name, param in model.named_parameters():



def main(c):
    # dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    train_dataset = get_datasets(c.dataset_nm, train=True, transform=transform)
    train_dataloader = get_dataloader(train_dataset, batch_size=c.batch_size)
    test_dataset = get_datasets(c.dataset_nm, train=False, transform=transform)
    test_dataloader = get_dataloader(test_dataset, batch_size=c.batch_size)

    # model
    model = model_dict[c.model_nm](c.in_dim, c.out_dim).to(tu.device)

    # loss func and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), **c.optim_cfg)
    if c.scheduler is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, )

    # pruning
    model_params_init = model.state_dict()






if __name__ == '__main__':
    dataset_nm = 'mnist'
    model_nm = 'lenet300_100'

    c = Configs(dataset_nm, model_nm)

    # main(c)