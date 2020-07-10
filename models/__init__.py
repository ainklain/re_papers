from . import lenet, vgg, resnet, convs
from functools import partial

model_dict = dict(
    lenet5=lenet.LeNet5,
    lenet300_100=lenet.LeNet_300_100,
    vgg11=vgg.vgg11,
    vgg13=vgg.vgg13,
    vgg16=vgg.vgg16,
    vgg19=vgg.vgg19,
)