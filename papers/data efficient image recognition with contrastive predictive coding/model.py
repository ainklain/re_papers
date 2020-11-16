

import os
import numpy as np
import time
from PIL import Image

import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_downsampling=False):
        super(ResBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.is_downsampling = is_downsampling

        


class ResEncoder(nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()

