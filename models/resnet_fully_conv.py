import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import torchvision

import json

import matplotlib.pyplot as plt


# ---------------------- convolution dims helper functions  ----------------------

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

def convtrans2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
    return outshape

# ----------------------    ----------------------


def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class ResidualBlock(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, downsample=None):

        super(ResidualBlock, self).__init__()

        self.in_chs = in_chs
        self.out_chs = out_chs
        self.kernel_size = kernel_size

        self.downsample = downsample

        self.conv1 = nn.Conv2d(self.in_chs, self.out_chs, kernel_size, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_chs)
        self.conv2 = nn.Conv2d(self.out_chs, self.out_chs, kernel_size, stride, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_chs)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        skip_residual = x
        if self.downsample:
            skip_residual = self.downsample(x)

        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))

        x = x + skip_residual
        out = self.relu(x)

        return out


def downsample(in_chs, out_chs, kernel_size, stride):
    downsample_op = nn.Sequential(
        nn.Conv2d(in_chs, out_chs, kernel_size, stride),
        nn.BatchNorm2d(out_chs)
    )
    return downsample_op


class ResNetAE(nn.Module):

    def __init__(self):
        super(ResNetAE, self).__init__()
        self.in_chs = 12

        self.conv = nn.Conv2d(3, 12, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(12)
        self.relu = nn.ReLU(inplace=True)

        # encoder - residual blocks
        '''
        if (stride != 1) or (self.in_chs != out_chs):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_chs, out_chs, 3, stride),
                nn.BatchNorm2d(out_chs))
        '''

        self.res_block1 = ResidualBlock(12, 24, 4, 2, downsample(12, 24, 5, 3))
        self.res_block2 = ResidualBlock(24, 48, 3, 1, downsample(24, 48, 1, 1))
        self.res_block3 = ResidualBlock(48, 96, 3, 2, downsample(48, 96, 3, 2))


        # encoder - FC layers
        self.fc_encoder = nn.Sequential(
            nn.Linear(192, 128),
            nn.BatchNorm1d(128, momentum=0.01),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=0.01),
            nn.ReLU(),
        )

        # decoder - FC layers
        self.fc_decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128, momentum=0.01),
            nn.ReLU(),
            nn.Linear(128, 192),
            nn.BatchNorm1d(192, momentum=0.01),
            nn.ReLU(),
        )

        self.conv_decoder = nn.Sequential(
            #nn.ConvTranspose2d(192, 96, 4, stride=1, padding=1),  # [batch, 96, 2, 2]
            #nn.ReLU(),
            nn.ConvTranspose2d(96, 72, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(72, 48, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def encoder_res_layer(self, out_chs, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_chs != out_chs):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_chs, out_chs, 3, stride),
                nn.BatchNorm2d(out_chs))
        layers = []
        layers.append(ResidualBlock(self.in_chs, out_chs, 2, stride, downsample))
        self.in_chs = out_chs
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_chs, out_chs, 2, stride))
        return nn.Sequential(*layers)

    def encode(self, x):
        out = self.conv(x)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):

        out = self.conv(x)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)

        out = out.view(out.size(0), -1)

        feat = out

        #out =  self.fc_decoder(feat)

        out = out.view(-1, 96, 1, 1)
        out = self.conv_decoder(out)



        return out
    
    def forward_latent(self, x):

        out = self.conv(x)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)

        out = out.view(out.size(0), -1)

        feat = out

        #out =  self.fc_decoder(feat)

        out = out.view(-1, 96, 1, 1)
        out = self.conv_decoder(out)

        return feat
