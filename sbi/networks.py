import numpy as np
import matplotlib.pyplot as plt

import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

import uproot
import awkward as ak

from sklearn.covariance import log_likelihood
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

class basic_block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(basic_block, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class resnet(nn.Module):
    def __init__(self, block, layers, num_classes=1, theta_dim=3):
        super(resnet, self).__init__()
        self.in_channels = 8

        self.embedding = nn.Linear(8 * 8 + theta_dim, 3 * 32 * 32, bias=True)
        self.emb_sigmoid = nn.Sigmoid()

        self.conv_1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)

        self.layer_1 = self._make_layer(block, 8, layers[0], stride=1)
        self.layer_2 = self._make_layer(block, 16, layers[1], stride=2)
        self.layer_3 = self._make_layer(block, 32, layers[2], stride=2)
        self.layer_4 = self._make_layer(block, 64, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, theta):
        out = torch.flatten(x, 1)
        out = torch.cat([out, theta], dim=1)
        out = self.embedding(out)
        out = self.emb_sigmoid(out)

        out = out.view(-1, 3, 32, 32)
        out = self.conv_1(out)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        log_r = self.fc(out)
        logit = self.sigmoid(log_r)
        return log_r, logit


def resnet_8x8():
    layers = [2, 2, 2, 2]
    return resnet(basic_block, layers)


# m = resnet_8x8()
# x = torch.randn(5, 1, 8, 8)
# theta = torch.randn(5, 3)
# print(m)
# total_trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
# print(f"total trainable params: {total_trainable_params}")
# print(m(x, theta))
