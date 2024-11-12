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
        self.in_channels = 16

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer_1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer_2 = self._make_layer(block, 64, layers[1], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
                nn.Linear(64 * block.expansion + theta_dim, 128, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes, bias=True),
            )
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
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.layer_1(out)
        out = self.layer_2(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = torch.cat([out, theta], dim=1)
        log_r = self.fc(out)
        logit = self.sigmoid(log_r)
        return log_r, logit


def resnet_10x10():
    layers = [2, 2]
    return resnet(basic_block, layers)



class ratio_net(nn.Module):
    def __init__(self, input_dim: int=100, theta_dim: int=3, num_classes: int=1):
        super(ratio_net, self).__init__()

        self.fc_1 = nn.Linear(input_dim + theta_dim, 256, bias=True)
        self.bn_1 = nn.BatchNorm1d(256)

        self.fc_2 = nn.Linear(256, 128, bias=True)
        self.bn_2 = nn.BatchNorm1d(128)

        self.fc_3 = nn.Linear(128, 64, bias=True)
        self.bn_3 = nn.BatchNorm1d(64)

        self.fc_4 = nn.Linear(64, num_classes, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, theta):
        x = torch.flatten(x[:, 0, :, :], 1)
        x = torch.cat([x, theta], dim=1)
        x = self.bn_1(self.fc_1(x))
        x = self.relu(x)
        x = self.bn_2(self.fc_2(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn_3(self.fc_3(x))
        x = self.relu(x)
        log_r = self.fc_4(x)
        logit = self.sigmoid(log_r)
        return log_r, logit


# m = ratio_net()
# x = torch.randn(5, 3, 10, 10)
# theta = torch.randn(5, 3)
# print(m)
# total_trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
# print(f"total trainable params: {total_trainable_params}")
# print(m(x, theta))
