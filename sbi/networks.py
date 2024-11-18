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


class basic_net(nn.Module):
    def __init__(self, mean, std, input_dim:int = 12 * 12, theta_dim:int = 3, num_classes:int = 1):
        super(basic_net, self).__init__()

        self.fc1 = nn.Linear(input_dim + theta_dim, 256, bias=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256, bias=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128, bias=True)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, num_classes, bias=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)
        self.flatten = nn.Flatten()

        self.mean = mean
        self.std = std

    def forward(self, x, theta):

        x = self.flatten(x)
        theta = (theta - self.mean)/self.std
        x = torch.cat((x, theta), dim=1)

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)

        log_ratio = self.fc4(out)
        logit = self.sigmoid(log_ratio)

        return log_ratio, logit


class basic_block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(basic_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class resnet(nn.Module):
    def __init__(self, block, layers, num_classes=1, theta_dim=3):
        super(resnet, self).__init__()
        self.in_channels = 32

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64 * block.expansion + theta_dim, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, 32, bias=True)
        self.fc4 = nn.Linear(32, num_classes, bias=True)
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = torch.cat((out, theta), dim=1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        log_ratio = self.fc4(out)
        logit = self.sigmoid(log_ratio)
        return log_ratio, logit


def resnet_12x12(num_classes:int = 1, theta_dim:int = 3):
    layers = [2, 2]
    return resnet(basic_block, layers, num_classes, theta_dim)


# m = basic_net()
# x = torch.randn(5, 1, 12, 12)
# theta = torch.randn(5, 3)
# print(m)
# total_trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
# print(f"total trainable params: {total_trainable_params}")
# print(m(x, theta))
