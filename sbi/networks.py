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


class residual_block(nn.Module):
    def __init__(self, planes: int = 16):
        super(residual_block, self).__init__()

        self.expantion = 2

        self.conv_1 = nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes),
            )
        self.relu_2 = nn.ReLU()
        self.conv_3 = nn.Sequential(
                nn.Conv2d(planes, planes * self.expantion, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(planes * self.expantion),

            )
        self.relu_3 = nn.ReLU()
        self.downsample = nn.Sequential(
                nn.Conv2d(planes, planes * self.expantion, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes * self.expantion),
            )

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.conv_2(out)
        out = self.relu_2(out)
        out = self.conv_3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu_3(out)
        return out


class ratio_net(nn.Module):
    def __init__(self, input_dim:int = 1, theta_dim:int = 3):
        super(ratio_net, self).__init__()

        self.planes:int = 5
        self.hidded_dim:int = 50
        self.output_dim:int = 1

        self.layer_0 = nn.Sequential(
                nn.Conv2d(input_dim, self.planes, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.planes),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.layer_1 = nn.Sequential(
                residual_block(self.planes),
                residual_block(2 * self.planes),
                nn.MaxPool2d(kernel_size=2, stride=2),
                residual_block(4 * self.planes),
                residual_block(8 * self.planes),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )
        self.layer_2 = nn.Sequential(
                nn.Linear(16 * self.planes * 1 * 1 + theta_dim, self.hidded_dim, bias=True),
                nn.BatchNorm1d(self.hidded_dim),
                nn.ReLU(),
                nn.Linear(self.hidded_dim, self.hidded_dim, bias=True),
                nn.BatchNorm1d(self.hidded_dim),
                nn.ReLU(),
                nn.Linear(self.hidded_dim, self.hidded_dim, bias=True),
                nn.BatchNorm1d(self.hidded_dim),
                nn.ReLU(),
                nn.Linear(self.hidded_dim, 1, bias=True),
                nn.Sigmoid(),
            )

    def forward(self, x, theta):
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, theta], dim=1)
        logit = self.layer_2(x)
        return logit



# m = ratio_net()
# x = torch.randn(5, 1, 10, 10)
# theta = torch.randn(5, 3)
# print(m)
# total_trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
# print(f"total trainable params: {total_trainable_params}")
# print(m(x, theta))
