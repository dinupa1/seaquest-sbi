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


class conv_net(nn.Module):
    def __init__(self, in_channels: int=1, out_channels: int=32):
        super(conv_net, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        ) # out = (size + 2)/2

    def forward(self, x):
        return self.net(x)


class batch_net(nn.Module):
    def __init__(self, in_features: int=1, out_features: int=32):
        super(batch_net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, out_features, bias=True),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ratio_net(nn.Module):
    def __init__(self, in_channels: int=1, theta_dim: int=3):
        super(ratio_net, self).__init__()

        self.feature_extraction = nn.Sequential(
            conv_net(in_channels, 64),
            conv_net(64, 32),
            conv_net(32, 16),
            conv_net(16, 8)
        )

        self.log_ratio = nn.Sequential(
            batch_net(8 * 2 * 2 + theta_dim, 64),
            nn.Dropout(p=0.1),
            batch_net(64, 32),
            nn.Linear(32, 1, bias=True),
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x, theta):
        x = self.feature_extraction(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, theta], dim=1)
        x = self.log_ratio(x)
        logit = self.sigmoid(x)
        return logit
