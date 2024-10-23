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


class ratio_10x10(nn.Module):
    def __init__(self):
        super(ratio_10x10, self).__init__()

        self.cnn_head = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.linear_head = nn.Sequential(
            nn.Linear(64 * 2 * 2 + 3, 64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )

        self.output_layer = nn.Linear(32, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def log_ratio(self, x, theta):
        x = self.cnn_head(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, theta], dim=1)
        x = self.linear_head(x)
        return self.output_layer(x)

    def forward(self, x, theta):
        ratio = self.log_ratio(x, theta)
        logit = self.sigmoid(ratio)
        return ratio, logit
