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


def layers_with_relu(input_dim:int, output_dim:int):

    layers = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
        )

    return layers


def layers_with_batchnorm(input_dim:int, output_dim:int):

    layers = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    return layers


class basic_network(nn.Module):
    def __init__(self, input_dim:int = 12 * 12, theta_dim:int = 3, num_classes:int = 1):
        super(basic_network, self).__init__()

        self.layer1 = layers_with_relu(input_dim + theta_dim, 128)
        self.layer2 = layers_with_relu(128, 128)
        self.layer3 = layers_with_relu(128, 128)
        self.fc = nn.Linear(128, num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, theta):
        x = torch.flatten(x, 1)
        x = torch.cat((x, theta), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        log_ratio = self.fc(x)
        logit = self.sigmoid(log_ratio)
        return log_ratio, logit


class Ratio_Network(nn.Module):
    def __init__(self, input_channels:int, theta_dim:int, num_classes:int):
        super(Ratio_Network, self).__init__()

        self.conv_head = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
            )

        self.mlp_head = nn.Sequential(
                nn.Linear(4* 2* 2 + 3, 128, bias=True),
                nn.ReLU(),
                nn.Linear(128, 128, bias=True),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, theta):

        out = self.conv_head(inputs)
        out = torch.flatten(out, 1)
        out = torch.cat([out, theta], dim=1)
        log_ratio = self.mlp_head(out)
        logit = self.sigmoid(log_ratio)

        return log_ratio, logit


def ratio_net12x12(input_channels:int = 1, theta_dim:int = 3, num_classes:int = 1):
    return Ratio_Network(input_channels, theta_dim, num_classes)
