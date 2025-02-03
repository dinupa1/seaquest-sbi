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

        self.layer1 = layers_with_relu(input_dim + theta_dim, 100)
        self.layer2 = layers_with_relu(100, 100)
        self.layer3 = layers_with_relu(100, 100)
        self.fc = nn.Linear(100, num_classes, bias=True)
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



# m = basic_network()
# x = torch.randn(5, 1, 12, 12)
# theta = torch.randn(5, 3)
# print(m)
# total_trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
# print(f"total trainable params: {total_trainable_params}")
# print(m(x, theta))
