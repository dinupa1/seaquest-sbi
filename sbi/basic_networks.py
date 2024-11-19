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


def linear_layers(input_dim:int, output_dim:int, batchnorm:bool, sigmoid:bool, dropout:float):

    layers = nn.Sequential()

    layers.add_module("linear", nn.Linear(input_dim, output_dim, bias=True))
    if batchnorm:
        layers.add_module("batchnorm", nn.BatchNorm1d(output_dim))
    if sigmoid:
        layers.add_module("sigmoid", nn.Sigmoid())
    else:
        layers.add_module("relu", nn.ReLU())
    if dropout > 0.:
        layers.add_module("dropout", nn.Dropout(p=dropout))

    return layers


class basic_network(nn.Module):
    def __init__(self, input_dim:int = 12 * 12, theta_dim:int = 3, num_classes:int = 1):
        super(basic_network, self).__init__()

        self.layer1 = linear_layers(input_dim, 128, batchnorm=True, sigmoid=False, dropout=0.)
        self.layer2 = linear_layers(128, 64, batchnorm=True, sigmoid=False, dropout=0.)
        self.layer3 = linear_layers(64, 32, batchnorm=True, sigmoid=False, dropout=0.)
        self.layer4 = linear_layers(32, 16, batchnorm=True, sigmoid=False, dropout=0.)

        self.layer5 = linear_layers(16 + theta_dim, 32, batchnorm=False, sigmoid=False, dropout=0.)
        self.layer6 = linear_layers(32, 32, batchnorm=False, sigmoid=False, dropout=0.)
        self.layer7 = linear_layers(32, 32, batchnorm=False, sigmoid=False, dropout=0.)

        self.fc = nn.Linear(32, num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, theta):
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.cat((x, theta), dim=1)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

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
