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


def conv_with_dropout(input_channels:int, output_channel:int, kernel:int, stride:int):

    layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channel, kernel, stride, padding=1),
            nn.Dropout2d(p=0.3),
            nn.ReLU(),
        )

    return layers


class basic_network(nn.Module):
    def __init__(self, input_dim:int = 12 * 12, theta_dim:int = 9, num_classes:int = 1):
        super(basic_network, self).__init__()

        hidden_dim = 128

        self.layer1 = layers_with_relu(input_dim + theta_dim, hidden_dim)
        self.layer2 = layers_with_relu(hidden_dim, hidden_dim)
        self.layer3 = layers_with_relu(hidden_dim, hidden_dim)
        self.layer4 = layers_with_relu(hidden_dim, hidden_dim)
        self.layer5 = layers_with_relu(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, theta):
        x = torch.flatten(x, 1)
        x = torch.cat((x, theta), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        log_ratio = self.fc(x)
        logit = self.sigmoid(log_ratio)
        return log_ratio, logit



class inference_network(nn.Module):
    def __init__(self, theta_dim:int = 9, num_classes:int = 1):
        super(inference_network, self).__init__()

        hidden_dim = 128
        out_features = 32

        self.layer1 = conv_with_dropout(1, out_features, 4, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer2 = layers_with_relu(out_features+theta_dim, hidden_dim)
        self.layer3 = layers_with_relu(hidden_dim, hidden_dim)
        self.layer4 = layers_with_relu(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, theta):
        x = self.layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.cat((x, theta), dim=1)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        log_ratio = self.fc(x)
        logit = self.sigmoid(log_ratio)
        return log_ratio, logit

# m = inference_network()
# x = torch.randn(5, 1, 12, 12)
# theta = torch.randn(5, 12)
# print(m)
# total_trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
# print(f"total trainable params: {total_trainable_params}")
# print(m(x, theta))
