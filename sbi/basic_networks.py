mport numpy as np
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


class linear_with_batchnorm(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super(linear_with_batchnorm, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class linear_with_relu(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super(linear_with_relu, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        return out


class linear_with_dropout(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, probs:float):
        super(linear_with_dropout, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(p=probs)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class basic_network(nn.Module):
    def __init__(self, input_dim:int = 12 * 12, theta_dim:int = 3, num_classes:int = 1):
        super(basic_network, self).__init__()

        self.layer1 = linear_with_batchnorm(input_dim, 128)
        self.layer2 = linear_with_batchnorm(128, 64)
        self.layer3 = linear_with_batchnorm(64, 32)
        self.layer4 = linear_with_batchnorm(32, 16)

        self.fc = nn.Linear(16 + theta_dim, num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, theta):
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.cat((x, theta), dim=1)
        log_ratio = self.fc(x)
        logit = self.sigmoid(x)
        return log_ratio, logit
