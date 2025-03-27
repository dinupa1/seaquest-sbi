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
from sklearn.metrics import confusion_matrix


class ReweightingNetwork(nn.Module):
    def __init__(self, input_dim:int = 3, hidden_dim:int = 32, num_layers:int = 3, num_classes:int = 1):
        super(ReweightingNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()

        layers = []

        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            layers.append(nn.ReLU())

        self.hidden_fcs = nn.Sequential(*layers)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        log_r = self.log_ratio(x)
        logit = self.sigmoid(log_r)
        return logit

    def log_ratio(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.hidden_fcs(out)
        log_r = self.fc2(out)
        return log_r

    @torch.no_grad()
    def event_reweighting(self, data_loader, device=None):
        event_weights = None
        for batch, data in enumerate(data_loader):

            inputs, weights, labels = data[0].double().to(device, non_blocking=True), data[1].double().to(device, non_blocking=True), data[2].double().to(device, non_blocking=True)

            log_r = self.log_ratio(inputs)

            event_weights = torch.cat([event_weights, log_r]) if event_weights is not None else log_r

        return torch.exp(event_weights).cpu().numpy().reshape(-1)

    def backpropagation(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train_one_epoch(self, data_loader, criterion, optimizer, epoch, size, device):

        for batch, data in enumerate(data_loader):

            inputs, weights, labels = data[0].double().to(device, non_blocking=True), data[1].double().to(device, non_blocking=True), data[2].double().to(device, non_blocking=True)

            logits = self.forward(inputs)

            loss = criterion(logits, labels, weights)

            # Backpropagation
            self.backpropagation(optimizer, loss)

            loss, current = loss.item(), (batch + 1) * len(inputs)
            print("\r" + f"[Epoch {epoch:>3d}] [{current:>5d}/{size:>5d}] [Train_loss: {loss:>5f}]", end="")


    @torch.no_grad()
    def val_one_epoch(self, data_loader, criterion, device):
        num_iterations = len(data_loader)
        running_loss, running_auc = 0, 0
        for batch, data in enumerate(data_loader):

            inputs, weights, labels = data[0].double().to(device, non_blocking=True), data[1].double().to(device, non_blocking=True), data[2].double().to(device, non_blocking=True)

            logits = self.forward(inputs)

            loss = criterion(logits, labels, weights)

            auc = roc_auc_score(labels.detach().cpu().numpy().ravel(), logits.detach().cpu().numpy().ravel(), sample_weight=weights.detach().cpu().numpy().ravel())

            running_loss += loss.item()
            running_auc += auc

        return running_loss/num_iterations, running_auc/num_iterations



class ReweightingDataset(Dataset):
    def __init__(self, X, weights, labels):
        super(ReweightingDataset, self).__init__()

        self.X = X
        self.weights = weights
        self.labels = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.weights[idx], self.labels[idx]


class ReweightingLoss(nn.Module):
    def __init__(self):
        super(ReweightingLoss, self).__init__()

        self.bce_loss = nn.BCELoss(reduction="none")

    def forward(self, outputs, targets, weights):
        loss = self.bce_loss(outputs, targets)
        weighted_loss = loss * weights
        return weighted_loss.mean()
