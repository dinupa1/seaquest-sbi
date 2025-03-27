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


class RatioNetwork(nn.Module):
    def __init__(self, input_dim:int = 12* 12, theta_dim:int = 3, hidden_dim:int = 64, num_layers:int = 5, num_classes:int = 1):
        super(RatioNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim+theta_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()

        layers = []

        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            layers.append(nn.ReLU())

        self.hidden_fcs = nn.Sequential(*layers)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, theta):
        log_r = self.log_ratio(x, theta)
        logit = self.sigmoid(log_r)
        return logit

    def log_ratio(self, x, theta):
        out = torch.flatten(x, 1)
        out = torch.cat((out, theta), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.hidden_fcs(out)
        log_r = self.fc2(out)
        return log_r

    @torch.no_grad()
    def samples(self, X, num_samples=10000, proposal_std=0.001, device=None):
        # this function used Metropolis-Hastings algorithm for sampling posterior probability
        chain = torch.zeros((num_samples, 3)).double()

        theta_current = torch.zeros((3)).double().to(device)
        log_r_current = self.log_ratio(X.unsqueeze(0), theta_current.unsqueeze(0))

        for i in range(num_samples):

            theta_proposal = np.random.multivariate_normal(theta_current.cpu().numpy(), proposal_std * np.eye(3))
            theta_proposal = torch.from_numpy(theta_proposal).double().to(device)

            if(theta_proposal[0] < -2.0 or 2.0 < theta_proposal[0] or
               theta_proposal[1] < -0.8 or 0.8 < theta_proposal[1] or
               theta_proposal[2] < -0.8 or 0.8 < theta_proposal[2]):
                chain[i, :] = theta_current
                continue

            log_r_proposal = self.log_ratio(X.unsqueeze(0), theta_proposal.unsqueeze(0))

            log_lambda = log_r_proposal - log_r_current

            threshold = np.random.uniform(0., 1.)

            if threshold < min(1, torch.exp(log_lambda).item()):
                theta_current = theta_proposal
                log_r_current = log_r_proposal

            chain[i, :] = theta_current

        return chain

class RatioDataset(Dataset):
    def __init__(self, X, theta):
        super(RatioDataset, self).__init__()

        self.X = X
        self.theta = theta

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.theta[idx]


def backpropagation(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_one_epoch(model, data_loader, criterion, optimizer, epoch, size, device=None):
    num_iterations = len(data_loader)//2
    loader = iter(data_loader)
    for batch in range(num_iterations):
        x_a, theta_a = next(loader)
        x_a, theta_a = x_a.double().to(device, non_blocking=True), theta_a.double().to(device, non_blocking=True)

        x_b, theta_b = next(loader)
        x_b, theta_b = x_b.double().to(device, non_blocking=True), theta_b.double().to(device, non_blocking=True)

        logit_dep_a = model(x_a, theta_a)
        logit_ind_a = model(x_b, theta_a)

        logit_dep_b = model(x_b, theta_b)
        logit_ind_b = model(x_a, theta_b)

        ones = torch.ones([len(theta_a), 1]).double().to(device, non_blocking=True)
        zeros = torch.zeros([len(theta_a), 1]).double().to(device, non_blocking=True)

        loss_a = criterion(logit_dep_a, ones) + criterion(logit_ind_a, zeros)
        loss_b = criterion(logit_dep_b, ones) + criterion(logit_ind_b, zeros)

        loss = loss_a + loss_b

        # Backpropagation
        backpropagation(optimizer, loss)

        loss, current = loss.item(), (batch + 1) * len(x_a)
        print("\r" + f"[Epoch {epoch:>3d}] [{current:>5d}/{size:>5d}] [Train_loss: {loss:>5f}]", end="")


@torch.no_grad()
def val_one_epoch(model, data_loader, criterion, device=None):
    num_iterations = len(data_loader)//2
    loader = iter(data_loader)
    loss, auc = 0, 0
    for batch in range(num_iterations):

        x_a, theta_a = next(loader)
        x_a, theta_a = x_a.double().to(device, non_blocking=True), theta_a.double().to(device, non_blocking=True)

        x_b, theta_b = next(loader)
        x_b, theta_b = x_b.double().to(device, non_blocking=True), theta_b.double().to(device, non_blocking=True)

        logit_dep_a = model(x_a, theta_a)
        logit_ind_a = model(x_b, theta_a)

        logit_dep_b = model(x_b, theta_b)
        logit_ind_b = model(x_a, theta_b)

        ones = torch.ones([len(theta_a), 1]).double().to(device, non_blocking=True)
        zeros = torch.zeros([len(theta_a), 1]).double().to(device, non_blocking=True)

        loss_a = criterion(logit_dep_a, ones) + criterion(logit_ind_a, zeros)
        loss_b = criterion(logit_dep_b, ones) + criterion(logit_ind_b, zeros)

        auc_a = roc_auc_score(torch.cat([ones, zeros]).cpu().numpy().reshape(-1), torch.cat([logit_dep_a, logit_ind_a]).cpu().numpy().reshape(-1))
        auc_b = roc_auc_score(torch.cat([ones, zeros]).cpu().numpy().reshape(-1), torch.cat([logit_dep_b, logit_ind_b]).cpu().numpy().reshape(-1))

        loss += loss_a + loss_b
        auc += auc_a + auc_b

    return loss.item()/num_iterations, auc/(2.* num_iterations)


@torch.no_grad()
def model_performance(model, data_loader, device=None):
    num_iterations = len(data_loader)//2
    loader = iter(data_loader)
    labels, logits = None, None
    for batch in range(num_iterations):
        x_a, theta_a = next(loader)
        x_a, theta_a = x_a.double().to(device, non_blocking=True), theta_a.double().to(device, non_blocking=True)

        x_b, theta_b = next(loader)
        x_b, theta_b = x_b.double().to(device, non_blocking=True), theta_b.double().to(device, non_blocking=True)

        logit_dep_a = model(x_a, theta_a)
        logit_ind_a = model(x_b, theta_a)
        logit_dep_b = model(x_b, theta_b)
        logit_ind_b = model(x_a, theta_b)

        ones = torch.ones([len(theta_a), 1]).double().to(device, non_blocking=True)
        zeros = torch.zeros([len(theta_a), 1]).double().to(device, non_blocking=True)

        labels = torch.cat([labels, ones]) if labels is not None else ones
        labels = torch.cat([labels, zeros]) if labels is not None else zeros
        labels = torch.cat([labels, ones]) if labels is not None else ones
        labels = torch.cat([labels, zeros]) if labels is not None else zeros

        logits = torch.cat([logits, logit_dep_a]) if logits is not None else logit_dep_a
        logits = torch.cat([logits, logit_ind_a]) if logits is not None else logit_ind_a
        logits = torch.cat([logits, logit_dep_b]) if logits is not None else logit_dep_b
        logits = torch.cat([logits, logit_ind_b]) if logits is not None else logit_ind_b

    return logits, labels
