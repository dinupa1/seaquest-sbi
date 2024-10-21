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


class ratio_dataset(Dataset):
    def __init__(self, X, theta1, theta0):
        super(ratio_dataset, self).__init__()
        self.X = np.concatenate([X, X])
        self.theta = np.concatenate([theta1, theta0])
        self.label = np.concatenate([np.ones((len(X), 1)), np.zeros((len(X), 1))])
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.theta[idx], self.label[idx]
