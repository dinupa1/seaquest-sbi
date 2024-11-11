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


def cross_section_ratio(theta_0, theta_1, theta_2, phi, costh):
    weight = 1. + theta_0 * costh * costh + 2. * theta_1 * costh * np.sqrt(1. - costh * costh) * np.cos(phi) + 0.5 * theta_2 * (1. - costh * costh) * np.cos(2. * phi)
    return weight/(1. + costh * costh)


def mean_and_error(prior, weights):

    length = len(prior)

    w_sum = np.sum(weights)
    xmean = np.sum(prior * weights)/w_sum
    sigma2 = np.sum(weights * (prior - xmean) * (prior - xmean))/((length - 1)/length * w_sum)

    return xmean, np.sqrt(sigma2)


def metropolis_hastings(ratio_model, X, num_samples=10000, proposal_std=0.1, device=None):

    samples = []

    theta_0_current = np.random.uniform(-1., 1.)
    theta_1_current = np.random.uniform(-0.5, 0.5)
    theta_2_current = np.random.uniform(-0.5, 0.5)

    theta_current = torch.tensor([theta_0_current, theta_1_current, theta_2_current]).unsqueeze(0).double().to(device)
    X_tensor = torch.from_numpy(X).unsqueeze(0).double().to(device)

    ratio_model.eval()
    with torch.no_grad():
        for i in range(num_samples):

            theta_0_proposal = np.random.normal(theta_0_current, proposal_std)
            theta_1_proposal = np.random.normal(theta_1_current, proposal_std)
            theta_2_proposal = np.random.normal(theta_2_current, proposal_std)

            theta_proposal = torch.tensor([theta_0_proposal, theta_1_proposal, theta_2_proposal]).unsqueeze(0).double().to(device)

            ratio_current, logit = ratio_model(X_tensor, theta_current)
            ratio_proposal, logit = ratio_model(X_tensor, theta_proposal)

            acceptance_ratio = ratio_proposal/ratio_current

            if np.random.rand() < acceptance_ratio:
                theta_0_current = theta_0_proposal
                theta_1_current = theta_1_proposal
                theta_2_current = theta_2_proposal

            samples.append([theta_0_current, theta_1_current, theta_2_current])

    return np.array(samples)
