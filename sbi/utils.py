import numpy as np
import matplotlib.pyplot as plt

import copy

import torch
import torch.nn as nn


def cross_section_ratio(theta_0, theta_1, theta_2, phi, costh):
    weight = 1. + theta_0 * costh * costh + 2. * theta_1 * costh * np.sqrt(1. - costh * costh) * np.cos(phi) + 0.5 * theta_2 * (1. - costh * costh) * np.cos(2. * phi)
    return weight/(1. + costh * costh)


def mean_and_error(prior, weights):

    length = len(prior)

    w_sum = np.sum(weights)
    xmean = np.sum(prior * weights)/w_sum
    sigma2 = np.sum(weights * (prior - xmean) * (prior - xmean))/((length - 1)/length * w_sum)

    return xmean, np.sqrt(sigma2)


def metropolis_hastings(ratio_model, X, num_samples=10000, device=None):
    chain = []

    # Initialize the parameter vector
    theta_current = torch.tensor([0., 0., 0.]).unsqueeze(0).double().to(device)
    X_tensor = torch.from_numpy(X).unsqueeze(0).double().to(device)

    theta_min = torch.tensor([-1., -0.5, -0.5])
    theta_max = torch.tensor([1., 0.5, 0.5])

    ratio_model.eval()
    with torch.no_grad():
        for _ in range(num_samples):

            theta_proposal = theta_max + (theta_min - theta_max) * torch.rand(3)
            theta_proposal = theta_proposal.unsqueeze(0).double().to(device)

            log_r_current, _ = ratio_model(X_tensor, theta_current)
            log_r_proposal, _ = ratio_model(X_tensor, theta_proposal)

            threshold = np.random.uniform(0., 1.)

            if threshold <= min(1, torch.exp(log_r_proposal - log_r_current).item()):
                theta_current = theta_proposal


            chain.append(theta_current.cpu().numpy())

    return np.array(chain)
