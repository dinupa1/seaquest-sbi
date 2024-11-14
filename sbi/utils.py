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



def metropolis_hastings(ratio_model, X, num_samples=10000, proposal_std=0.1, device=None):
    chain = []

    theta_current = torch.tensor([0., 0., 0.]).double().to(device)
    X_tensor = torch.from_numpy(X).double().to(device)

    ratio_model.eval()
    with torch.no_grad():
        for _ in range(num_samples):

            theta_proposal = torch.randn(3).double().to(device) * torch.tensor([proposal_std, proposal_std, proposal_std]).double().to(device) + theta_current

            if (theta_proposal[0] < -1.5 or 1.5 < theta_proposal[0] or theta_proposal[1] < -0.6 or 0.6 < theta_proposal[1] or theta_proposal[2] < -0.6 or 0.6 < theta_proposal[2]):
                chain.append(theta_current.cpu().numpy())
                continue

            log_r_current, _ = ratio_model(X_tensor.unsqueeze(0), theta_current.unsqueeze(0))
            log_r_proposal, _ = ratio_model(X_tensor.unsqueeze(0), theta_proposal.unsqueeze(0))

            threshold = np.random.uniform(0., 1.)

            if threshold <= min(1, torch.exp(log_r_proposal - log_r_current).item()):
                theta_current = theta_proposal


            chain.append(theta_current.cpu().numpy())

    return np.array(chain)
