import numpy as np
import matplotlib.pyplot as plt

import copy

import torch
from torch.distributions.uniform import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal


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

    prior = Uniform(torch.tensor([-1., -0.5, -0.5]), torch.tensor([1., 0.5, 0.5]))

    theta_current = prior.sample()
    X_tensor = torch.from_numpy(X)

    log_r_current, logit = ratio_model(X_tensor.unsqueeze(0).double().to(device), theta_current.unsqueeze(0).double().to(device))
    log_prior_current = prior.log_prob(theta_current)

    ratio_model.eval()
    with torch.no_grad():
        for i in range(num_samples):

            proposal = MultivariateNormal(theta_current, proposal_std * torch.eye(3))

            theta_proposal = proposal.sample()

            log_r_proposal, logit = ratio_model(X_tensor.unsqueeze(0).double().to(device), theta_proposal.unsqueeze(0).double().to(device))
            log_prior_proposal = prior.log_prob(theta_proposal)

            acceptance_ratio = (log_prior_proposal + log_r_proposal) - (log_prior_current + log_r_current)
            acceptance_probability = min([1, torch.exp(acceptance_ratio).item()])

            if np.random.uniform() <= acceptance_probability:
                theta_current = theta_proposal
                log_r_current = log_r_proposal
                log_prior_current = log_prior_proposal


            chain.append([theta_current[0].item(), theta_current[1].item(), theta_current[2].item()])

    return np.array(chain)
