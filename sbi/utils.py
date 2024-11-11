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

    # Set prior bounds
    prior = Uniform(torch.tensor([-1., -0.5, -0.5], device=device), torch.tensor([1., 0.5, 0.5], device=device))

    # Initialize the parameter vector
    theta_current = torch.tensor([0., 0., 0.], device=device)
    X_tensor = torch.from_numpy(X).to(device).double()

    ratio_model.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            # Proposal distribution centered around current theta
            proposal = MultivariateNormal(theta_current, proposal_std * torch.eye(3, device=device))
            theta_proposal = proposal.sample()

            # Check if proposal falls within the prior bounds
            if torch.all(torch.abs(theta_proposal) <= torch.tensor([1.0, 0.5, 0.5], device=device)):
                # Calculate log ratio estimations for current and proposal
                log_r_current, _ = ratio_model(X_tensor.unsqueeze(0), theta_current.unsqueeze(0))
                log_prior_current = prior.log_prob(theta_current).sum()

                log_r_proposal, _ = ratio_model(X_tensor.unsqueeze(0), theta_proposal.unsqueeze(0))
                log_prior_proposal = prior.log_prob(theta_proposal).sum()

                # Calculate acceptance probability
                acceptance_ratio = (log_prior_proposal + log_r_proposal) - (log_prior_current + log_r_current)
                acceptance_probability = min(1, torch.exp(acceptance_ratio).item())

                # Accept or reject the proposal
                if torch.rand(1).item() <= acceptance_probability:
                    theta_current = theta_proposal

            # Append current theta to the chain
            chain.append(theta_current.cpu().numpy())

    return np.array(chain)
