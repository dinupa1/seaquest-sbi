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

from sbi import ratio_10x10
from sbi import ratio_dataset
from sbi import ratio_trainner
from sbi import test_ratio_model
from sbi import mean_and_error

from simulator import generator
from simulator import forward_simulation

dvc = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {dvc} device")


batch_size = 1000

num_events = np.array([50000, 100000, 200000, 500000, 1000000])

lambda_mean, lambda_error = [], []
mu_mean, mu_error = [], []
nu_mean, nu_error = [], []

for events in num_events:

    # generate events
    generator()

    # forward simulation
    forward_simulation(0, int(events), 15000)

    #
    # inference model
    #

    # train events
    train_tree = uproot.open("./data/outputs.root:train_tree")
    X = train_tree["X"].array().to_numpy()
    theta = train_tree["theta"].array().to_numpy()

    N_sim = len(theta)

    lambdas = np.random.uniform(-1, 1., N_sim)
    mus = np.random.uniform(-0.5, 0.5, N_sim)
    nus = np.random.uniform(-0.5, 0.5, N_sim)

    theta0 = np.array([(lam, mu, nu) for lam, mu, nu in zip(lambdas, mus, nus)])

    ratio_ds = ratio_dataset(X, theta, theta0)
    ratio_ds_train, ratio_ds_val = random_split(ratio_ds, [0.5, 0.5])

    train_loader = DataLoader(ratio_ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(ratio_ds_val, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ratio_10x10().double().to(dvc)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    tr = ratio_trainner(train_loader, val_loader, model, criterion, optimizer, device=dvc)
    tr.fit()

    #
    # test data
    #

    test_tree = uproot.open("./data/outputs.root:test_tree")
    X_test = test_tree["X"].array().to_numpy()
    theta_test = test_tree["theta"].array().to_numpy()

    N_samples = 20000

    lambda_prior = np.random.uniform(-1., 1., N_samples)
    mu_prior = np.random.uniform(-0.5, 0.5, N_samples)
    nu_prior = np.random.uniform(-0.5, 0.5, N_samples)


    theta_prior = torch.tensor([(lam, mu, nu) for lam, mu, nu in zip(lambda_prior, mu_prior, nu_prior)]).double().to(dvc)

    lambda_mean2, lambda_error2 = [], []
    mu_mean2, mu_error2 = [], []
    nu_mean2, nu_error2 = [], []

    for i in range(len(theta_test)):
        X_test_array = np.array([X_test[i] for j in range(N_samples)])
        X_test_tensor = torch.from_numpy(X_test_array).double().to(dvc)

        log_ratio = test_ratio_model(model, X_test_tensor, theta_prior, batch_size=batch_size, device=dvc)

        lambda1, lambda2 = mean_and_error(lambda_prior, log_ratio)
        lambda_mean2.append(lambda1)
        lambda_error2.append(lambda2)

        mu1, mu2 = mean_and_error(mu_prior, log_ratio)
        mu_mean2.append(mu1)
        mu_error2.append(mu2)

        nu1, nu2 = mean_and_error(nu_prior, log_ratio)
        nu_mean2.append(nu1)
        nu_error2.append(nu2)

    lambda_mean3 = (theta_test[:, 0] - np.array(lambda_mean2))/np.array(lambda_error2)
    lambda_mean.append(np.mean(lambda_mean3))
    lambda_error.append(np.std(lambda_mean3))

    mu_mean3 = (theta_test[:, 1] - np.array(mu_mean2))/np.array(mu_error2)
    mu_mean.append(np.mean(mu_mean3))
    mu_error.append(np.std(mu_mean3))

    nu_mean3 = (theta_test[:, 2] - np.array(nu_mean2))/np.array(nu_error2)
    nu_mean.append(np.mean(nu_mean3))
    nu_error.append(np.std(nu_mean3))


outfile = uproot.recreate("./data/hyperparameter0.root", compression=uproot.ZLIB(4))
outfile["tree"] = {
    "lambda_mean": np.array(lambda_mean),
    "mu_mean": np.array(mu_mean),
    "nu_mean": np.array(nu_mean),
    "lambda_error": np.array(lambda_error),
    "mu_error": np.array(mu_error),
    "nu_error": np.array(nu_error),
    "train_size" : num_events,
}
outfile.close()
