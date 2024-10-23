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

N_sim = 200000
batch_size = 128
N_samples = 20000
N_data = 15000

tree = {
    "lambda_true": [],
    "mu_true": [],
    "nu_true": [],
    "lambda_mean": [],
    "mu_mean": [],
    "nu_mean": [],
    "lambda_error": [],
    "mu_error": [],
    "nu_error": [],
}


# generate events
generator()

# forward simulation
forward_simulation(0, N_sim, N_data)


#
# inference model
#

# train events
train_tree = uproot.open("./data/outputs.root:train_tree")
X = train_tree["X"].array().to_numpy()
theta = train_tree["theta"].array().to_numpy()

lambdas = np.random.uniform(-1, 1., N_sim)
mus = np.random.uniform(-0.5, 0.5, N_sim)
nus = np.random.uniform(-0.5, 0.5, N_sim)

theta0 = np.array([(lam, mu, nu) for lam, mu, nu in zip(lambdas, mus, nus)])

ratio_ds = ratio_dataset(X, theta, theta0)
ratio_ds_train, ratio_ds_val = random_split(ratio_ds, [0.5, 0.5])

train_loader = DataLoader(ratio_ds_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ratio_ds_val, batch_size=batch_size, shuffle=False)

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

lambda_prior = np.random.uniform(-1., 1., N_samples)
mu_prior = np.random.uniform(-0.5, 0.5, N_samples)
nu_prior = np.random.uniform(-0.5, 0.5, N_samples)

theta_prior = torch.tensor([(lam, mu, nu) for lam, mu, nu in zip(lambda_prior, mu_prior, nu_prior)]).double().to(dvc)

for i in range(len(theta_test)):
    X_test_array = np.array([X_test[i] for j in range(N_samples)])
    X_test_tensor = torch.from_numpy(X_test_array).double().to(dvc)

    log_ratio = test_ratio_model(model, X_test_tensor, theta_prior, batch_size=batch_size, device=dvc)

    lambda1, lambda2 = mean_and_error(lambda_prior, log_ratio)
    tree["lambda_mean"].append(lambda1)
    tree["lambda_error"].append(lambda2)

    mu1, mu2 = mean_and_error(mu_prior, log_ratio)
    tree["mu_mean"].append(mu1)
    tree["mu_error"].append(mu2)

    nu1, nu2 = mean_and_error(nu_prior, log_ratio)
    tree["nu_mean"].append(nu1)
    tree["nu_error"].append(nu2)

    tree["lambda_true"].append(theta_test[i , 0])
    tree["mu_true"].append(theta_test[i , 1])
    tree["nu_true"].append(theta_test[i , 2])


outfile = uproot.recreate("./data/eval.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile.close()
