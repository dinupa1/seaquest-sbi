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

from sbi import resnet_12x12
from sbi import basic_network
from sbi import ratio_dataset
from sbi import ratio_trainner
from sbi import metropolis_hastings

from simulators import sim_reader
from simulators import simulator

dvc = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {dvc} device")

seed: int = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size: int = 256
num_resamples: int = 100

#
# inference model
#

tree = {"score": [],}

# train events
out_tree = uproot.open("./data/outputs.root:out_tree")
X = out_tree["X"].array().to_numpy()
theta = out_tree["theta"].array().to_numpy()

X_train_val, X_test, theta_train_val, theta_test = train_test_split(X, theta, test_size=0.01, shuffle=True)

X_test = X_test[(np.abs(theta_test[:, 0]) < 1.) & (np.abs(theta_test[:, 1]) < 0.4) & (np.abs(theta_test[:, 2]) < 0.4)]
theta_test = theta_test[(np.abs(theta_test[:, 0]) < 1.) & (np.abs(theta_test[:, 1]) < 0.4) & (np.abs(theta_test[:, 2]) < 0.4)]

for _ in range(num_resamples):

    X_train_val1, theta_train_val1 = resample(X_train_val, theta_train_val)
    X_train, X_val, theta_train, theta_val = train_test_split(X_train_val1, theta_train_val1, test_size=0.2, shuffle=True)

    ds_train = ratio_dataset(X_train, theta_train)
    ds_val = ratio_dataset(X_val, theta_val)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    model = basic_network().double().to(dvc)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, amsgrad=True)
    criterion = nn.BCELoss()

    tr = ratio_trainner(train_loader, val_loader, model, criterion, optimizer, device=dvc)
    tr.fit()

    #
    # test data
    #

    theta_score = []

    for _ in range(50):
        posterior = metropolis_hastings(model, X_test[i], num_samples=10000, proposal_std=0.01, device=dvc)

        theta_mean = np.mean(posterior, axis=0)
        theta_std = np.std(posterior, axis=0)

        theta_score.append((theta_test - theta_mean)/theta_std)

    tree["score"].append(theta_score)


outfile = uproot.recreate("./data/eval.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile.close()
