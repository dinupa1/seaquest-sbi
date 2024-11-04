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

from sbi import ratio_net
from sbi import ratio_dataset
from sbi import ratio_trainner
from sbi import test_ratio_model
from sbi import mean_and_error

dvc = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {dvc} device")

seed: int = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size: int = 1000

tree = {
    "lambda_true": [],
    "mu_true": [],
    "nu_true": [],
    "weights": [],
    }

# forward simulation
forward_simulation(seed)


#
# inference model
#

# train events
train_tree = uproot.open("./data/outputs.root:train_tree")
X_train = train_tree["X"].array().to_numpy()
theta_train = train_tree["theta"].array().to_numpy()
theta0_train = train_tree["theta0"].array().to_numpy()

val_tree = uproot.open("./data/outputs.root:val_tree")
X_val = val_tree["X"].array().to_numpy()
theta_val = val_tree["theta"].array().to_numpy()
theta0_val = val_tree["theta0"].array().to_numpy()

test_tree = uproot.open("./data/outputs.root:test_tree")
X_test = test_tree["X"].array().to_numpy()
theta_test = test_tree["theta"].array().to_numpy()
theta0_test = test_tree["theta0"].array().to_numpy()

ds_train = ratio_dataset(X_train, theta_train, theta0_train)
ds_val = ratio_dataset(X_val, theta_val, theta0_val)

train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

model = ratio_net().double().to(dvc)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

tr = ratio_trainner(train_loader, val_loader, model, criterion, optimizer, device=dvc)
tr.fit()

#
# test data
#

for i in range(len(theta_test)):
    X_test_array = np.array([X_test[i] for j in range(len(theta0_test))])
    tree["lambda_true"].append(theta_test[i, 0])
    tree["mu_true"].append(theta_test[i, 1])
    tree["nu_true"].append(theta_test[i, 2])
    tree["weights"].append(test_ratio_model(model, X_test_array, theta0_test, batch_size=500, device=dvc))

    if i%1000 == 0:
        print(f"[===> {i} tests are done ]")

outfile = uproot.recreate("./data/eval.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile["priors"] = {
    "theta": theta0_test,
    }
outfile.close()
