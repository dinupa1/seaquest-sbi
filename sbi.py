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

from sbi import resnet_10x10
from sbi import ratio_dataset
from sbi import ratio_trainner
from sbi import test_ratio_model
from sbi import mean_and_error
from sbi import metropolis_hastings

from simulators import sim_reader
from simulators import simulator

from plots import plots_reader
from plots import ratio_plots

dvc = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {dvc} device")

seed: int = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size: int = 1024


#
# inference model
#

# train events
train_tree = uproot.open("./data/outputs.root:train_tree")
X_train = train_tree["X"].array().to_numpy()
theta_train = train_tree["theta"].array().to_numpy()

test_tree = uproot.open("./data/outputs.root:test_tree")
X_test = test_tree["X"].array().to_numpy()
theta_test = test_tree["theta"].array().to_numpy()

ds_ratio = ratio_dataset(X_train, theta_train)

ds_train, ds_val = random_split(ds_ratio, [0.8, 0.2])

train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

model = resnet_10x10().double().to(dvc)
optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
criterion = nn.BCELoss()

tr = ratio_trainner(train_loader, val_loader, model, criterion, optimizer, device=dvc)
tr.fit()
tr.val_auc()
tr.save()

#
# test data
#

tree = {
        "theta": [],
        "meas": [],
        "error": [],
    }

trees = {
        "theta": [],
        "posterior": [],
    }

for i in range(2):
    posterior = metropolis_hastings(model, X_test[i], num_samples=10000, proposal_std=0.1, device=dvc)
    tree["theta"].append(theta_test[i])
    tree["meas"].append(np.mean(posterior, axis=0))
    tree["error"].append(np.std(posterior, axis=0))

    if i < 5:
        trees["theta"].append(theta_test[i])
        trees["posterior"].append(posterior)

    print(f"[===> {i+1} tests are done]")


outfile = uproot.recreate("./data/eval.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile["trees"] = trees
outfile.close()
