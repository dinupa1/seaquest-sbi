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

from simulators import reader
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

#
# inference model
#

# train events
out_tree = uproot.open("./data/outputs.root:out_tree")
X = out_tree["X"].array().to_numpy()
theta = out_tree["theta"].array().to_numpy()

out_tree = uproot.open("./data/RS67_LH2_hist.root:out_tree")
X_test = out_tree["X"].array().to_numpy()

X_train, X_val, theta_train, theta_val = train_test_split(X, theta, test_size=0.2, shuffle=True)

ds_train = ratio_dataset(X_train, theta_train)
ds_val = ratio_dataset(X_val, theta_val)

train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4)

model = basic_network().double().to(dvc)

optimizer = optim.Adam(model.parameters(), lr=0.0001, amsgrad=True)
criterion = nn.BCELoss()

tr = ratio_trainner(train_loader, val_loader, model, criterion, optimizer, device=dvc)
tr.fit()

#
# test data
#

tree = {"posterior": [],}

posterior = metropolis_hastings(model, X_test[0], num_samples=10000, proposal_std=0.001, device=dvc)
tree["posterior"].append(posterior)

outfile = uproot.recreate("./data/posterior_RS67_LH2_data.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile.close()
