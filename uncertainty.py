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

batch_size: int = 10000
proposal_std: float = 0.01
learning_rate: float = 0.0001
num_samples: int = 10000

data_input_name = "RS67_LH2_hist_pT_0"
data_output_name = "systematics_RS67_LH2_data_pT_0"
MC_output_name = "systematics_LH2_messy_MC_pT_0"

#
# inference model
#
tree = {"posterior": [], "mean": [], "std": [], "theta": []}
history = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}

# train events
train_tree = uproot.open("./data/outputs.root:train_tree")
X_train = train_tree["X"].array().to_numpy()
theta_train = train_tree["theta"].array().to_numpy()

val_tree = uproot.open("./data/outputs.root:val_tree")
X_val = val_tree["X"].array().to_numpy()
theta_val = val_tree["theta"].array().to_numpy()

test_tree = uproot.open("./data/outputs.root:test_tree")
X_test = test_tree["X"].array().to_numpy()
theta_test = test_tree["theta"].array().to_numpy()

RS67_LH2_tree = uproot.open(f"./data/{data_input_name}.root:out_tree")
X_RS67_LH2 = RS67_LH2_tree["X"].array().to_numpy()

LH2_tree = {
        "posterior": [],
        "mean": [],
        "std": [],
    }

for i in range(num_resamples):

    print(f"[====> Iteration {i+1}]")

    X_resample, theta_resample = resample(X_train, theta_train)

    ds_train = ratio_dataset(X_resample, theta_resample)
    ds_val = ratio_dataset(X_val, theta_val)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4)

    model = basic_network().double().to(dvc)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, amsgrad=True)
    criterion = nn.BCELoss()

    tr = ratio_trainner(train_loader, val_loader, model, criterion, optimizer, device=dvc)
    tr.fit()

    history["train_loss"].append(tr.fit_history["train_loss"])
    history["train_auc"].append(tr.fit_history["train_auc"])
    history["val_loss"].append(tr.fit_history["val_loss"])
    history["val_auc"].append(tr.fit_history["val_auc"])

    posterior = metropolis_hastings(model, X_test[2], num_samples=10000, proposal_std=0.001, device=dvc)

    tree["posterior"].append(posterior)
    tree["mean"].append(np.mean(posterior, axis=0))
    tree["std"].append(np.std(posterior, axis=0))
    tree["theta"].append(theta_test[2])

    posterior = metropolis_hastings(model, X_RS67_LH2[0], num_samples=10000, proposal_std=0.001, device=dvc)

    LH2_tree["posterior"].append(posterior)
    LH2_tree["mean"].append(np.mean(posterior, axis=0))
    LH2_tree["std"].append(np.std(posterior, axis=0))


outfile = uproot.recreate(f"./data/{MC_output_name}.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile["history"] = history
outfile.close()

outfile = uproot.recreate(f"./data/{data_output_name}.root", compression=uproot.ZLIB(4))
outfile["tree"] = LH2_tree
outfile.close()
