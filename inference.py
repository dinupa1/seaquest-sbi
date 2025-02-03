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
from sklearn.metrics import confusion_matrix

from sbi import resnet_12x12
from sbi import basic_network
from sbi import ratio_dataset
from sbi import ratio_trainner
from sbi import metropolis_hastings

dvc = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {dvc} device")

seed: int = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size: int = 10000
proposal_std: float = 0.0005
learning_rate: float = 0.0001
num_samples: int = 2

#
# inference model
#

# train events
out_tree = uproot.open("./data/outputs.root:out_tree")
X = out_tree["X"].array().to_numpy()
theta = out_tree["theta"].array().to_numpy()

X_train_val, X_test, theta_train_val, theta_test = train_test_split(X, theta, test_size=0.1, shuffle=True)
X_train, X_val, theta_train, theta_val = train_test_split(X_train_val, theta_train_val, test_size=0.2, shuffle=True)

ds_train = ratio_dataset(X_train, theta_train)
ds_val = ratio_dataset(X_val, theta_val)
ds_test = ratio_dataset(X_test, theta_test)

train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)

model = basic_network().double().to(dvc)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
criterion = nn.BCELoss()

tr = ratio_trainner(train_loader, val_loader, model, criterion, optimizer, device=dvc)
fit_history = tr.fit()

#
# test samples
#
labels, logits = tr.prediction(test_loader)

#
# inference
#
tree = {
        "theta": [],
        "theta_50": [],
        "theta_16": [],
        "theta_83": [],
    }

trees = {
        "theta": [],
        "posterior": [],
        "theta_50": [],
        "theta_16": [],
        "theta_83": [],
    }

for i in range(100):
    posterior = metropolis_hastings(model, X_test[i], num_samples=num_samples, proposal_std=proposal_std, device=dvc)
    theta_16, theta_50, theta_83 = np.percentile(posterior, [16.5, 50.0, 83.5])
    tree["theta"].append(theta_test[i])
    tree["theta_50"].append(theta_50)
    tree["theta_16"].append(theta_16)
    tree["theta_83"].append(theta_83)

    if i < 20:
        trees["theta"].append(theta_test[i])
        trees["posterior"].append(posterior)
        trees["theta_50"].append(theta_50)
        trees["theta_16"].append(theta_16)
        trees["theta_83"].append(theta_83)

    print(f"[===> {i+1} tests are done]")


outfile = uproot.recreate("./data/posterior_LH2_messy_MC.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile["trees"] = trees
outfile["history"] = fit_history
outfile["test_samples"] = {"labels": labels, "logits": logits}
outfile.close()



#
# inference RS67 LH2 data
#
RS67_LH2_tree = uproot.open("./data/RS67_LH2_hist.root:out_tree")
X_RS67_LH2 = RS67_LH2_tree["X"].array().to_numpy()

posterior = metropolis_hastings(model, X_RS67_LH2[0], num_samples=num_samples, proposal_std=proposal_std, device=dvc)

theta_16, theta_50, theta_83 = np.percentile(posterior, [16.5, 50.0, 83.5])

tree = {
    "posterior": posterior,
    "theta_50": theta_50,
    "theta_16": theta_16,
    "theta_83": theta_83,
    }

outfile = uproot.recreate("./data/posterior_RS67_LH2_data.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile.close()
