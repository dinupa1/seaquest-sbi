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
proposal_std: float = 0.001
learning_rate: float = 0.0001
num_samples: int = 10000

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
tr.fit()

model.load_state_dict(tr.best_state)

#
# test samples
#
num_iterations = len(test_loader)//2
loader = iter(test_loader)
model.eval()
labels, logits = None, None
with torch.no_grad():
    for batch in range(num_iterations):

        x_a, theta_a = next(loader)
        x_a, theta_a = x_a.double().to(self.device, non_blocking=True), theta_a.double().to(self.device, non_blocking=True)

        x_b, theta_b = next(loader)
        x_b, theta_b = x_b.double().to(self.device, non_blocking=True), theta_b.double().to(self.device, non_blocking=True)

        _, logit_dep_a = model(x_a, theta_a)
        _, logit_ind_a = model(x_b, theta_a)
        _, logit_dep_b = model(x_b, theta_b)
        _, logit_ind_b = model(x_a, theta_b)

        ones = torch.ones([len(theta_a), 1]).double().to(self.device, non_blocking=True)
        zeros = torch.zeros([len(theta_a), 1]).double().to(self.device, non_blocking=True)

        labels = torch.cat([labels, ones]) if labels is not None else ones
        labels = torch.cat([labels, zeros]) if labels is not None else zeros
        labels = torch.cat([labels, ones]) if labels is not None else ones
        labels = torch.cat([labels, zeros]) if labels is not None else zeros

        logits = torch.cat([logits, logit_dep_a]) if logits is not None else logit_dep_a
        logits = torch.cat([logits, logit_ind_a]) if logits is not None else logit_ind_a
        logits = torch.cat([logits, logit_dep_b]) if logits is not None else logit_dep_b
        logits = torch.cat([logits, logit_ind_b]) if logits is not None else logit_ind_b


logits, labels = labels.detach().cpu().numpy(), logits.detach().cpu().numpy()

#
# inference
#
tree = {
        "theta": [],
        "posterior": [],
        "mean": [],
        "std": [],
    }

X_test = X_test[(np.abs(theta_test[:, 0]) < 1.) & (np.abs(theta_test[:, 1]) < 0.5) & (np.abs(theta_test[:, 2]) < 0.5) ]
theta_test = theta_test[(np.abs(theta_test[:, 0]) < 1.) & (np.abs(theta_test[:, 1]) < 0.5) & (np.abs(theta_test[:, 2]) < 0.5) ]

for i in range(200):
    posterior = metropolis_hastings(model, X_test[i], num_samples=num_samples, proposal_std=proposal_std, device=dvc)

    tree["theta"].append(theta_test[i])
    tree["posterior"].append(posterior)
    tree["mean"].append(np.mean(posterior, axis=0))
    tree["std"].append(np.std(posterior, axis=0))

    print(f"[===> {i+1} samples are done ]")


outfile = uproot.recreate("./data/posterior_LH2_messy_MC.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile["history"] = tr.fit_history
outfile["test_samples"] = {"labels": labels, "logits": logits}
outfile.close()



#
# inference RS67 LH2 data
#
print("[===> inference RS67 LH2 data]")

RS67_LH2_tree = uproot.open("./data/RS67_LH2_hist.root:out_tree")
X_RS67_LH2 = RS67_LH2_tree["X"].array().to_numpy()

tree = {
        "posterior": [],
        "mean": [],
        "std": [],
    }

posterior = metropolis_hastings(model, X_RS67_LH2[0], num_samples=num_samples, proposal_std=proposal_std, device=dvc)

tree["posterior"].append(posterior)
tree["mean"].append(np.mean(posterior, axis=0))
tree["std"].append(np.std(posterior, axis=0))

outfile = uproot.recreate("./data/posterior_RS67_LH2_data.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile.close()
