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

batch_size: int = 256
n_train: int = 500000
n_test: int = 5000

#
# forward simulation
#

# sim = simulator()
# sim.samples(n_train, n_test)
# sim.save()

tree = {
        "theta":[],
        "weights": [],
    }


#
# inference model
#

# train events
train_tree = uproot.open("./data/outputs.root:train_tree")
X_train = train_tree["X"].array().to_numpy()
theta_train = train_tree["theta"].array().to_numpy()
theta_0_train = train_tree["theta_0"].array().to_numpy()


test_tree = uproot.open("./data/outputs.root:test_tree")
X_test = test_tree["X"].array().to_numpy()
theta_test = test_tree["theta"].array().to_numpy()

theta_prior = uproot.open("./data/outputs.root:prior_tree")
theta_0_test = theta_prior["theta_0"].array().to_numpy()

ds_ratio = ratio_dataset(X_train, theta_train, theta_0_train)

ds_train, ds_val = random_split(ds_ratio, [0.8, 0.2])

train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4)

model = resnet_10x10().double().to(dvc)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

tr = ratio_trainner(train_loader, val_loader, model, criterion, optimizer, device=dvc)
tr.fit()

#
# test data
#

# for i in range(len(theta_test)):
#     X_test_array = np.array([X_test[i] for j in range(len(theta_0_test))])
#     tree["theta"].append(theta_test[i])
#     tree["weights"].append(test_ratio_model(model, X_test_array, theta_0_test, batch_size=batch_size, device=dvc))
#
#     if i%1000 == 0:
#         print(f"[===> {i} tests are done ]")
#
# outfile = uproot.recreate("./data/eval.root", compression=uproot.ZLIB(4))
# outfile["tree"] = tree
# outfile["prior"] = {"theta_0": theta_0_test,}
# outfile.close()


#
# plots
#

# rp = ratio_plots()
# rp.fill()
# rp.plot()


# metropolis_hastings
posterior = metropolis_hastings(model, X_test[0], num_samples=10000, proposal_std=0.1, device=dvc)

theta_mean = np.mean(posterior, axis=0)
theta_std = np.std(posterior, axis=0)


bins = np.linspace(-1., 1., 31)
plt.figure(figsize=(8., 8.))
plt.hist(posterior[:, 0], bins=bins, histtype="step", density=True)
plt.axvline(x=theta_test[0, 0], linestyle="--", color="r", label=r"$\lambda_{true}$")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$p(\lambda|x)$")
plt.text(0., 0.1, f"$\lambda_{{fit}}$ = {theta_mean[0]:.3f} +/- {theta_std[0]:.3f}")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("plots/p_lambda_x.png")
plt.close("all")



bins = np.linspace(-0.5, 0.5, 31)
plt.figure(figsize=(8., 8.))
plt.hist(posterior[:, 1], bins=bins, histtype="step", density=True)
plt.axvline(x=theta_test[0, 1], linestyle="--", color="r", label=r"$\mu_{true}$")
plt.xlabel(r"$\mu$")
plt.ylabel(r"$p(\mu|x)$")
plt.text(0., 0.1, f"$\mu_{{fit}}$ = {theta_mean[1]:.3f} +/- {theta_std[1]:.3f}")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("plots/p_mu_x.png")
plt.close("all")


plt.figure(figsize=(8., 8.))
plt.hist(posterior[:, 2], bins=bins, histtype="step", density=True)
plt.axvline(x=theta_test[0, 2], linestyle="--", color="r", label=r"$\nu_{true}$")
plt.xlabel(r"$\nu$")
plt.ylabel(r"$p(\nu|x)$")
plt.text(0., 0.1, f"$\nu_{{fit}}$ = {theta_mean[2]:.3f} +/- {theta_std[2]:.3f}")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("plots/p_nu_x.png")
plt.close("all")
