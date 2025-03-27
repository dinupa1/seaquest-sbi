import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colormaps
import mplhep as hep

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

from inference_methods import RatioNetwork, RatioDataset


dvc = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {dvc} device")


batch_size = 10240
learning_rate = 0.001
step_size = 0.01
num_samples = 10000
num_inference = 200
max_epochs = 1000
patience = 10
num_iterations = 100
burn = int(0.2* num_samples)


outfile = uproot.open("./data/outfile.root")
train_out = outfile["train_out"]
val_out = outfile["val_out"]
test_out = outfile["test_out"]

X_train = train_out["X"].array().to_numpy()
theta_train = train_out["theta"].array().to_numpy()

X_val = val_out["X"].array().to_numpy()
theta_val = val_out["theta"].array().to_numpy()

X_test = test_out["X"].array().to_numpy()
theta_test = test_out["theta"].array().to_numpy()

tree = uproot.open(f"./data/RS67_LH2_hist.root:out_tree")
X = tree["X"].array().to_numpy()
X = torch.from_numpy(X).double().to(dvc)


ds_val = RatioDataset(X_val, theta_val)
ds_test = RatioDataset(X_test, theta_test)

val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)

criterion = nn.BCELoss()

summary ={"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}

theta_test = np.zeros((num_iterations, 3))
theta_posterior = np.zeros((num_iterations, 3))
results = np.zeros((num_iterations, 3))

quantiles = torch.tensor([0.165, 0.5, 0.835]).double()


for iteration in range(num_iterations):

    print(f"[===> Iteration {iteration+1}]")

    X_resample, theta_resample = resample(X_train, theta_train)

    ds_train = RatioDataset(X_resample, theta_resample)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)

    model = RatioNetwork().double().to(dvc)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    #
    # model training
    #
    print("===================== Ratio Network =====================")
    print(model)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total trainable params: {total_trainable_params}")

    best_state = model.state_dict()
    best_val_loss = None
    best_auc = None
    i_try = 0
    size = len(train_loader.dataset)
    i_summary ={"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, max_epochs+1):
        model.train()
        model.train_one_epoch(train_loader, criterion, optimizer, epoch, size, dvc)

        model.eval()
        train_loss, train_auc = model.val_one_epoch(train_loader, criterion, dvc)

        model.eval()
        val_loss, val_auc = model.val_one_epoch(val_loader, criterion, dvc)

        print("\r" + " " * (50), end="")
        print("\r" + f"[Epoch {epoch:>3d}] [Train_loss: {train_loss:>5f} Train_auc: {train_auc:>5f}] [Val_loss: {val_loss:>5f} Val_auc: {val_auc:>5f}]")

        i_summary["train_loss"].append(train_loss)
        i_summary["train_auc"].append(train_auc)
        i_summary["val_loss"].append(val_loss)
        i_summary["val_auc"].append(val_auc)

        if best_val_loss == None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            i_try = 0
        elif i_try < patience:
            i_try += 1
        else:
            print(f"Early stopping! Restore state at epoch {best_epoch}.")
            print(f"[Best_val_loss: {best_val_loss:>5f}, Best_ROC_AUC: {best_auc:>5f}]")
            model.load_state_dict(best_state)
            break

    summary["train_loss"].append(i_summary["train_loss"])
    summary["train_auc"].append(i_summary["train_auc"])
    summary["val_loss"].append(i_summary["val_loss"])
    summary["val_auc"].append(i_summary["val_auc"])

    model.eval()
    posterior = model.samples(torch.from_numpy(X_test[0]).double().to(dvc), num_samples, step_size, dvc)
    predictions = torch.quantile(posterior[burn:, :].double(), quantiles, dim=0)

    theta_posterior[iteration, 0] = predictions[1, 0].item()
    theta_posterior[iteration, 1] = predictions[1, 1].item()
    theta_posterior[iteration, 2] = predictions[1, 2].item()

    theta_test[iteration, :] = theta_test[0]

    model.eval()
    posterior = model.samples(X[0], num_samples, step_size, dvc)
    predictions = torch.quantile(posterior[burn:, :].double(), quantiles, dim=0)

    results[iteration, 0] = predictions[1, 0].item()
    results[iteration, 1] = predictions[1, 1].item()
    results[iteration, 2] = predictions[1, 2].item()

outfile = uproot.recreate(f"./data/results_with_weights.root", compression=uproot.ZLIB(4))

outfile["theta_test"] = {
        "theta_test": theta_test,
        "theta_posterior": theta_posterior,
    }

outfile["results"] = {"theta": results}

outfile.close()
