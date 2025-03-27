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

from event_weights_methods import ReweightingNetwork, ReweightingDataset, ReweightingLoss

dvc = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {dvc} device")

batch_size = 1024
learning_rate = 0.001
patience = 10
max_epochs = 1000

save = uproot.open("../data/LH2_messy_MC_events.root:save")
branches = ["mass", "pT", "xF", "phi", "costh", "true_mass", "true_pT", "true_xF", "true_phi", "true_costh", "occuD1"]
events = save.arrays(branches)

events1 = events[(4.5 < events.mass) & (events.mass < 8.0) & (events.xF > -0.1) & (events.xF < 0.95) & (np.abs(events.costh) < 0.45) & (events.occuD1 < 300.) & (0.19 < events.pT) & (events.pT < 2.24)]


tree = uproot.open("../data/RS67_LH2_data.root:tree")
branches2 = ["mass", "pT", "xF", "phi", "costh", "weight"]
events2 = tree.arrays(branches2)
weight2 = events2.weight.to_numpy().reshape(-1, 1)

size = len(events2)

events3 = events1[:size]
weight3 = np.ones((size, 1))

X0 = np.array([(mass, pT, xF) for mass, pT, xF in zip(events3.mass, events3.pT, events3.xF)])
Y0 = np.zeros((size, 1))

X1 = np.array([(mass, pT, xF) for mass, pT, xF in zip(events2.mass, events2.pT, events2.xF)])
Y1 = np.ones((size, 1))

X = np.concatenate((X0, X1))
W = np.concatenate((weight3, weight2))
Y = np.concatenate((Y0, Y1))

ds = ReweightingDataset(X, W, Y)

ds_train, ds_val = random_split(ds, [0.5, 0.5])

train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4)

model = ReweightingNetwork().double().to(dvc)

print("===================== Reweighting Network =====================")
print(model)
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total trainable params: {total_trainable_params}")

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
criterion = ReweightingLoss()

#
# train the model
#
summary = {
        "train_loss": [],
        "train_auc": [],
        "val_loss": [],
        "val_auc": [],
    }

best_epoch = None
best_state = None
best_val_loss = None
best_auc = None
i_try = 0
size = len(train_loader.dataset)
for epoch in range(1, max_epochs+1):

    model.train()
    model.train_one_epoch(train_loader, criterion, optimizer, epoch, size, dvc)

    model.eval()
    train_loss, train_auc = model.val_one_epoch(train_loader, criterion, dvc)

    model.eval()
    val_loss, val_auc = model.val_one_epoch(train_loader, criterion, dvc)

    summary["train_loss"].append(train_loss)
    summary["val_loss"].append(val_loss)
    summary["train_auc"].append(train_auc)
    summary["val_auc"].append(val_auc)

    print("\r" + " " * (50), end="")
    print("\r" + f"[Epoch {epoch:>3d}] [Train_loss: {train_loss:>5f} Train_auc: {train_auc:>5f}] [Val_loss: {val_loss:>5f} Val_auc: {val_auc:>5f}]")

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

#
# event reweighting
#
X = np.array([(mass, pT, xF) for mass, pT, xF in zip(events1.mass, events1.pT, events1.xF)])
Y = np.zeros((len(X), 1))
W = np.ones((len(X), 1))

ds = ReweightingDataset(X, W, Y)

data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

model.eval()
event_weights = model.event_reweighting(data_loader, dvc)

event_weights = ak.from_numpy(event_weights)

events4 = events[(4.5 < events1.mass) & (events1.mass < 8.0) & (events1.xF > -0.1) & (events1.xF < 0.95) & (np.abs(events1.costh) < 0.45) & (events1.occuD1 < 300.) & (0.19 < events1.pT) & (events1.pT < 2.24)]

event_weights4 = event_weights[(4.5 < events1.mass) & (events1.mass < 8.0) & (events1.xF > -0.1) & (events1.xF < 0.95) & (np.abs(events1.costh) < 0.45) & (events1.occuD1 < 300.) & (0.19 < events1.pT) & (events1.pT < 2.24)]

train_val_events, test_events, weight_train_val, weight_test = train_test_split(events1.to_numpy(), event_weights4.to_numpy(), test_size=0.2, shuffle=True)
train_events, val_events, weight_train, weight_val = train_test_split(train_val_events, weight_train_val, test_size=0.25, shuffle=True)

outputs = uproot.recreate("./data/generation.root", compression=uproot.ZLIB(4))

outputs["train_tree"] = {
    "mass": train_events["mass"],
    "pT": train_events["pT"],
    "xF": train_events["xF"],
    "phi": train_events["phi"],
    "costh": train_events["costh"],
    "true_mass": train_events["true_mass"],
    "true_pT": train_events["true_pT"],
    "true_xF": train_events["true_xF"],
    "true_phi": train_events["true_phi"],
    "true_costh": train_events["true_costh"],
    "weight": weight_train,
    }

outputs["val_tree"] = {
    "mass": val_events["mass"],
    "pT": val_events["pT"],
    "xF": val_events["xF"],
    "phi": val_events["phi"],
    "costh": val_events["costh"],
    "true_mass": val_events["true_mass"],
    "true_pT": val_events["true_pT"],
    "true_xF": val_events["true_xF"],
    "true_phi": val_events["true_phi"],
    "true_costh": val_events["true_costh"],
    "weight":weight_val,
    }

outputs["test_tree"] = {
    "mass": test_events["mass"],
    "pT": test_events["pT"],
    "xF": test_events["xF"],
    "phi": test_events["phi"],
    "costh": test_events["costh"],
    "true_mass": test_events["true_mass"],
    "true_pT": test_events["true_pT"],
    "true_xF": test_events["true_xF"],
    "true_phi": test_events["true_phi"],
    "true_costh": test_events["true_costh"],
    "weight":weight_test,
    }

outputs.close()
