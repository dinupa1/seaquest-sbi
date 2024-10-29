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


seed: int = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size: int = 10000

dvc = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {dvc} device")

ds_train = ratio_dataset("./data/outputs.root:train_tree")
ds_val = ratio_dataset("./data/outputs.root:val_tree")

train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

model = ratio_net().double().to(dvc)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

tr = ratio_trainner(train_loader, val_loader, model, criterion, optimizer, device=dvc)
tr.fit()
