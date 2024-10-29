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

seed: int = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size: int = 5000

dvc = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {dvc} device")


tree = uproot.open("./data/eval.root:tree")
X_test = tree["Xs"].array().to_numpy()
theta_test = tree["priors"].array().to_numpy()


model = ratio_net().double().to(dvc)

weights = test_ratio_model(model, X_test, theta_test, batch_size, dvc)


outputs = uproot.recreate("./data/weights.root", compression=uproot.ZLIB(4))
outputs["tree"] = {
    "lambda": theta_test[:, 0],
    "mu": theta_test[:, 1],
    "nu": theta_test[:, 2],
    "weight": weights,
}
outputs.close()
