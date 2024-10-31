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


class ratio_trainner:
    def __init__(self, train_dataloader, val_dataloader, ratio_model, criterion, optimizer, max_epoch=1000, patience=10, device=None):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.ratio_model = ratio_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.patience = patience
        self.device = device
        
        print("===================== Ratio Model =====================")
        print(self.ratio_model)
        total_trainable_params = sum(p.numel() for p in self.ratio_model.parameters() if p.requires_grad)
        print(f"total trainable params: {total_trainable_params}")
        
        self.best_state = self.ratio_model.state_dict()
        self.best_epoch = None
        self.best_val_loss = None
        self.best_auc = None
        self.i_try = 0
        self.epoch = 0
        self.size = len(train_dataloader.dataset)
        
    def backpropagation(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def train_step(self):
        self.ratio_model.train()
        for batch, (X, theta, label) in enumerate(self.train_dataloader):

            X, theta, label = X.double().to(self.device), theta.double().to(self.device), label.double().to(self.device)

            logit = self.ratio_model(X, theta)
            
            loss = self.criterion(logit, label)

            # Backpropagation
            self.backpropagation(loss)

            loss, current = loss.item(), (batch + 1) * len(X)
            print("\r" + f"[Epoch {self.epoch:>3d}] [{current:>5d}/{self.size:>5d}] [Train_loss: {loss:>5f}]", end="")
            
    def eval_step(self, data_loader):
        self.ratio_model.eval()
        logits, labels = None, None
        with torch.no_grad():
            for batch, (X, theta, label) in enumerate(data_loader):
                
                X, theta, label = X.double().to(self.device), theta.double().to(self.device), label.double().to(self.device)
                
                logit = self.ratio_model(X, theta)

                logits = torch.cat([logits, logit]) if logits is not None else logit
                labels = torch.cat([labels, label]) if labels is not None else label
                
            loss = self.criterion(logits, labels)
            auc = roc_auc_score(labels.cpu().numpy().reshape(-1), logits.cpu().numpy().reshape(-1))
            return loss, auc
            
    def fit(self, n_epoch=None):
        max_epoch = (self.epoch + n_epoch + 1) if n_epoch else self.max_epoch
        
        for epoch in range(self.epoch + 1, max_epoch):
            self.epoch = epoch
            
            # train
            self.train_step()
            
            # evaluate loss for training set
            train_loss, train_auc = self.eval_step(self.train_dataloader)
            
            # evaluate loss for validation set
            val_loss, val_auc = self.eval_step(self.val_dataloader)

            print("\r" + " " * (50), end="")
            print("\r" + f"[Epoch {epoch:>3d}] [Train_loss: {train_loss:>5f} Train_auc: {train_auc:>5f}] [Val_loss: {val_loss:>5f} Val_auc: {val_auc:>5f}]")

            if self.best_val_loss == None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_auc = val_auc
                self.best_state = copy.deepcopy(self.ratio_model.state_dict())
                self.best_epoch = epoch
                self.i_try = 0
            elif self.i_try < self.patience:
                self.i_try += 1
            else:
                print(f"Early stopping! Restore state at epoch {self.best_epoch}.")
                print(f"[Best_val_loss: {self.best_val_loss:>5f}, Best_ROC_AUC: {self.best_auc:>5f}]")
                self.ratio_model.load_state_dict(self.best_state)
                break


def test_ratio_model(ratio_model, X_test, theta_test, batch_size=5000, device=None):

    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(theta_test))
    dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logits = None

    ratio_model.eval()
    with torch.no_grad():
        for batch, (X, theta) in enumerate(dataloader):
            X, theta = X.double().to(device), theta.double().to(device)

            logit = ratio_model(X, theta)
            logits = torch.cat([logits, logit]) if logits is not None else logit

    logits = logits.cpu().detach().numpy()

    return logits[:, 0]/(1. - logits[:, 0] + 0.000000000000000001)
