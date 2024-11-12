import numpy as np
import matplotlib.pyplot as plt

import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

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
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)
        self.max_epoch = max_epoch
        self.patience = patience
        self.device = device
        
        print("===================== ResNet_18 =====================")
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
        num_iterations = len(self.train_dataloader)//2
        loader = iter(self.train_dataloader)
        self.ratio_model.train()
        for batch in range(num_iterations):

            x_a, theta_a = next(loader)
            x_a, theta_a = x_a.double().to(self.device), theta_a.double().to(self.device)

            x_b, theta_b = next(loader)
            x_b, theta_b = x_b.double().to(self.device), theta_b.double().to(self.device)

            _, logit_dep_a = self.ratio_model(x_a, theta_a)
            _, logit_ind_a = self.ratio_model(x_a, theta_b)

            _, logit_dep_b = self.ratio_model(x_b, theta_b)
            _, logit_ind_b = self.ratio_model(x_b, theta_a)

            ones = torch.ones([len(theta_a), 1]).double().to(self.device)
            zeros = torch.zeros([len(theta_a), 1]).double().to(self.device)


            loss_a = self.criterion(logit_dep_a, ones) + self.criterion(logit_ind_a, zeros)
            loss_b = self.criterion(logit_dep_b, ones) + self.criterion(logit_ind_b, zeros)
            
            loss = loss_a + loss_b

            # Backpropagation
            self.backpropagation(loss)

            loss, current = loss.item(), (batch + 1) * len(x_a)
            print("\r" + f"[Epoch {self.epoch:>3d}] [{current:>5d}/{self.size:>5d}] [Train_loss: {loss:>5f}]", end="")
            
    def eval_step(self, data_loader):
        num_iterations = len(data_loader)//2
        loader = iter(data_loader)
        self.ratio_model.eval()
        with torch.no_grad():
            for batch in range(num_iterations):

                x_a, theta_a = next(loader)
                x_a, theta_a = x_a.double().to(self.device), theta_a.double().to(self.device)

                x_b, theta_b = next(loader)
                x_b, theta_b = x_b.double().to(self.device), theta_b.double().to(self.device)

                _, logit_dep_a = self.ratio_model(x_a, theta_a)
                _, logit_ind_a = self.ratio_model(x_a, theta_b)

                _, logit_dep_b = self.ratio_model(x_b, theta_b)
                _, logit_ind_b = self.ratio_model(x_b, theta_a)

                ones = torch.ones([len(theta_a), 1]).double().to(self.device)
                zeros = torch.zeros([len(theta_a), 1]).double().to(self.device)

                loss_a = self.criterion(logit_dep_a, ones) + self.criterion(logit_ind_a, zeros)
                loss_b = self.criterion(logit_dep_b, ones) + self.criterion(logit_ind_b, zeros)

                auc_a = roc_auc_score(torch.cat([ones, zeros], dim=1).cpu().numpy().reshape(-1), torch.cat([logit_dep_a, logit_ind_a]).cpu().numpy().reshape(-1))
                auc_b = roc_auc_score(torch.cat([ones, zeros], dim=1).cpu().numpy().reshape(-1), torch.cat([logit_dep_b, logit_ind_b]).cpu().numpy().reshape(-1))

                loss += loss_a + loss_b
                auc += auc_a + auc_b

            return loss/num_iterations, auc/num_iterations
            
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

            self.scheduler.step()

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

    log_rs = None

    ratio_model.eval()
    with torch.no_grad():
        for batch, (X, theta) in enumerate(dataloader):
            X, theta = X.double().to(device), theta.double().to(device)

            log_r, logit = ratio_model(X, theta)
            log_rs = torch.cat([log_rs, log_r]) if log_rs is not None else log_r

    log_rs = log_rs.cpu().detach().numpy()

    return np.exp(log_rs)
