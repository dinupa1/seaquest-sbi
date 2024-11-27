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
        self.max_epoch = max_epoch
        self.patience = patience
        self.device = device
        
        print("===================== Ratio Network =====================")
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
            x_a, theta_a = x_a.double().to(self.device, non_blocking=True), theta_a.double().to(self.device, non_blocking=True)

            x_b, theta_b = next(loader)
            x_b, theta_b = x_b.double().to(self.device, non_blocking=True), theta_b.double().to(self.device, non_blocking=True)

            _, logit_dep_a = self.ratio_model(x_a, theta_a)
            _, logit_ind_a = self.ratio_model(x_b, theta_a)

            _, logit_dep_b = self.ratio_model(x_b, theta_b)
            _, logit_ind_b = self.ratio_model(x_a, theta_b)

            ones = torch.ones([len(theta_a), 1]).double().to(self.device, non_blocking=True)
            zeros = torch.zeros([len(theta_a), 1]).double().to(self.device, non_blocking=True)


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
        loss, auc = 0, 0
        with torch.no_grad():
            for batch in range(num_iterations):

                x_a, theta_a = next(loader)
                x_a, theta_a = x_a.double().to(self.device, non_blocking=True), theta_a.double().to(self.device, non_blocking=True)

                x_b, theta_b = next(loader)
                x_b, theta_b = x_b.double().to(self.device, non_blocking=True), theta_b.double().to(self.device, non_blocking=True)

                _, logit_dep_a = self.ratio_model(x_a, theta_a)
                _, logit_ind_a = self.ratio_model(x_b, theta_a)

                _, logit_dep_b = self.ratio_model(x_b, theta_b)
                _, logit_ind_b = self.ratio_model(x_a, theta_b)

                ones = torch.ones([len(theta_a), 1]).double().to(self.device, non_blocking=True)
                zeros = torch.zeros([len(theta_a), 1]).double().to(self.device, non_blocking=True)

                loss_a = self.criterion(logit_dep_a, ones) + self.criterion(logit_ind_a, zeros)
                loss_b = self.criterion(logit_dep_b, ones) + self.criterion(logit_ind_b, zeros)

                auc_a = roc_auc_score(torch.cat([ones, zeros]).cpu().numpy().reshape(-1), torch.cat([logit_dep_a, logit_ind_a]).cpu().numpy().reshape(-1))
                auc_b = roc_auc_score(torch.cat([ones, zeros]).cpu().numpy().reshape(-1), torch.cat([logit_dep_b, logit_ind_b]).cpu().numpy().reshape(-1))

                loss += loss_a + loss_b
                auc += auc_a + auc_b

            return loss/num_iterations, auc/(2. * num_iterations)
            
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


def metropolis_hastings(ratio_model, X, num_samples=10000, proposal_std=0.01, device=None):
    chain = []

    theta_current = torch.zeros(9).double().to(device)
    X_tensor = torch.from_numpy(X).double().to(device)

    ratio_model.eval()
    with torch.no_grad():
        for _ in range(num_samples):

            theta_proposal = np.random.multivariate_normal(theta_current.cpu().numpy(), proposal_std * np.eye(9))

            theta_proposal = torch.from_numpy(theta_proposal).double().to(device)

            if (theta_proposal[0] < -1.5 or 1.5 < theta_proposal[0] or theta_proposal[1] < -1.5 or 1.5 < theta_proposal[1] or theta_proposal[2] < -1.5 or 1.5 < theta_proposal[2] or theta_proposal[3] < -0.6 or 0.6 < theta_proposal[3] or theta_proposal[4] < -0.6 or 0.6 < theta_proposal[4] or theta_proposal[5] < -0.6 or 0.6 < theta_proposal[5] or theta_proposal[6] < -0.6 or 0.6 < theta_proposal[6] or theta_proposal[7] < -0.6 or 0.6 < theta_proposal[7] or theta_proposal[8] < -0.6 or 0.6 < theta_proposal[8]):
                chain.append(theta_current.cpu().numpy())
                continue

            log_r_current, _ = ratio_model(X_tensor.unsqueeze(0), theta_current.unsqueeze(0))
            log_r_proposal, _ = ratio_model(X_tensor.unsqueeze(0), theta_proposal.unsqueeze(0))

            threshold = np.random.uniform(0., 1.)

            if threshold <= min(1, torch.exp(log_r_proposal - log_r_current).item()):
                theta_current = theta_proposal


            chain.append(theta_current.cpu().numpy())

    return np.array(chain)
