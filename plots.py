import numpy as np
import matplotlib.pyplot as plt

import mplhep as hep

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

import uproot
import awkward as ak


plt.style.use([hep.style.ROOT, hep.style.firamath])


trees = uproot.open("./data/posterior_LH2_messy_MC.root:trees")
theta = trees["theta"].array().to_numpy()
posterior = trees["posterior"].array().to_numpy()

tree = uproot.open("./data/posterior_LH2_messy_MC.root:tree")
theta = tree["theta"].array().to_numpy()
meas = tree["meas"].array().to_numpy()
error = tree["error"].array().to_numpy()

systematics = uproot.open("./data/systematics_LH2_messy_MC.root:tree")
means = systematics["mean"].array().to_numpy()
theta_par = systematics["theta"].array().to_numpy()


for i in range(20):

    mean = np.mean(posterior[i, :, :], axis=0)
    error = np.std(posterior[i, :, :], axis=0)

    bins = np.linspace(-1.5, 1.5, 31)

    plt.figure(figsize=(8., 8.))
    hep.histplot(posterior[i, :, 0], bins=bins, histtype="step", label=fr"$\lambda_{{posterior}}$ = {mean[0]:.3f} +/- {error[0]:.3f}")
    plt.axvline(x=theta[i, 0], linestyle="--", color="r", label=fr"$\lambda_{{par}}$ = {theta[i, 0]:.3f}")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$p(\lambda|\phi, cos\theta)$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/lambda_{i}.png")
    plt.close("all")

    bins = np.linspace(-0.6, 0.6, 31)

    plt.figure(figsize=(8., 8.))
    hep.histplot(posterior[i, :, 1], bins=bins, histtype="step", label=fr"$\mu_{{posterior}}$ = {mean[1]:.3f} +/- {error[1]:.3f}")
    plt.axvline(x=theta[i, 1], linestyle="--", color="r", label=fr"$\mu_{{par}}$ = {theta[i, 1]:.3f}")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$p(\mu|\phi, cos\theta)$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/mu_{i}.png")
    plt.close("all")

    plt.figure(figsize=(8., 8.))
    hep.histplot(posterior[i, :, 2], bins=bins, histtype="step", label=fr"$\nu_{{posterior}}$ = {mean[2]:.3f} +/- {error[2]:.3f}")
    plt.axvline(x=theta[i, 2], linestyle="--", color="r", label=fr"$\nu_{{par}}$ = {theta[i, 2]:.3f}")
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"$p(\nu|\phi, cos\theta)$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/nu_{i}.png")
    plt.close("all")


plt.figure(figsize=(8., 8.))
plt.plot(posterior[0, :, 0], label=r"$\lambda$")
plt.plot(theta[0, 0] * np.ones(len(theta)), label="target")
plt.xlabel("step")
plt.ylabel(r"$\lambda$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/lambda_chain.png")
plt.close("all")

plt.figure(figsize=(8., 8.))
plt.plot(posterior[0, :, 1], label=r"$\mu$")
plt.plot(theta[0, 1] * np.ones(len(theta)), label="target")
plt.xlabel("step")
plt.ylabel(r"$\mu$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/mu_chain.png")
plt.close("all")

plt.figure(figsize=(8., 8.))
plt.plot(posterior[0, :, 2], label=r"$\nu$")
plt.plot(theta[0, 2] * np.ones(len(theta)), label="target")
plt.xlabel("step")
plt.ylabel(r"$\nu$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/nu_chain.png")
plt.close("all")


xvals = np.linspace(-1., 1., 50)
plt.figure(figsize=(8., 8.))
plt.errorbar(theta[:, 0], meas[:, 0], yerr=error[:, 0], fmt="bo", capsize=1.)
plt.plot(xvals, xvals, linestyle="--", color="r", label=r"$\lambda_{true} = \lambda_{meas}$")
plt.xlabel(r"$\lambda_{par}$")
plt.ylabel(r"$\lambda_{meas}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/lambda.png")
plt.close("all")


xvals = np.linspace(-0.4, 0.4, 50)
plt.figure(figsize=(8., 8.))
plt.errorbar(theta[:, 1], meas[:, 1], yerr=error[:, 1], fmt="bo", capsize=1.)
plt.plot(xvals, xvals, linestyle="--", color="r", label=r"$\mu_{true} = \mu_{meas}$")
plt.xlabel(r"$\mu_{par}$")
plt.ylabel(r"$\mu_{meas}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/mu.png")
plt.close("all")


xvals = np.linspace(-0.4, 0.4, 50)
plt.figure(figsize=(8., 8.))
plt.errorbar(theta[:, 2], meas[:, 2], yerr=error[:, 2], fmt="bo", capsize=1.)
plt.plot(xvals, xvals, linestyle="--", color="r", label=r"$\nu_{true} = \nu_{meas}$")
plt.xlabel(r"$\nu_{par}$")
plt.ylabel(r"$\nu_{meas}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/nu.png")
plt.close("all")


score = (theta - meas)/error

score_mean = np.mean(score, axis=0)
score_error = np.std(score, axis=0)

bins = np.linspace(-5., 5., 31)

plt.figure(figsize=(8., 8.))
hep.histplot(score[:, 0], bins=bins, histtype="step", label=f"Mean = {score_mean[0]:.3f}, Std. Dev. = {score_error[0]:.3f}")
plt.xlabel(r"$\frac{\lambda_{par} - \lambda_{meas}}{\sigma_{\lambda}}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/lambda_score.png")
plt.close("all")

plt.figure(figsize=(8., 8.))
hep.histplot(score[:, 1], bins=bins, histtype="step", label=f"Mean = {score_mean[1]:.3f}, Std Dev = {score_error[1]:.3f}")
plt.xlabel(r"$\frac{\mu_{par} - \mu_{meas}}{\sigma_{\mu}}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/mu_score.png")
plt.close("all")

plt.figure(figsize=(8., 8.))
hep.histplot(score[:, 2], bins=bins, histtype="step", label=f"Mean = {score_mean[2]:.3f}, Std. Dev. {score_error[2]:.3f}")
plt.xlabel(r"$\frac{\nu_{par} - \nu_{meas}}{\sigma_{\nu}}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/nu_score.png")
plt.close("all")


ybins = np.linspace(-5., 5., 31)
xbins = np.linspace(-1., 1., 31)

ones = np.ones(len(xbins))

plt.figure(figsize=(8., 8.))
plt.plot(theta[:, 0], score[:, 0], "bo")
plt.plot(xbins, 2. * ones, "r--", label=r"$2\sigma$ interval")
plt.plot(xbins, -2. * ones, "r--")
plt.ylim(-5., 5.)
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\frac{\lambda_{par} - \lambda_{meas}}{\sigma_{\lambda}}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/lambda_true_score.png")
plt.close("all")

xbins = np.linspace(-0.4, 0.4, 31)

plt.figure(figsize=(8., 8.))
plt.plot(theta[:, 1], score[:, 1], "bo")
plt.plot(xbins, 2. * ones, "r--", label=r"$2\sigma$ interval")
plt.plot(xbins, -2. * ones, "r--")
plt.ylim(-5., 5.)
plt.xlabel(r"$\mu$")
plt.ylabel(r"$\frac{\mu_{par} - \mu_{meas}}{\sigma_{\mu}}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/mu_true_score.png")
plt.close("all")


plt.figure(figsize=(8., 8.))
plt.plot(theta[:, 2], score[:, 2], "bo")
plt.plot(xbins, 2. * ones, "r--", label=r"$2\sigma$ interval")
plt.plot(xbins, -2. * ones, "r--")
plt.ylim(-5., 5.)
plt.xlabel(r"$\nu$")
plt.ylabel(r"$\frac{\nu_{par} - \nu_{meas}}{\sigma_{\nu}}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/nu_true_score.png")
plt.close("all")

ybins = np.linspace(0., 1., 31)
xbins = np.linspace(-1., 1., 31)

plt.figure(figsize=(8., 8.))
plt.plot(theta[:, 0], error[:, 0], "bo")
plt.ylim(0., 0.5)
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\sigma_{\lambda}$")
plt.tight_layout()
plt.savefig("./plots/lambda_true_error.png")
plt.close("all")


xbins = np.linspace(-0.4, 0.4, 31)

plt.figure(figsize=(8., 8.))
plt.plot(theta[:, 1], error[:, 1], "bo")
plt.ylim(0., 0.2)
plt.xlabel(r"$\mu$")
plt.ylabel(r"$\sigma_{\mu}$")
plt.tight_layout()
plt.savefig("./plots/mu_true_error.png")
plt.close("all")

plt.figure(figsize=(8., 8.))
plt.plot(theta[:, 2], error[:, 2], "bo")
plt.ylim(0., 0.2)
plt.xlabel(r"$\nu$")
plt.ylabel(r"$\sigma_{\nu}$")
plt.tight_layout()
plt.savefig("./plots/nu_true_error.png")
plt.close("all")


sigma_sys = np.std(means, axis=0)
x_min = np.min(means, axis=0)
x_max = np.max(means, axis=0)

bins = np.linspace(x_min[0]+0.05, x_max[1]+0.05, 21)
plt.figure(figsize=(8., 8.))
hep.histplot(mean[:, 0], bins=bins, histtype="step", label=f"Std. Dev. = {sigma_sys[0]:.3f}")
plt.axvline(x=theta_par[0, 0], linestyle="--", color="r", label=fr"$\lambda_{{par}}$ = {theta_par[0, 0]:.3f}")
plt.xlabel(r"$\lambda$")
plt.ylabel("counts")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/lambda_systematics_MC.png")
plt.close("all")

bins = np.linspace(x_min[1]+0.05, x_max[1]+0.05, 21)
plt.figure(figsize=(8., 8.))
hep.histplot(mean[:, 1], bins=bins, histtype="step", label=f"Std. Dev. = {sigma_sys[1]:.3f}")
plt.axvline(x=theta_par[0, 1], linestyle="--", color="r", label=fr"$\mu_{{par}}$ = {theta_par[0, 1]:.3f}")
plt.xlabel(r"$\mu$")
plt.ylabel("counts")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/mu_systematics_MC.png")
plt.close("all")

bins = np.linspace(x_min[0]+0.05, x_max + 0.05, 21)
plt.figure(figsize=(8., 8.))
hep.histplot(mean[:, 2], bins=bins, histtype="step", label=f"Std. Dev. = {sigma_sys[2]:.3f}")
plt.axvline(x=theta_par[0, 2], linestyle="--", color="r", label=fr"$\nu_{{par}}$ = {theta_par[0, 2]:.3f}")
plt.xlabel(r"$\nu$")
plt.ylabel("counts")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/nu_systematics_MC.png")
plt.close("all")
