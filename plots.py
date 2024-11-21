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


input_file = "./data/eval_pT.root"

trees = uproot.open(f"{input_file}:trees")
theta = trees["theta"].array().to_numpy()
posterior = trees["posterior"].array().to_numpy()

pT_edges = np.array([0., 0.404, 0.633, 0.909, 2.500])


for i in range(5):

    mean = np.mean(posterior[i, :, :], axis=0)
    error = np.std(posterior[i, :, :], axis=0)

    for j in range(4):

        bins = np.linspace(-1.5, 1.5, 31)

        plt.figure(figsize=(8., 8.))
        plt.hist(posterior[i, :, 3 * j + 0], bins=bins, histtype="step", label=fr"$\lambda_{{posterior}}$ = {mean[3 * j + 0]:.3f} +/- {error[3 * j + 0]:.3f}")
        plt.axvline(x=theta[i, 3 * j + 0], linestyle="--", color="r", label=fr"$\lambda_{{true}}$ = {theta[i, 3 * j + 0]:.3f}")
        plt.xlabel(fr"$\lambda$ [{pT_edges[j]} < $p_{{T}}^{{true}}$ <= {pT_edges[j + 1]}]")
        plt.ylabel(r"$p(\lambda|x)$")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f"./plots/lambda_{i}_pT_bin_{j}.png")
        plt.close("all")

        bins = np.linspace(-0.6, 0.6, 31)

        plt.figure(figsize=(8., 8.))
        plt.hist(posterior[i, :, 3 * j + 1], bins=bins, histtype="step", label=fr"$\mu_{{posterior}}$ = {mean[3 * j + 1]:.3f} +/- {error[3 * j + 1]:.3f}")
        plt.axvline(x=theta[i, 3 * j + 1], linestyle="--", color="r", label=fr"$\mu_{{true}}$ = {theta[i, 3 * j + 1]:.3f}")
        plt.xlabel(fr"$\mu$ [{pT_edges[j]} < $p_{{T}}^{{true}}$ <= {pT_edges[j + 1]}]")
        plt.ylabel(r"$p(\mu|x)$")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f"./plots/mu_{i}_pT_bin_{j}.png")
        plt.close("all")

        plt.figure(figsize=(8., 8.))
        plt.hist(posterior[i, :, 3 * j + 2], bins=bins, histtype="step", label=fr"$\nu_{{posterior}}$ = {mean[3 * j + 2]:.3f} +/- {error[3 * j + 2]:.3f}")
        plt.axvline(x=theta[i, 3 * j + 2], linestyle="--", color="r", label=fr"$\nu_{{true}}$ = {theta[i, 3 * j + 2]:.3f}")
        plt.xlabel(fr"$\nu$ [{pT_edges[j]} < $p_{{T}}^{{true}}$ <= {pT_edges[j + 1]}]")
        plt.ylabel(r"$p(\nu|x)$")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f"./plots/nu_{i}_pT_bin_{j}.png")
        plt.close("all")


tree = uproot.open(f"{input_file}:tree")
theta = tree["theta"].array().to_numpy()
meas = tree["meas"].array().to_numpy()
error = tree["error"].array().to_numpy()

score = (theta - meas)/error

score_mean = np.mean(score, axis=0)
score_error = np.std(score, axis=0)


for i in range(4):

    xvals = np.linspace(-1., 1., 50)
    plt.figure(figsize=(8., 8.))
    plt.errorbar(theta[:, 3 * i + 0], meas[:, 3 * i + 0], yerr=error[:, 3 * i + 0], fmt="bo", capsize=1.)
    plt.plot(xvals, xvals, linestyle="--", color="r", label=r"$\lambda_{true} = \lambda_{meas}$")
    plt.xlabel(fr"$\lambda_{{true}}$ [{pT_edges[i]} < $p_{{T}}^{{true}}$ <= {pT_edges[i + 1]}]")
    plt.ylabel(r"$\lambda_{meas}$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/lambda_pT_bin_{i}.png")
    plt.close("all")

    xvals = np.linspace(-0.4, 0.4, 50)
    plt.figure(figsize=(8., 8.))
    plt.errorbar(theta[:, 3 * i + 1], meas[:, 3 * i + 1], yerr=error[:, 3 * i + 1], fmt="bo", capsize=1.)
    plt.plot(xvals, xvals, linestyle="--", color="r", label=r"$\mu_{true} = \mu_{meas}$")
    plt.xlabel(fr"$\mu_{{true}}$ [{pT_edges[i]} < $p_{{T}}^{{true}}$ <= {pT_edges[i + 1]}]")
    plt.ylabel(r"$\mu_{meas}$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/mu_pT_bin_{i}.png")
    plt.close("all")

    plt.figure(figsize=(8., 8.))
    plt.errorbar(theta[:, 3 * i + 2], meas[:, 3 * i + 2], yerr=error[:, 3 * i + 2], fmt="bo", capsize=1.)
    plt.plot(xvals, xvals, linestyle="--", color="r", label=r"$\nu_{true} = \nu_{meas}$")
    plt.xlabel(fr"$\nu_{{true}}$ [{pT_edges[i]} < $p_{{T}}^{{true}}$ <= {pT_edges[i + 1]}]")
    plt.ylabel(r"$\nu_{meas}$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/nu_pT_bin_{i}.png")
    plt.close("all")

    bins = np.linspace(-5., 5., 31)

    plt.figure(figsize=(8., 8.))
    plt.hist(score[:, 3 * i + 0], bins=bins, histtype="step", label=f"Mean = {score_mean[3 * i + 0]:.3f}, Std. Dev. = {score_error[3 * i + 0]:.3f}")
    plt.xlabel(fr"$\frac{{\lambda_{{true}} - \lambda_{{meas}}}} {{\sigma_{{\lambda}}}}$ [{pT_edges[i]} < $p_{{T}}^{{true}}$ <= {pT_edges[i + 1]}]")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/lambda_score_pT_bin_{i}.png")
    plt.close("all")

    plt.figure(figsize=(8., 8.))
    plt.hist(score[:, 3 * i + 1], bins=bins, histtype="step", label=f"Mean = {score_mean[3 * i + 1]:.3f}, Std Dev = {score_error[3 * i + 1]:.3f}")
    plt.xlabel(fr"$\frac{{\mu_{{true}} - \mu_{{meas}}}} {{\sigma_{{\mu}}}}$ [{pT_edges[i]} < $p_{{T}}^{{true}}$ <= {pT_edges[i + 1]}]")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/mu_score_pT_bin_{i}.png")
    plt.close("all")

    plt.figure(figsize=(8., 8.))
    plt.hist(score[:, 2], bins=bins, histtype="step", label=f"Mean = {score_mean[2]:.3f}, Std. Dev. {score_error[2]:.3f}")
    plt.xlabel(fr"$\frac{{\nu_{{true}} - \nu_{{meas}}}} {{\sigma_{{\nu}}}}$ [{pT_edges[i]} < $p_{{T}}^{{true}}$ <= {pT_edges[i + 1]}]")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/nu_score_pT_bin_{i}.png")
    plt.close("all")

    xbins = np.linspace(-1., 1., 31)

    ones = np.ones(len(xbins))

    plt.figure(figsize=(8., 8.))
    plt.plot(theta[:, 3 * i + 0], score[:, 3 * i + 0], "bo")
    plt.plot(xbins, 2. * ones, "r--", label=r"$2\sigma$ interval")
    plt.plot(xbins, -2. * ones, "r--")
    plt.ylim(-5., 5.)
    plt.xlabel(fr"$\lambda_{{true}}$ [{pT_edges[i]} < $p_{{T}}^{{true}}$ <= {pT_edges[i + 1]}]")
    plt.ylabel(r"$\frac{\lambda_{true} - \lambda_{meas}}{\sigma_{\lambda}}$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/lambda_true_score_pT_bin_{i}.png")
    plt.close("all")

    xbins = np.linspace(-0.4, 0.4, 31)

    plt.figure(figsize=(8., 8.))
    plt.plot(theta[:, 3 * i + 1], score[:, 3 * i + 1], "bo")
    plt.plot(xbins, 2. * ones, "r--", label=r"$2\sigma$ interval")
    plt.plot(xbins, -2. * ones, "r--")
    plt.ylim(-5., 5.)
    plt.xlabel(fr"$\mu_{{true}}$ [{pT_edges[i]} < $p_{{T}}^{{true}}$ <= {pT_edges[i + 1]}]")
    plt.ylabel(r"$\frac{\mu_{true} - \mu_{meas}}{\sigma_{\mu}}$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/mu_true_score_pT_bin_{i}.png")
    plt.close("all")

    plt.figure(figsize=(8., 8.))
    plt.plot(theta[:, 3 * i + 2], score[:, 3 * i + 2], "bo")
    plt.plot(xbins, 2. * ones, "r--", label=r"$2\sigma$ interval")
    plt.plot(xbins, -2. * ones, "r--")
    plt.ylim(-5., 5.)
    plt.xlabel(fr"$\nu_{{true}}$ [{pT_edges[i]} < $p_{{T}}^{{true}}$ <= {pT_edges[i + 1]}]")
    plt.ylabel(r"$\frac{\nu_{true} - \nu_{meas}}{\sigma_{\nu}}$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/nu_true_score_pT_bin_{i}.png")
    plt.close("all")

    plt.figure(figsize=(8., 8.))
    plt.plot(theta[:, 3 * i + 0], error[:, 3 * i + 0], "bo")
    plt.ylim(0., 0.5)
    plt.xlabel(fr"$\lambda_{{true}}$ [{pT_edges[i]} < $p_{{T}}^{{true}}$ <= {pT_edges[i + 1]}]")
    plt.ylabel(r"$\sigma_{\lambda}$")
    plt.tight_layout()
    plt.savefig(f"./plots/lambda_true_error_pT_bin_{i}.png")
    plt.close("all")

    plt.figure(figsize=(8., 8.))
    plt.plot(theta[:, 3 * i + 1], error[:, 3 * i + 1], "bo")
    plt.ylim(0., 0.2)
    plt.xlabel(fr"$\mu_{{true}}$ [{pT_edges[i]} < $p_{{T}}^{{true}}$ <= {pT_edges[i + 1]}]")
    plt.ylabel(r"$\sigma_{\mu}$")
    plt.tight_layout()
    plt.savefig(f"./plots/mu_true_error_pT_bin_{i}.png")
    plt.close("all")

    plt.figure(figsize=(8., 8.))
    plt.plot(theta[:, 3 * i + 2], error[:, 3 * i + 2], "bo")
    plt.ylim(0., 0.2)
    plt.xlabel(fr"$\nu_{{true}}$ [{pT_edges[i]} < $p_{{T}}^{{true}}$ <= {pT_edges[i + 1]}]")
    plt.ylabel(r"$\sigma_{\nu}$")
    plt.tight_layout()
    plt.savefig(f"./plots/nu_true_error_pT_bin_{i}.png")
    plt.close("all")
