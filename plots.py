import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

import mplhep as hep

plt.style.use([hep.style.ROOT, hep.style.firamath])

#imports
tree = uproot.open("./data/hyperparameter0.root:tree")
lambda_mean = tree["lambda_mean"].array().to_numpy()
lambda_std = tree["lambda_error"].array().to_numpy()

mu_mean = tree["mu_mean"].array().to_numpy()
mu_std = tree["mu_error"].array().to_numpy()

nu_mean = tree["nu_mean"].array().to_numpy()
nu_std = tree["nu_error"].array().to_numpy()

train_size = tree["train_size"].array().to_numpy()
batch_size = tree["batch_size"].array().to_numpy()

train_size1 = np.unique(train_size)

for i in range(len(train_size1)):

    plt.figure(figsize=(8., 8.))

    plt.errorbar(batch_size[train_size==train_size1[i]], lambda_mean[train_size==train_size1[i]], yerr=lambda_std[train_size==train_size1[i]], fmt="bs", label=rf"$\lambda$ [{train_size1[i]}]")
    plt.errorbar(batch_size[train_size==train_size1[i]], mu_mean[train_size==train_size1[i]], yerr=mu_std[train_size==train_size1[i]], fmt="rs", label=rf"$\mu$ [{train_size1[i]}]")
    plt.errorbar(batch_size[train_size==train_size1[i]], nu_mean[train_size==train_size1[i]], yerr=nu_std[train_size==train_size1[i]], fmt="gs", label=rf"$\nu$ [{train_size1[i]}]")
    plt.xlabel("batch size")
    plt.ylabel("values")
    plt.ylim(-5., 5.)
    plt.legend(frameon=False, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"./plots/hyper_para0{i}.png")





tree = uproot.open("./data/hyperparameter1.root:tree")
lambda_mean = tree["lambda_mean"].array().to_numpy()
lambda_std = tree["lambda_error"].array().to_numpy()

mu_mean = tree["mu_mean"].array().to_numpy()
mu_std = tree["mu_error"].array().to_numpy()

nu_mean = tree["nu_mean"].array().to_numpy()
nu_std = tree["nu_error"].array().to_numpy()

sample_size = tree["sample_size"].array().to_numpy()


plt.figure(figsize=(8., 8.))

plt.errorbar(sample_size, lambda_mean, yerr=lambda_std, fmt="bs", label=r"$\lambda$")
plt.errorbar(sample_size, mu_mean, yerr=mu_std, fmt="rs", label=r"$\mu$")
plt.errorbar(sample_size, nu_mean, yerr=nu_std, fmt="gs", label=r"$\nu$")
plt.xlabel("sample size")
plt.ylabel("values")
plt.ylim(-5., 5.)
plt.legend(frameon=False, fontsize=16)
plt.tight_layout()
plt.savefig("./plots/hyper_para1.png")


tree = uproot.open("./data/hyperparameter2.root:tree")
lambda_mean = tree["lambda_mean"].array().to_numpy()
lambda_std = tree["lambda_error"].array().to_numpy()

mu_mean = tree["mu_mean"].array().to_numpy()
mu_std = tree["mu_error"].array().to_numpy()

nu_mean = tree["nu_mean"].array().to_numpy()
nu_std = tree["nu_error"].array().to_numpy()

data_size = tree["data_size"].array().to_numpy()


plt.figure(figsize=(8., 8.))

plt.errorbar(data_size, lambda_mean, yerr=lambda_std, fmt="bs", label=r"$\lambda$")
plt.errorbar(data_size, mu_mean, yerr=mu_std, fmt="rs", label=r"$\mu$")
plt.errorbar(data_size, nu_mean, yerr=nu_std, fmt="gs", label=r"$\nu$")
plt.xlabel("data size")
plt.ylabel("values")
plt.ylim(-5., 5.)
plt.legend(frameon=False, fontsize=16)
plt.tight_layout()
plt.savefig("./plots/hyper_para2.png")


tree = uproot.open("./data/eval.root:tree")
lambda_mean = tree["lambda_mean"].array().to_numpy()
lambda_std = tree["lambda_error"].array().to_numpy()

mu_mean = tree["mu_mean"].array().to_numpy()
mu_std = tree["mu_error"].array().to_numpy()

nu_mean = tree["nu_mean"].array().to_numpy()
nu_std = tree["nu_error"].array().to_numpy()

lambda_true = tree["lambda_true"].array().to_numpy()
mu_true = tree["mu_true"].array().to_numpy()
nu_true = tree["nu_true"].array().to_numpy()

bins = np.linspace(-5., 5., 21)

plt.figure(figsize=(8., 8.))
plt.hist((lambda_true - lambda_mean)/lambda_std, bins=bins, histtype="step")
plt.xlabel(r"$\lambda$")
plt.ylabel("counts")
plt.tight_layout()
plt.savefig("./plots/lambda.png")

plt.figure(figsize=(8., 8.))
plt.hist((mu_true - mu_mean)/mu_std,  bins=bins, histtype="step")
plt.xlabel(r"$\mu$")
plt.ylabel("counts")
plt.tight_layout()
plt.savefig("./plots/mu.png")

plt.figure(figsize=(8., 8.))
plt.hist((nu_true - nu_mean)/nu_std, bins=bins, histtype="step")
plt.xlabel(r"$\nu$")
plt.ylabel("counts")
plt.tight_layout()
plt.savefig("./plots/nu.png")
