import numpy as np
import matplotlib.pyplot as plt

import mplhep as hep

import uproot
import awkward as ak


plt.style.use([hep.style.ROOT, hep.style.firamath])


trees = uproot.open("./data/eval.root:trees")
theta = trees["theta"].array().to_numpy()
posterior = trees["posterior"].array().to_numpy()


for i in range(5):

    mean = np.mean(posterior[i, :, :], axis=0)
    error = np.std(posterior[i, :, :], axis=0)

    bins = np.linspace(-1.5, 1.5, 31)

    plt.figure(figsize=(8., 8.))
    plt.hist(posterior[i, :, 0], bins=bins, histtype="step", label=fr"$\lambda_{{posterior}}$ = {mean[0]:.3f} +/- {error[0]:.3f}")
    plt.axvline(x=theta[i, 0], linestyle="--", color="r", label=fr"$\lambda_{{true}}$ = {theta[i, 0]:.3f}")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$p(\lambda|x)$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/lambda_{i}.png")
    plt.close("all")

    bins = np.linspace(-0.6, 0.6, 31)

    plt.figure(figsize=(8., 8.))
    plt.hist(posterior[i, :, 1], bins=bins, histtype="step", label=fr"$\mu_{{posterior}}$ = {mean[1]:.3f} +/- {error[1]:.3f}")
    plt.axvline(x=theta[i, 1], linestyle="--", color="r", label=fr"$\mu_{{true}}$ = {theta[i, 1]:.3f}")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$p(\mu|x)$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./plots/mu_{i}.png")
    plt.close("all")

    plt.figure(figsize=(8., 8.))
    plt.hist(posterior[i, :, 2], bins=bins, histtype="step", label=fr"$\nu_{{posterior}}$ = {mean[2]:.3f} +/- {error[2]:.3f}")
    plt.axvline(x=theta[i, 2], linestyle="--", color="r", label=fr"$\nu_{{true}}$ = {theta[i, 2]:.3f}")
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"$p(\nu|x)$")
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


tree = uproot.open("./data/eval.root:tree")
theta = tree["theta"].array().to_numpy()
meas = tree["meas"].array().to_numpy()
error = tree["error"].array().to_numpy()


xvals = np.linspace(-1., 1., 50)
plt.figure(figsize=(8., 8.))
plt.errorbar(theta[:, 0], meas[:, 0], yerr=error[:, 0], fmt="bo", capsize=1.)
plt.plot(xvals, xvals, linestyle="--", color="r", label=r"$\lambda_{true} = \lambda_{meas}$")
plt.xlabel(r"$\lambda_{true}$")
plt.ylabel(r"$\lambda_{meas}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/lambda.png")
plt.close("all")


xvals = np.linspace(-0.4, 0.4, 50)
plt.figure(figsize=(8., 8.))
plt.errorbar(theta[:, 1], meas[:, 1], yerr=error[:, 1], fmt="bo", capsize=1.)
plt.plot(xvals, xvals, linestyle="--", color="r", label=r"$\mu_{true} = \mu_{meas}$")
plt.xlabel(r"$\mu_{true}$")
plt.ylabel(r"$\mu_{meas}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/mu.png")
plt.close("all")


xvals = np.linspace(-0.4, 0.4, 50)
plt.figure(figsize=(8., 8.))
plt.errorbar(theta[:, 2], meas[:, 2], yerr=error[:, 2], fmt="bo", capsize=1.)
plt.plot(xvals, xvals, linestyle="--", color="r", label=r"$\nu_{true} = \nu_{meas}$")
plt.xlabel(r"$\nu_{true}$")
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
plt.hist(score[:, 0], bins=bins, histtype="step", label=fr"$\lambda$ = {score_mean[0]:.3f} +/- {score_error[0]:.3f}")
plt.xlabel(r"$\frac{\lambda_{true} - \lambda_{meas}}{\sigma_{\lambda}}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/lambda_score.png")
plt.close("all")

plt.figure(figsize=(8., 8.))
plt.hist(score[:, 1], bins=bins, histtype="step", label=fr"$\nu$ = {score_mean[1]:.3f} +/- {score_error[1]:.3f}")
plt.xlabel(r"$\frac{\mu_{true} - \mu_{meas}}{\sigma_{\mu}}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/mu_score.png")
plt.close("all")

plt.figure(figsize=(8., 8.))
plt.hist(score[:, 2], bins=bins, histtype="step", label=fr"$\nu$ = {score_mean[2]:.3f} +/- {score_error[2]:.3f}")
plt.xlabel(r"$\frac{\nu_{true} - \nu_{meas}}{\sigma_{\nu}}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("./plots/nu_score.png")
plt.close("all")


ybins = np.linspace(-5., 5., 31)
xbins = np.linspace(-1., 1., 31)

plt.figure(figsize=(8., 8.))
plt.plot(theta[:, 0], score[:, 0], "bo")
plt.ylim(-5., 5.)
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\frac{\lambda_{true} - \lambda_{meas}}{\sigma_{\lambda}}$")
plt.tight_layout()
plt.savefig("./plots/lambda_true_score.png")
plt.close("all")

xbins = np.linspace(-0.4, 0.4, 31)

plt.figure(figsize=(8., 8.))
plt.plot(theta[:, 1], score[:, 1], "bo")
plt.ylim(-5., 5.)
plt.xlabel(r"$\mu$")
plt.ylabel(r"$\frac{\mu_{true} - \mu_{meas}}{\sigma_{\mu}}$")
plt.tight_layout()
plt.savefig("./plots/mu_true_score.png")
plt.close("all")


plt.figure(figsize=(8., 8.))
plt.plot(theta[:, 2], score[:, 2], "bo")
plt.ylim(-5., 5.)
plt.xlabel(r"$\nu$")
plt.ylabel(r"$\frac{\nu_{true} - \nu_{meas}}{\sigma_{\nu}}$")
plt.tight_layout()
plt.savefig("./plots/nu_true_score.png")
plt.close("all")

ybins = np.linspace(0., 1., 31)
xbins = np.linspace(-1., 1., 31)

plt.figure(figsize=(8., 8.))
plt.plot(theta[:, 0], error[:, 0], "bo")
plt.ylim(0., 1.)
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\sigma_{\lambda}$")
plt.tight_layout()
plt.savefig("./plots/lambda_true_error.png")
plt.close("all")


xbins = np.linspace(-0.4, 0.4, 31)

plt.figure(figsize=(8., 8.))
plt.plot(theta[:, 1], error[:, 1], "bo")
plt.ylim(0., 1.)
plt.xlabel(r"$\mu$")
plt.ylabel(r"$\sigma_{\mu}$")
plt.tight_layout()
plt.savefig("./plots/mu_true_error.png")
plt.close("all")

plt.figure(figsize=(8., 8.))
plt.plot(theta[:, 2], error[:, 2], "bo")
plt.ylim(0., 1.)
plt.xlabel(r"$\nu$")
plt.ylabel(r"$\sigma_{\nu}$")
plt.tight_layout()
plt.savefig("./plots/nu_true_error.png")
plt.close("all")
