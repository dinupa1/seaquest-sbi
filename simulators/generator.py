import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

from sklearn.model_selection import train_test_split

seed: int=42

np.random.seed(seed)

n_data:int = 2000000
n_theta:int = 3
n_prior:int = 15000

print("[ ===> train test split ]")

save = uproot.open("./data/data.root:save")
branches = ["mass", "pT", "xF", "phi", "costh", "true_mass", "true_pT", "true_xF", "true_phi", "true_costh", "occuD1"]
events = save.arrays(branches)

#
# pT = {0., 0.43, 0.67, 0.96, 2.5}
#

events1 = events[(events.mass > 4.5) & (events.mass < 9.) & (events.xF > 0.) & (events.xF < 1.) & (np.abs(events.costh) < 0.4) & (events.occuD1 < 300.)]

X_train_val, X_test = train_test_split(events1.to_numpy(), test_size=0.3, shuffle=True)
X_train, X_val = train_test_split(X_train_val, test_size=0.5, shuffle=True)

theta = np.random.uniform([-1.5, -0.6, -0.6], [1.5, 0.6, 0.6], [n_data, n_theta])

theta_train_val, theta_test = train_test_split(theta, test_size=0.3, shuffle=True)
theta_train, theta_val = train_test_split(theta_train_val, test_size=0.3, shuffle=True)

theta_test, theta_prior = train_test_split(theta_test, test_size=n_prior, shuffle=True)
theta0_val, theta1_val = train_test_split(theta_val, test_size=0.5, shuffle=True)
theta0_train, theta1_train = train_test_split(theta_train, test_size=0.5, shuffle=True)

print(f"[ ===> prior size {theta_prior.shape} ]")
print(f"[ ===> test size {theta_test.shape} ]")
print(f"[ ===> val size {theta0_val.shape}, {theta1_val.shape} ]")
print(f"[ ===> train size {theta0_train.shape}, {theta1_train.shape} ]")


outputs = uproot.recreate("./data/generator.root", compression=uproot.ZLIB(4))

outputs["X_train"] = {
    "mass": X_train["mass"],
    "pT": X_train["pT"],
    "xF": X_train["xF"],
    "phi": X_train["phi"],
    "costh": X_train["costh"],
    "true_mass": X_train["true_mass"],
    "true_pT": X_train["true_pT"],
    "true_xF": X_train["true_xF"],
    "true_phi": X_train["true_phi"],
    "true_costh": X_train["true_costh"],
    }

outputs["X_val"] = {
    "mass": X_val["mass"],
    "pT": X_val["pT"],
    "xF": X_val["xF"],
    "phi": X_val["phi"],
    "costh": X_val["costh"],
    "true_mass": X_val["true_mass"],
    "true_pT": X_val["true_pT"],
    "true_xF": X_val["true_xF"],
    "true_phi": X_val["true_phi"],
    "true_costh": X_val["true_costh"],
    }

outputs["X_test"] = {
    "mass": X_test["mass"],
    "pT": X_test["pT"],
    "xF": X_test["xF"],
    "phi": X_test["phi"],
    "costh": X_test["costh"],
    "true_mass": X_test["true_mass"],
    "true_pT": X_test["true_pT"],
    "true_xF": X_test["true_xF"],
    "true_phi": X_test["true_phi"],
    "true_costh": X_test["true_costh"],
    }

outputs["theta_train"] = {
    "theta0": theta0_train,
    "theta": theta1_train,
    }

outputs["theta_val"] = {
    "theta0": theta0_val,
    "theta": theta1_val,
    }

outputs["theta_test"] = {
    "theta": theta_test,
    }

outputs["theta_prior"] = {
    "theta0": theta_prior,
    }

outputs.close()
