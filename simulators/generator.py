import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

from sklearn.model_selection import train_test_split

seed: int=42

np.random.seed(seed)

save = uproot.open("./data/data.root:save")
branches = ["mass", "pT", "xF", "phi", "costh", "true_mass", "true_pT", "true_xF", "true_phi", "true_costh", "occuD1"]
events = save.arrays(branches)

events1 = events[(events.mass > 4.5) & (events.mass < 9.) & (events.xF > 0.) & (events.xF < 1.) & (np.abs(events.costh) < 0.4) & (events.occuD1 < 300.)]

X_train_val, X_test = train_test_split(events1.to_numpy(), test_size=0.3, shuffle=True)
X_train, X_val = train_test_split(X_train_val, test_size=0.5, shuffle=True)


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

outputs.close()
