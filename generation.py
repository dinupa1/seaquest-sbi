import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

print("[===> e906 messy mc]")

save = uproot.open("../data/LH2_messy_MC_events.root:save")
branches = ["mass", "pT", "xF", "phi", "costh", "true_mass", "true_pT", "true_xF", "true_phi", "true_costh", "occuD1"]
events = save.arrays(branches)

events_after_cuts = events[(4.5 < events.mass) & (events.mass < 8.0) & (-0.1 < events.xF) & (events.xF < 0.95) & (np.abs(events.costh) < 0.45) & (events.occuD1 < 300.) & (0.19 < events.pT) & (events.pT < 2.24)]

train_val_events, test_events = train_test_split(events_after_cuts.to_numpy(), test_size=0.2, shuffle=True)
train_events, val_events = train_test_split(train_val_events, test_size=0.25, shuffle=True)

outputs = uproot.recreate("./data/generation.root", compression=uproot.ZLIB(4))

outputs["train_tree"] = {
    "mass": train_events["mass"],
    "pT": train_events["pT"],
    "xF": train_events["xF"],
    "phi": train_events["phi"],
    "costh": train_events["costh"],
    "true_mass": train_events["true_mass"],
    "true_pT": train_events["true_pT"],
    "true_xF": train_events["true_xF"],
    "true_phi": train_events["true_phi"],
    "true_costh": train_events["true_costh"],
    "weight": np.ones(len(train_events["mass"])),
    }

outputs["val_tree"] = {
    "mass": val_events["mass"],
    "pT": val_events["pT"],
    "xF": val_events["xF"],
    "phi": val_events["phi"],
    "costh": val_events["costh"],
    "true_mass": val_events["true_mass"],
    "true_pT": val_events["true_pT"],
    "true_xF": val_events["true_xF"],
    "true_phi": val_events["true_phi"],
    "true_costh": val_events["true_costh"],
    "weight": np.ones(len(val_events["mass"])),
    }

outputs["test_tree"] = {
    "mass": test_events["mass"],
    "pT": test_events["pT"],
    "xF": test_events["xF"],
    "phi": test_events["phi"],
    "costh": test_events["costh"],
    "true_mass": test_events["true_mass"],
    "true_pT": test_events["true_pT"],
    "true_xF": test_events["true_xF"],
    "true_phi": test_events["true_phi"],
    "true_costh": test_events["true_costh"],
    "weight": np.ones(len(test_events["mass"])),
    }

outputs.close()
