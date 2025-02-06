import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

from sklearn.model_selection import train_test_split

seed: int=42

np.random.seed(seed)

print("[===> E906 messy MC]")

save = uproot.open("../data/LH2_messy_MC_data.root:save")
branches = ["mass", "pT", "xF", "phi", "costh", "true_mass", "true_pT", "true_xF", "true_phi", "true_costh", "occuD1"]
events = save.arrays(branches)

events1 = events[(events.mass > 4.5) & (events.mass < 8.) & (events.xF > -0.1) & (events.xF < 0.9) & (np.abs(events.costh) < 0.4) & (events.occuD1 < 300.) & (0.19 < events.pT) & (events.pT < 2.24)]


outputs = uproot.recreate("./data/generator.root", compression=uproot.ZLIB(4))

outputs["tree"] = {
    "mass": events1["mass"],
    "pT": events1["pT"],
    "xF": events1["xF"],
    "phi": events1["phi"],
    "costh": events1["costh"],
    "true_mass": events1["true_mass"],
    "true_pT": events1["true_pT"],
    "true_xF": events1["true_xF"],
    "true_phi": events1["true_phi"],
    "true_costh": events1["true_costh"],
    }

outputs.close()
