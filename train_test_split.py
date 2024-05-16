import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

from sklearn.model_selection import train_test_split


tree_P = uproot.open("small.root:save")
tree_D = uproot.open("/seaquest/users/knagai/Public/data/E906/simple_str/mc/LH2_run3_messy_1201_2398/data.root:save")

events_P = tree_P.arrays(["mass", "pT", "xF", "phi", "costh"])
events_D = tree_D.arrays(["phi", "costh", "true_phi", "true_costh", "occuD1", "true_mass", "true_pT", "true_xF"])

data_train_val_P, data_test_P = train_test_split(events_P.to_numpy(), test_size=0.33, shuffle=True)
data_train_P, data_val_P = train_test_split(data_train_val_P, test_size=0.5, shuffle=True)

train_dic_P = {
    "mass": data_train_P["mass"],
    "pT": data_train_P["pT"],
    "xF": data_train_P["xF"],
    "phi": data_train_P["phi"],
    "costh": data_train_P["costh"],
}

val_dic_P = {
    "mass": data_val_P["mass"],
    "pT": data_val_P["pT"],
    "xF": data_val_P["xF"],
    "phi": data_val_P["phi"],
    "costh": data_val_P["costh"],
}

test_dic_P = {
    "mass": data_test_P["mass"],
    "pT": data_test_P["pT"],
    "xF": data_test_P["xF"],
    "phi": data_test_P["phi"],
    "costh": data_test_P["costh"],
}

events_D1 = events_D[(events_D.true_mass > 4.5) & (events_D.occuD1 < 300.)].to_numpy()[:int(data_fit_P["mass"].shape[0]* 0.06)]

data_train_val_D, data_test_D = train_test_split(events_D1, test_size=0.33, shuffle=True)
data_train_D, data_val_D = train_test_split(data_train_val_D, test_size=0.5, shuffle=True)

train_dic_D = {
    "true_mass": data_train_D["true_mass"],
    "true_pT": data_train_D["true_pT"],
    "true_xF": data_train_D["true_xF"],
    "phi": data_train_D["phi"],
    "costh": data_train_D["costh"],
}

val_dic_D = {
    "true_mass": data_val_D["true_mass"],
    "true_pT": data_val_D["true_pT"],
    "true_xF": data_val_D["true_xF"],
    "phi": data_val_D["phi"],
    "costh": data_val_D["costh"],
}

test_dic_D = {
    "true_mass": data_test_D["true_mass"],
    "true_pT": data_test_D["true_pT"],
    "true_xF": data_test_D["true_xF"],
    "phi": data_test_D["phi"],
    "costh": data_test_D["costh"],
}

outfile = uproot.recreate("train_test_data.root", compression=uproot.ZLIB(4))
outfile["train_P"] = train_dic_P
outfile["val_P"] = val_dic_P
outfile["test_P"] = test_dic_P
outfile["train_D"] = train_dic_D
outfile["val_D"] = val_dic_D
outfile["test_D"] = test_dic_D
outfile.close()

