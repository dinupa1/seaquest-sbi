import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

from sklearn.model_selection import train_test_split


tree_P = uproot.open("/seaquest/users/knagai/Public/data/E906/simple_str/4pi/67_LH2_4pi/data.root:save")
tree_D = uproot.open("/seaquest/users/knagai/Public/data/E906/simple_str/mc/LH2_run3_messy_1201_2398/data.root:save")


events_P = tree_P.arrays(["mass", "pT", "x1", "x2", "xF", "phi", "costh", "occuD1"])
events_D = tree_D.arrays(["mass", "pT", "xF", "phi", "costh", "true_phi", "true_costh", "occuD1", "true_mass"])

events_P1 = events_P[(events_P.mass > 5.5) & (events_P.occuD1 < 300.)].to_numpy()


data_P0, data_P1 = train_test_split(events_P1, test_size=0.5, shuffle=True)

data_train_val_P, data_test_fit_P = train_test_split(data_P0, test_size=0.5, shuffle=True)
data_train_P, data_val_P = train_test_split(data_train_val_P, test_size=0.5, shuffle=True)
data_test_P, data_fit_P = train_test_split(data_test_fit_P, test_size=0.5, shuffle=True)

train_dic_P = {
    "mass": data_train_P["mass"],
    "pT": data_train_P["pT"],
    "x1": data_train_P["x1"],
    "x2": data_train_P["x2"],
    "xF": data_train_P["xF"],
    "phi": data_train_P["phi"],
    "costh": data_train_P["costh"],
}

val_dic_P = {
    "mass": data_val_P["mass"],
    "pT": data_val_P["pT"],
    "x1": data_val_P["x1"],
    "x2": data_val_P["x2"],
    "xF": data_val_P["xF"],
    "phi": data_val_P["phi"],
    "costh": data_val_P["costh"],
}

test_dic_P = {
    "mass": data_test_P["mass"],
    "pT": data_test_P["pT"],
    "x1": data_test_P["x1"],
    "x2": data_test_P["x2"],
    "xF": data_test_P["xF"],
    "phi": data_test_P["phi"],
    "costh": data_test_P["costh"],
}

fit_dic_P = {
    "mass": data_fit_P["mass"],
    "pT": data_fit_P["pT"],
    "x1": data_fit_P["x1"],
    "x2": data_fit_P["x2"],
    "xF": data_fit_P["xF"],
    "phi": data_fit_P["phi"],
    "costh": data_fit_P["costh"],
}

events_D1 = events_D[(events_D.true_mass > 5.5) & (events_D.occuD1 < 300.)].to_numpy()

data_train_val_D, data_test_fit_D = train_test_split(events_D1, test_size=0.5, shuffle=True)
data_train_D, data_val_D = train_test_split(data_train_val_D, test_size=0.5, shuffle=True)
data_test_D, data_fit_D = train_test_split(data_test_fit_D, test_size=0.5, shuffle=True)

train_dic_D = {
    "mass": data_train_D["mass"],
    "pT": data_train_D["pT"],
    "x1": data_train_D["x1"],
    "x2": data_train_D["x2"],
    "xF": data_train_D["xF"],
    "phi": data_train_D["phi"],
    "costh": data_train_D["costh"],
}

val_dic_D = {
    "mass": data_val_D["mass"],
    "pT": data_val_D["pT"],
    "x1": data_val_D["x1"],
    "x2": data_val_D["x2"],
    "xF": data_val_D["xF"],
    "phi": data_val_D["phi"],
    "costh": data_val_D["costh"],
}

test_dic_D = {
    "mass": data_test_D["mass"],
    "pT": data_test_D["pT"],
    "x1": data_test_D["x1"],
    "x2": data_test_D["x2"],
    "xF": data_test_D["xF"],
    "phi": data_test_D["phi"],
    "costh": data_test_D["costh"],
}

fit_dic_D = {
    "mass": data_fit_D["mass"],
    "pT": data_fit_D["pT"],
    "x1": data_fit_D["x1"],
    "x2": data_fit_D["x2"],
    "xF": data_fit_D["xF"],
    "phi": data_fit_D["phi"],
    "costh": data_fit_D["costh"],
}

outfile = uproot.recreate("train_test_data.root", compression=uproot.ZLIB(4))
outfile["train_P"] = train_dic_P
outfile["val_P"] = val_dic_P
outfile["test_P"] = test_dic_P
outfile["fit_P"] = fit_dic_P
outfile["train_D"] = train_dic_D
outfile["val_D"] = val_dic_D
outfile["test_D"] = test_dic_D
outfile["fit_D"] = fit_dic_D
outfile.close()

