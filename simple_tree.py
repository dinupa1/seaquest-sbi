import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

tree = uproot.open("/seaquest/users/knagai/Public/data/E906/simple_str/4pi/67_LH2_4pi/data.root:save")

events = tree.arrays(["mass", "pT", "x1", "x2", "xF", "phi", "costh"])

events1 = events[events.mass > 4.5].to_numpy()


tree_dic = {
    "mass": events1["mass"],
    "pT": events1["pT"],
    "x1": events1["x1"],
    "x2": events1["x2"],
    "xF": events1["xF"],
    "phi": events1["phi"],
    "costh": events1["costh"],
}

outfile = uproot.recreate("simple_data.root", compression=uproot.ZLIB(4))
outfile["save"] = tree_dic
outfile.close()