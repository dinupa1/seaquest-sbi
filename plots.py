import numpy as np

import ROOT

from plots import plots3D

ROOT.gStyle.SetOptFit(111);

infile = ROOT.TFile.Open("./data/eval.root", "read")
tree = infile.Get("tree")
prior = infile.Get("prior")


fig = plots3D()
fig.fill(tree, prior)
fig.plots()
