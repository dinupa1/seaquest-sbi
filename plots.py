import numpy as np

import ROOT

from plots import plots

ROOT.gStyle.SetOptFit(111);

infile = ROOT.TFile.Open("./data/eval.root", "read")
tree = infile.Get("tree")
prior = infile.Get("prior")


fig = plots()
fig.fill(tree, prior)
fig.plot()
