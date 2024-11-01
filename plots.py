import numpy as np

import ROOT

from plots import plots2D

ROOT.gStyle.SetOptFit(111);

infile = ROOT.TFile.Open("~/Downloads/eval.root", "read")
tree = infile.Get("tree")
priors = infile.Get("priors")


fig = plots2D()
fig.fill(tree, priors)
fig.plots()
