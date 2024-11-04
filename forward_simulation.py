import numpy as np

import ROOT

from simulators import simulator2D

seed: int = 42

ROOT.gSystem.Exec("python ./simulators/simulator.py")

generator = ROOT.TRandom3(seed)

infile = ROOT.TFile.Open("./data/generator.root", "read")
X_train = infile.Get("X_train")
X_val = infile.Get("X_val")
X_test = infile.Get("X_test")

theta_train = infile.Get("theta_train")
theta_val = infile.Get("theta_val")
theta_test = infile.Get("theta_test")

outfile = ROOT.TFile("./data/outputs.root", "recreate")

sim1 = simulator2D("train_tree")
sim1.train_samples(X_train, theta_train, generator)
sim1.save()


sim2 = simulator2D("val_tree")
sim2.train_samples(X_val, theta_val, generator)
sim2.save()

sim3 = simulator2D("test_tree")
sim3.test_samples(X_test, theta_test, generator)
sim3.save()

outfile.Write()
outfile.Close()
