import ROOT

ROOT.gSystem.cd("./plots")

ROOT.gSystem.CompileMacro("plots.cc", "kfgO", "lib_plots")

ROOT.gSystem.cd("../")

ROOT.gInterpreter.ProcessLine('#include "./plots/plots.h"')
ROOT.gSystem.Load("./plots/lib_plots.so")

from ROOT import plots2D
