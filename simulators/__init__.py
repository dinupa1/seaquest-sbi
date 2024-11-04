import ROOT

ROOT.gSystem.cd("./simulators")

ROOT.gSystem.CompileMacro("simulator.cc", "kfgO", "lib_simulator")

ROOT.gSystem.cd("../")

ROOT.gInterpreter.ProcessLine('#include "./simulators/simulator.h"')
ROOT.gSystem.Load("./simulators/lib_simulator.so")

from ROOT import simulator2D
