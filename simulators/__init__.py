import ROOT

ROOT.gSystem.CompileMacro("./simulator/simulator.cc", "fgO", "simulator")

ROOT.gInterpreter.ProcessLine('#include "./simulator/simulator.h"')
ROOT.gSystem.Load("./simulator/simulator.so")


from ROOT import simulator
from ROOT import forward_simulation
from .generator import generator
