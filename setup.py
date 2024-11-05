# source /cvmfs/sft.cern.ch/lcg/views/LCG_105_cuda/x86_64-el9-gcc11-opt/setup.sh

import ROOT

ROOT.gSystem.ChangeDirectory("./simulators/")
ROOT.gSystem.CompileMacro("simulator.cc", "kfgO", "lib_simulator")
ROOT.gSystem.ChangeDirectory("../")

