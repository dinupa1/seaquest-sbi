#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_105_cuda/x86_64-el9-gcc11-opt/setup.sh

cd ./simulators

root -b -e 'gSystem->CompileMacro("simulator.cc", "kfgO", "lib_simulator")' -q

cd ..
