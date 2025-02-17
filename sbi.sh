source /cvmfs/sft.cern.ch/lcg/views/LCG_105_cuda/x86_64-el9-gcc11-opt/setup.sh

# echo "[===> RS67 LH2 data]"
# root -b -q phi_costheta.cc

# echo "[===> build simulators]"
# cd ./simulators
# root -b -q __build__.cc
# cd ../
#
# echo "[===> generation]"
# python ./simulators/generator.py
#
# echo "[===> simulations]"
# python simulations.py

echo "[===> inference]"
python inference.py
#
# echo "[===> bootstrapping ]"
# python uncertainty.py
