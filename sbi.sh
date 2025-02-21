#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_105_cuda/x86_64-el9-gcc11-opt/setup.sh

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -d|--data)
            echo "[===> preparing RS67 LH2 data]"
            root -b -q ./phi_costheta.cc
            ;;
        -c|--compile)
            echo "[===> compile dynamic dictionaries]"
            cd ./simulators
            root -b -q ./__build__.cc
            cd ..
            ;;
        -s| --simulation)
            echo "[===> running simulations]"
            python ./simulators/generator.py
            python ./simulations.py
            ;;
        -i| --inference)
            echo "[===> running inference]"
            python inference.py
            ;;
        -b| --bootstrap)
            echo "[===> running bootstrapping method]"
            python uncertainty.py
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done
