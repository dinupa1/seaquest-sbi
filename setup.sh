#!/bin/bash

# source /cvmfs/sft.cern.ch/lcg/views/LCG_105_cuda/x86_64-el9-gcc11-opt/setup.sh

function build-simulation {
    echo "[ ===> building simulation ]"
    cd ./simulators/
    python setup.py
    cd ../
}


function simulation {
    echo "[ ===> simulation ]"
    python ./simulations.py
}


function inference {
    echo "[ ===> inference ]"
    python ./inference.py
}


function uncertainty {
    echo "[ ===> uncertainty ]"
    python ./uncertainty.py
}
