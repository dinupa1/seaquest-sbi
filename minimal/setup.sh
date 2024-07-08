#!/bin/bash


if [ -d "build" ]; then
	rm -rf build
fi


if [ -d "plots" ]; then
	rm -rf plots
fi

mkdir build
mkdir plots

cd build
root -e 'gSystem->CompileMacro("../src/Unfold.cc", "kfgO", "lib_unfold")' -q
cd ..

