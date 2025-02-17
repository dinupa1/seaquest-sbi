#include <TSystem.h>
#include <iostream>


int __build__() {

    gSystem->CompileMacro("simulator.cc", "kfgO", "lib_simulator");

    return 0;
}
