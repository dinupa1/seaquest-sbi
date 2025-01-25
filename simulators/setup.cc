#include <TSystem.h>
#include <iostream>


int setup() {

    gSystem->CompileMacro("simulator.cc", "kfgO", "lib_simulator");

    return 0;
}
