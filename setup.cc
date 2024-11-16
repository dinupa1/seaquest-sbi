#include <TSystem.h>
#include <iostream>


int setup() {

    gSystem->ChangeDirectory("./simulators/");
    gSystem->CompileMacro("simulator.cc", "kfgO", "lib_simulator");
    gSystem->ChangeDirectory("../");

    return 0;
}
