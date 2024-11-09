#include <TSystem.h>
#include <iostream>


int setup() {

    gSystem->ChangeDirectory("./simulators/");
    gSystem->CompileMacro("simulator.cc", "kfgO", "lib_simulator");
    gSystem->ChangeDirectory("../");

    gSystem->ChangeDirectory("./plots/");
    gSystem->CompileMacro("plots.cc", "kfgO", "lib_ratio_plots");
    gSystem->ChangeDirectory("../");

    return 0;
}
