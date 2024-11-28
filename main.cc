#include <TSystem.h>
#include <iostream>


int main() {

    gSystem->mkdir("./plots/", 1);

    std::cout << "[ ===> build directories ]" << std::endl;

    gSystem->ChangeDirectory("./simulators/");
    gSystem->CompileMacro("simulator.cc", "kfgO", "lib_simulator");
    gSystem->ChangeDirectory("../");

    std::cout << "[ ===> forward simulation ]" << std::endl;

    gSystem->Exec("python simulation.py");

    std::cout << "[ ===> uncertainty ]" << std::endl;

    gSystem->Exec("python uncertainty.py");

    return 0;
}
