#include <TSystem.h>
#include <iostream>

int main() {
    std::cout << "[===> seaquest simulation based inference framework]" << std::endl;

    std::cout << "[===> build simulators]" << std::endl;
    gSystem->Exec("cd ./simulators/ && root -b -q setup.cc && cd ..");

    /*
    std::cout << "[===> RS67 LH2 unmix, mixed and flask events]" << std::endl;
    gSystem->Exec("python RS67_LH2_data.py");
    */

    /*
    std::cout << "[===> create phi vs. costh histograms 12x12]" << std::endl;
    gSystem->Exec("root -b -q phi_costheta.cc");
    */

    std::cout << "[===> generators]" << std::endl;
    gSystem->Exec("python ./simulators/generator.py");

    std::cout << "[===> simulation]" << std::endl;
    gSystem->Exec("python simulations.py");

    /*
    std::cout << "[===> inference]" << std::endl;
    gSystem->Exec("python inference.py");

    std::cout << "[===> uncertainty]" << std::endl;
    gSystem->Exec("python uncertainty.py");

    std::cout << "[===> RS67 LH2 inference]" << std::endl;
    gSystem->Exec("python inference_RS67_LH2_data.py");

    std::cout << "[===> RS67 LH2 uncertainty]" << std::endl;
    gSystem->Exec("python uncertainty_RS67_LH2_data.py");
    */

    return 0;
}
