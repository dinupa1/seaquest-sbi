R__LOAD_LIBRARY(simulators/lib_simulator.so)

#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TH3D.h>
#include <TH2D.h>
#include <TSystem.h>
#include <TMath.h>
#include <TString.h>
#include <iostream>

#include "simulators/simulator.h"


void data_augmentation() {

    int N_train = 10;
    int N_test = 10;
    int N_data = 15000;
    int seed = 42;

    auto generator = new TRandom3(seed);

    std::cout << "[ ===> Forward simulation ]" << std::endl;

    gSystem->Exec("python ./simulators/generator.py");

    auto infile = TFile::Open("./data/generator.root", "read");
    auto train = (TTree*)infile->Get("train");
    auto test = (TTree*)infile->Get("test");

    auto outfile = new TFile("./data/outputs.root", "recreate");

    auto prior = new TTree("prior", "prior");

    double thetas[3];

    prior->Branch("thetas", thetas, "thetas[3]/D");

    for(int ii = 0; ii < N_data; ii++) {
        thetas[0] = generator->Uniform(-1., 1.);
        thetas[1] = generator->Uniform(-0.5, 0.5);
        thetas[2] = generator->Uniform(-0.5, 0.5);

        prior->Fill();
    }

    auto sim1 = new simulator2D("train_tree");
    sim1->samples(train, generator, N_train, N_data);
    sim1->save();

    auto sim2 = new simulator2D("test_tree");
    sim2->samples(test, generator, N_test, N_data);
    sim2->save();

    outfile->Write();
    outfile->Close();
}
