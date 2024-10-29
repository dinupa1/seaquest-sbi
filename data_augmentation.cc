R__LOAD_LIBRARY(./simulators/lib_simulator.so)
R__LOAD_LIBRARY(./fitters/lib_fitter.so)

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
#include "fitters/fitter.h"


void data_augmentation() {

    int N_train = 10;
    int N_test = 10;
    int N_data = 15000;
    int seed = 42;

    auto generator = new TRandom3(seed);

    /*
    std::cout << "[ ===> Forward simulation ]" << std::endl;

    gSystem->Exec("python ./simulators/generator.py");

    auto infile = TFile::Open("./data/generator.root", "read");
    auto X_train = (TTree*)infile->Get("X_train");
    auto X_val = (TTree*)infile->Get("X_val");
    auto X_test = (TTree*)infile->Get("X_test");

    auto outfile = new TFile("./data/outputs.root", "recreate");

    auto sim1 = new simulator2D("train_tree");
    sim1->samples(X_train, generator, N_train, N_data);
    sim1->save();

    auto sim2 = new simulator2D("val_tree");
    sim2->samples(X_val, generator, N_train, N_data);
    sim2->save();

    auto sim3 = new simulator2D("test_tree");
    sim3->samples(X_test, generator, N_test, N_data);
    sim3->save();

    outfile->Write();
    outfile->Close();
    */

    // gSystem->Exec("python train_ratio_model.py");

    auto infile2 = TFile::Open("./data/outputs.root", "read");
    auto test_tree = (TTree*)infile2->Get("test_tree");

    auto outfile2 = new TFile("./data/final.root", "recreate");

    auto fit1 = new fit2D();
    fit1->run(test_tree, generator, N_data);
    fit1->plots();
    fit1->done();

    outfile2->Write();
    outfile2->Close();

}
