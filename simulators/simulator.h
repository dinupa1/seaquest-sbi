#ifndef _SIMULATOR__H_
#define _SIMULATOR__H_

#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TH3D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TString.h>
#include <TSystem.h>
#include <iostream>

double pi = TMath::Pi();

double cross_section(double lambda, double mu, double nu, double phi, double costh);

void forward_simulation(int seed);

class simulator {
    double X[1][10][10];
    double theta[3];
    double theta0[3];
    TTree* tree;
    int n_data = 15000;
public:
    simulator(TString tname);
    virtual ~simulator(){;}
    void train_samples(TTree* inputs, TTree* prior, TRandom3* generator);
    void test_samples(TTree* inputs, TTree* prior, TRandom3* generator);
    void save(){tree->Write();}
};

#endif /* _SIMULATOR__H_ */

