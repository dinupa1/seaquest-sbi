#ifndef _H_SIMULATOR_H_
#define _H_SIMULATOR_H_

#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TH3D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TString.h>
#include <iostream>

double cross_section(double lambda, double mu, double nu, double phi, double costh);

void forward_simulation(int seed, int events, int ndata);

class simulator {
    double X[1][10][10];
    double theta[3];
    TTree* tree;
public:
    simulator(TString tname);
    virtual ~simulator(){;}
    void samples(TTree* inputs, TRandom3* prior, int events, int ndata); // train data
    // void samples(TTree* inputs, TRandom3* prior, double lambda, double mu, double nu); // test data
    void save(){tree->Write();}
};

#endif /* _H_SIMULATOR_H_ */

