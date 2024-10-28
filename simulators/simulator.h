#ifndef _SIMULATOR__H_
#define _SIMULATOR__H_

#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TH3D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TString.h>
#include <iostream>

double cross_section(double lambda, double mu, double nu, double phi, double costh);

void forward_simulation(int seed, int train_size, int ndata, int test_size);

class simulator2D {
    double Xs[1][10][10];
    double thetas[3];
    double thetas0[3];
    TTree* tree;
public:
    simulator2D(TString tname);
    virtual ~simulator2D(){;}
    void samples(TTree* inputs, TRandom3* generator, int events, int ndata);
    void save(){tree->Write();}
};


class simulator3D {
    double Xs[4][10][10];
    double thetas[12];
    double thetas0[12];
    TTree* tree;
    double pT_edges[5] = {0., 0.4088, 0.64025, 0.91765, 2.5};
    double phi_edges[11];
    double costh_edges[11];
public:
    simulator3D(TString tname);
    virtual ~simulator3D(){;}
    void samples(TTree* inputs, TRandom3* generator, int events, int ndata);
    void save(){tree->Write();}
};

#endif /* _SIMULATOR__H_ */

