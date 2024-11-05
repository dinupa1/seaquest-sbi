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

class simulator2D {
    double X[1][10][10];
    double theta[3];
    double theta0[3];
    TTree* tree;
public:
    simulator2D(TString tname);
    virtual ~simulator2D(){;}
    void train_samples(TTree* inputs, TTree* prior, TRandom3* generator);
    void test_samples(TTree* inputs, TTree* prior, TRandom3* generator);
    void save(){tree->Write();}
};


class simulator3D {
    double X[4][12][12];
    double theta[12];
    double theta0[12];
    TTree* tree;
    double pT_edges[5] = {0., 0.4088, 0.64025, 0.91765, 2.5};
    double phi_edges[13];
    double costh_edges[13];
public:
    simulator3D(TString tname);
    virtual ~simulator3D(){;}
    void train_samples(TTree* inputs, TTree* prior, TRandom3* generator);
    void test_samples(TTree* inputs, TTree* prior, TRandom3* generator);
    void save(){tree->Write();}
};

#endif /* _SIMULATOR__H_ */

