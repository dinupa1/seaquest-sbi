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

// void forward_simulation(int seed);

class reader {
    double pT;
    double phi;
    double costh;
    double true_pT;
    double true_phi;
    double true_costh;
    TTree* tree;
    int n_events;
    int n_data = 15000;
    double pT_edges[5] = {0., 0.4088, 0.64025, 0.91765, 2.5};
public:
    reader(TFile* inputs, TString tname);
    virtual ~reader(){};
    void fill(double theta[12], TH2D* hist, double threshold, TRandom3* generator);
};

class simulator {
    reader* train_reader;
    reader* val_reader;
    reader* test_reader;
    TRandom3* generator_0;
    TRandom3* generator_1;
    int n_data = 15000;
    double pi = TMath::Pi();
public:
    double X[1][10][10];
    double theta[12];
    double theta_0[12];
    TFile* outputs;
    TTree* train_tree;
    TTree* val_tree;
    TTree* test_tree;
    TTree* prior_tree;
    simulator();
    virtual ~simulator(){;}
    void prior(double theta[12], TRandom3* generator, double lambda_min, double lambda_max, double mu_min, double mu_max, double nu_min, double nu_max);
    void read(double X[1][10][10], TH2D* hist);
    void samples(int n_train, int n_val, int n_test);
    void save();
};

#endif /* _SIMULATOR__H_ */

