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
int n_data = 10000;

double cross_section(double lambda, double mu, double nu, double phi, double costh);

// void forward_simulation(int seed);


class sim_reader {
    double pT;
    double phi;
    double costh;
    double true_pT;
    double true_phi;
    double true_costh;
    TTree* tree;
    int n_events;
public:
    sim_reader(TFile* inputs, TString tname);
    virtual ~sim_reader(){};
    void fill(double theta[3], std::unique_ptr<TH2D> &hist, std::unique_ptr<TRandom3> &generator);
};


class simulator {
    sim_reader* train_reader;
    sim_reader* test_reader;
    std::unique_ptr<TRandom3> generator;
public:
    double X[1][32][32];
    double theta[3];
    TFile* outputs;
    TTree* train_tree;
    TTree* test_tree;
    simulator();
    virtual ~simulator(){;}
    void read(double X[1][32][32], std::unique_ptr<TH2D> &hist);
    void samples(int n_train, int n_test);
    void save();
};

#endif /* _SIMULATOR__H_ */

