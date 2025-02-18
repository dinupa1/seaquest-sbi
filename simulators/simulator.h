#ifndef _SIMULATOR__H_
#define _SIMULATOR__H_

#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TH3D.h>
#include <TH2D.h>
#include <TF1.h>
#include <TF2.h>
#include <TMath.h>
#include <TString.h>
#include <TSystem.h>
#include <iostream>

double cross_section(double lambda, double mu, double nu, double phi, double costh);

double pi = TMath::Pi();
int num_entries = 2000;
double lambda_min = -2.;
double lambda_max = 2.;
double mu_min = -0.5;
double mu_max = 0.5;
double nu_min = -0.5;
double nu_max = 0.5;

class reader {
    double pT;
    double phi;
    double costh;
    double true_pT;
    double true_phi;
    double true_costh;
    TTree* tree;
    int num_events;
public:
    reader(TFile* inputs, TString tname);
    virtual ~reader(){};
    void fill(double theta[3], std::unique_ptr<TH2D> &hist, std::unique_ptr<TRandom3> &generator);
};


class simulator {
    reader* train_rdr;
    reader* val_rdr;
    reader* test_rdr;
    std::unique_ptr<TRandom3> generator;
public:
    double X[1][12][12];
    double theta[3];
    TFile* outputs;
    TTree* train_tree;
    TTree* val_tree;
    TTree* test_tree;
    simulator();
    virtual ~simulator(){;}
    void read(double X[1][12][12], std::unique_ptr<TH2D> &hist);
    void samples(int train_samples, int val_samples, int test_samples);
    void save();
};

#endif /* _SIMULATOR__H_ */

