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
int num_data = 2500;

double cross_section(double lambda, double mu, double nu, double phi, double costh);


class sim_reader {
    double pT;
    double phi;
    double costh;
    double true_pT;
    double true_phi;
    double true_costh;
    TTree* tree;
    int num_events;
    double pT_edges[5] = {0., 0.404, 0.633, 0.909, 2.500};
public:
    sim_reader(TFile* inputs, TString tname);
    virtual ~sim_reader(){};
    void fill(double theta[3], std::unique_ptr<TH2D> &hist, std::unique_ptr<TRandom3> &generator);
};


class simulator {
    sim_reader* rdr;
    std::unique_ptr<TRandom3> generator;
public:
    double X[1][12][12];
    double theta[3];
    TFile* outputs;
    TTree* out_tree;
    simulator();
    virtual ~simulator(){;}
    void read(double X[1][12][12], std::unique_ptr<TH2D> &hist);
    void samples(int num_samples);
    void save();
};

#endif /* _SIMULATOR__H_ */

