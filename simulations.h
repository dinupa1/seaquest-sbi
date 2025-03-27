#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TF1.h>
#include <TSystem.h>
#include <TRandom3.h>
#include <iostream>

double cross_section_ratio(double phi, double costh, double lambda, double mu, double nu);

void prior_sample(double lambda, double mu, double nu);

void simulation_sample(TTree* tree, TH2D* hist);

void likelihood_sample(TH2D* hist);

void input_tree(TTree* tree);

void out_tree(TTree* tree);

double pi = TMath::Pi();
int num_data = 5120;
int base_size = 1024;
int train_samples = 700;
int val_samples = 200;
int test_samples = 100;

int num_bins = 12;

double mass;
double pT;
double xF;
double phi;
double costh;

double true_phi;
double true_costh;
double true_pT;
double true_xF;
double true_mass;
double weight;

double theta[3];
double X[1][12][12];
double weight1;

TRandom3* events;

double lambda_limits = 3.0;
double mu_limits = 1.0;
double nu_limits = 1.0;
double costh_limits = 0.45;

TTree* train_tree;
TTree* val_tree;
TTree* test_tree;

TTree* train_out;
TTree* val_out;
TTree* test_out;
