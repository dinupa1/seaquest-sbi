#ifndef _PLOTS__H_
#define _PLOTS__H_

#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TH1D.h>
#include <TH3D.h>
#include <TH2D.h>
#include <TF1.h>
#include <TLine.h>
#include <TGraphErrors.h>
#include <TCanvas.h>
#include <TMath.h>
#include <TString.h>
#include <TSystem.h>
#include <iostream>

TCanvas* can;

class plots_reader {
    TH1D* hist_lambda;
    TH1D* hist_mu;
    TH1D* hist_nu;
    TGraphErrors* lambda_meas;
    TGraphErrors* mu_meas;
    TGraphErrors* nu_meas;
public:
    TTree* prior;
    int n_prior;
    double theta[3];
    double meas[3];
    double error[3];
    double score[3];
    plots_reader(TFile* inputs);
    virtual ~plots_reader(){;}
    void fill(double theta_true[3], double weights[10000]);
    void plot_one(double theta_true, double meas, double error, TH1D* hist, TGraphErrors* graph, TString pname);
    void plot(double theta_true[3], int ii);
    void histograms(double theta_true[3], TH1D* lambda_score, TH1D* mu_score, TH1D* nu_score, TH1D* lambda_error, TH1D* mu_error, TH1D* nu_error, TH2D* lambda_true_score, TH2D* mu_true_score, TH2D* nu_true_score, TH2D* lambda_true_error, TH2D* mu_true_error, TH2D* nu_true_error);
    void graphs(double theta_true[3], TGraphErrors* lambda_graph, TGraphErrors* mu_graph, TGraphErrors* nu_graph, int ii);
};


class ratio_plots {
    TH1D* lambda_score;
    TH1D* mu_score;
    TH1D* nu_score;

    TH1D* lambda_error;
    TH1D* mu_error;
    TH1D* nu_error;

    TGraphErrors* lambda_graph;
    TGraphErrors* mu_graph;
    TGraphErrors* nu_graph;

    TH2D* lambda_true_score;
    TH2D* mu_true_score;
    TH2D* nu_true_score;

    TH2D* lambda_true_error;
    TH2D* mu_true_error;
    TH2D* nu_true_error;
public:
    double theta_true[3];
    double weights[15000];
    TTree* tree;
    int n_events;
    plots_reader* rdr;
    ratio_plots();
    virtual ~ratio_plots(){;}
    void fill();
    void plot_hist1D(TH1D* hist, TString pname);
    void plot_hist2D(TH2D* hist, TString pname);
    void plot_graph(TGraphErrors* graph, double xmin, double xmax, TString gname, TString tname, TString pname);
    void plot();
};

#endif /* _PLOTS__H_ */
