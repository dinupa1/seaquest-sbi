#ifndef _FITTER__H_
#define _FITTER__H_

#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TF1.h>
#include <TGraphErrors.h>
#include <TRandom3.h>
#include <TSystem.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TString.h>
#include <TLine.h>
#include <iostream>

double gaussian_function(double x, double a, double mu, double sigma);

class fit2D {
    TTree* posteriors;
    TH1D* lambda_score;
    TH1D* mu_score;
    TH1D* nu_score;
    TH1D* sigma_lambda;
    TH1D* sigma_mu;
    TH1D* sigma_nu;
    // TH1D* chisq_lambda;
    // TH1D* chisq_mu;
    // TH1D* chisq_nu;
    TGraphErrors* lambda_score_g1;
    TGraphErrors* mu_score_g1;
    TGraphErrors* nu_score_g1;
    TH2D* lambda_true_error;
    TH2D* mu_true_error;
    TH2D* nu_true_error;
    TH2D* lambda_true_score;
    TH2D* mu_true_score;
    TH2D* nu_true_score;
    double Xs[1][10][10];
    double thetas[3];
    double measures[3];
    double errors[3];
    // double chisqs[3];
    TCanvas* can;
public:
    fit2D();
    virtual ~fit2D(){;}
    void eval(TRandom3* generator, int ndata);
    void fit(int ii);
    void run(TTree* tree, TRandom3* generator, int ndata);
    void plots();
    void done(){posteriors->Write();}
};

#endif /* _FITTER__H_ */
