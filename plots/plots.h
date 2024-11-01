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


class plots2D {
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

    TH1D* lambda_chisq;
    TH1D* mu_chisq;
    TH1D* nu_chisq;

    TCanvas* can;
public:
    double meas[3];
    double errors[3];
    double chisq[3];
    double score[3];
    plots2D();
    virtual ~plots2D(){;}
    void fill(TTree* tree, TTree* prior);
    void plots();
};

#endif /* _PLOTS__H_ */
