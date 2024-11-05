#ifndef _PLOTS3D__H_
#define _PLOTS3D__H_

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


class plots3D {
    TH1D* lambda_score[4];
    TH1D* mu_score[4];
    TH1D* nu_score[4];

    TH1D* lambda_error[4];
    TH1D* mu_error[4];
    TH1D* nu_error[4];

    TGraphErrors* lambda_graph[4];
    TGraphErrors* mu_graph[4];
    TGraphErrors* nu_graph[4];

    TH2D* lambda_true_score[4];
    TH2D* mu_true_score[4];
    TH2D* nu_true_score[4];

    TH2D* lambda_true_error[4];
    TH2D* mu_true_error[4];
    TH2D* nu_true_error[4];

    TCanvas* can;
public:
    double meas[12];
    double errors[12];
    double chisq[12];
    double score[12];
    plots3D();
    virtual ~plots3D(){;}
    void fill(TTree* tree, TTree* prior);
    void plots();
};

#endif /* _PLOTS3D__H_ */
