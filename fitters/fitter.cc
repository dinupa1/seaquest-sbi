#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TF1.h>
#include <TRandom3.h>
#include <TSystem.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TString.h>
#include <TLine.h>
#include <iostream>

#include "fitter.h"

double gaussian_function(double x, double a, double mu, double sigma) {
    double exponent = (x - mu)/sigma;
    return a * exp(-0.5 * exponent * exponent);
}



fit2D::fit2D() {
    gStyle->SetOptFit(1011);
    gStyle->SetOptStat(0);

    gSystem->mkdir("./plots/", 1);

    posteriors = new TTree("posteriors", "posteriors");

    posteriors->Branch("theta", theta, "theta[3]/D");
    posteriors->Branch("measures", measures, "measures[3]/D");
    posteriors->Branch("errors", errors, "errors[3]/D");
    posteriors->Branch("chisq", chisq, "chisq[3]/D");

    can = new TCanvas("can", "can", 800, 800);
}


void fit2D::eval(TRandom3* generator, int ndata) {

    TFile* outfile = new TFile("./data/eval.root", "recreate");

    TTree* outtree = new TTree("tree", "tree");

    double priors[3];

    outtree->Branch("Xs", Xs, "Xs[1][10][10]/D");
    outfile->Branch("priors", priors, "priors[3]/D");

    for(int ii = 0; ii < ndata; ii++) {
        priors[0] = generator->Uniform(-1., 1.);
        priors[1] = generator->Uniform(-0.5, 0.5);
        priors[2] = generator->Uniform(-0.5, 0.5);
        outtree->Fill();
    }

    outfile->Write();
    outfile->Close();
}


void fit2D::plot_fits(TH1D* lambda_hist, TH1D* mu_hist, TH1D* nu_hist, int ii) {

    lambda_hist->Draw("HIST");

    TLine* lambda_line = new TLine(theta[0], 0., theta[0], 1.);
    lambda_line->SetLineColor(2);
    lambda_line->SetLineWidth(2);
    lambda_line->SetLineStyle(2);

    lambda_line->Draw("SAME");

    TString lambda_save = Form("./plots/lambda_%d.png", ii);
    can->Update();
    can->SaveAs(lambda_save.Data());

    mu_hist->Draw("HIST");

    TLine* mu_line = new TLine(theta[1], 0., theta[1], 1.);
    mu_line->SetLineColor(2);
    mu_line->SetLineWidth(2);
    mu_line->SetLineStyle(2);

    mu_line->Draw("SAME");

    TString mu_save = Form("./plots/mu_%d.png", ii);
    can->Update();
    can->SaveAs(mu_save.Data());

    nu_hist->Draw("HIST");

    TLine* nu_line = new TLine(theta[2], 0., theta[2], 1.);
    nu_line->SetLineColor(2);
    nu_line->SetLineWidth(2);
    nu_line->SetLineStyle(2);

    nu_line->Draw("SAME");

    TString nu_save = Form("./plots/nu_%d.png", ii);
    can->Update();
    can->SaveAs(nu_save.Data());

    delete lambda_line;
    delete mu_line;
    delete nu_line;
}


void fit2D::fit(int ii) {

    TFile* preds = TFile::Open("./data/weights.root", "read");
    TTree* tree = (TTree*)preds->Get("tree");

    double lambda, mu, nu, weight;

    tree->SetBranchAddress("lambda", &lambda);
    tree->SetBranchAddress("mu", &mu);
    tree->SetBranchAddress("nu", &nu);
    tree->SetBranchAddress("weight", weight);

    TH1D* lambda_hist = new TH1D("lambda_hist", "; #lambda; p(#lambda | x)", 30, -1., 1.);
    TH1D* mu_hist = new TH1D("mu_hist", "; #mu; p(#mu | x)", 30, -0.5, 0.5);
    TH1D* nu_hist = new TH1D("nu_hist", "; #nu; p(#nu | x)", 30, -0.5, 0.5);

    TF1* lambda_fit = new TF1("lambda_fit", "gaussian_function(x, [0], [1], [2])", -1., 1.);
    TF1* mu_fit = new TF1("mu_fit", "gaussian_function(x, [0], [1], [2])", -0.5, 0.5);
    TF1* nu_fit = new TF1("nu_fit", "gaussian_function(x, [0], [1], [2])", -0.5, 0.5);


    for(int ii = 0; ii < tree->GetEntries(); ii++) {
        tree->GetEntry(ii);
        lambda_hist->Fill(lambda, weight);
        mu_hist->Fill(mu, weight);
        nu_hist->Fill(nu, weight);
    }

    lambda_hist->Scale(1./lambda_hist->GetMaximum());

    lambda_fit->SetParameter(0, 1.);
    lambda_fit->SetParameter(1, lambda_hist->GetMean());
    lambda_fit->SetParameter(2, lambda_hist->GetStdDev());

    lambda_hist->Fit(lambda_fit);
    measures[0] = lambda_fit->GetParameter(1);
    errors[0] = lambda_fit->GetParameter(2);
    chisqs[0] = lambda_fit->GetChisquare()/lambda_fit->GetNDF();

    mu_hist->Scale(1./mu_hist->GetMaximum());

    mu_fit->SetParameter(0, 1.);
    mu_fit->SetParameter(1, mu_hist->GetMean());
    mu_fit->SetParameter(2, mu_hist->GetStdDev());

    mu_hist->Fit(mu_fit);
    measures[1] = mu_fit->GetParameter(1);
    errors[1] = mu_fit->GetParameter(2);
    chisqs[1] = mu_fit->GetChisquare()/mu_fit->GetNDF();

    nu_hist->Scale(1./nu_hist->GetMaximum());

    nu_fit->SetParameter(0, 1.);
    nu_fit->SetParameter(1, nu_hist->GetMean());
    nu_fit->SetParameter(2, nu_hist->GetStdDev());

    nu_hist->Fit(nu_fit);
    measures[2] = nu_fit->GetParameter(1);
    errors[2] = nu_fit->GetParameter(2);
    chisqs[2] = nu_fit->GetChisquare()/nu_fit->GetNDF();

    lambda_norm->Fill((thetas[0] - measures[0])/errors[0]);
    mu_norm->Fill((thetas[1] - measures[1])/errors[1]);
    nu_norm->Fill((thetas[2] - measures[2])/errors[2]);

    sigma_lambda->Fill(errors[0]);
    sigma_mu->Fill(errors[1]);
    sigma_nu->Fill(errors[2]);

    chisq_lambda->Fill(chisqs[0]);
    chisq_mu->Fill(chisqs[1]);
    chisq_nu->Fill(chisqs[2]);

    posteriors->Fill();

    if(ii < 5){plot_fits(lambda_hist, mu_hist, nu_hist, ii);}

    delete lambda_hist;
    delete mu_hist;
    delete nu_hist;
    delete lambda_fit;
    delete mu_fit;
    delete nu_fit;
}


void fit2D::run(TTree* tree, TRandom3* generator, int ndata) {

    tree->SetBranchAddress("Xs", Xs);
    tree->SetBranchAddress("thetas", thetas);

    lambda_norm = new TH1D("lambda_norm", "; #frac{#lambda_{true} - #lambda_{meas}}{#sigma_{#lambda}}; counts", 30, -5., 5.);
    mu_norm = new TH1D("mu_norm", "; #frac{#mu_{true} - #mu_{meas}}{#sigma_{#mu}}; counts", 30, -5., 5.);
    nu_norm = new TH1D("nu_norm", "; #frac{#nu_{true} - #nu_{meas}}{#sigma_{#nu}}; counts", 30, -5., 5.);

    sigma_lambda = new TH1D("sigma_lambda", "; #sigma_{#lambda}; counts", 30, 0., 2.);
    sigma_mu = new TH1D("sigma_mu", "; #sigma_{#mu}; counts", 30, 0., 2.);
    sigma_nu = new TH1D("sigma_nu", "; #sigma_{#nu}; counts", 30, 0., 2.);

    chisq_lambda = new TH1D("chisq_lambda", "; #chi^{2}/DOF; counts", 30, 0., 2.);
    chisq_mu = new TH1D("chisq_mu", "; #chi^{2}/DOF; counts", 30, 0., 2.);
    chisq_nu = new TH1D("chisq_nu", "; #chi^{2}/DOF; counts", 30, 0., 2.);

    for(int ii = 0; ii < 5; ii++) {
        tree->GetEntry(ii);

        eval(generator, ndata);

        gSystem->Exec("python inference.py");

        fit(lambda_hist, mu_hist, nu_hist, lambda_fit, mu_fit, nu_fit);
    }
}
