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

    posteriors->Branch("thetas", thetas, "theta[3]/D");
    posteriors->Branch("measures", measures, "measures[3]/D");
    posteriors->Branch("errors", errors, "errors[3]/D");
    // posteriors->Branch("chisqs", chisqs, "chisq[3]/D");

    can = new TCanvas("can", "can", 800, 800);
}


void fit2D::eval(TRandom3* generator, int ndata) {

    TFile* outfile = new TFile("./data/eval.root", "recreate");

    TTree* outtree = new TTree("tree", "tree");

    double priors[3];

    outtree->Branch("Xs", Xs, "Xs[1][10][10]/D");
    outtree->Branch("priors", priors, "priors[3]/D");

    for(int ii = 0; ii < ndata; ii++) {
        priors[0] = generator->Uniform(-1., 1.);
        priors[1] = generator->Uniform(-0.5, 0.5);
        priors[2] = generator->Uniform(-0.5, 0.5);
        outtree->Fill();
    }

    outfile->Write();
    outfile->Close();
}


void fit2D::fit(int ii) {

    TFile* preds = TFile::Open("./data/weights.root", "read");
    TTree* tree = (TTree*)preds->Get("tree");

    double lambda, mu, nu, weight;

    tree->SetBranchAddress("lambda", &lambda);
    tree->SetBranchAddress("mu", &mu);
    tree->SetBranchAddress("nu", &nu);
    tree->SetBranchAddress("weight", &weight);

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

    lambda_hist->Scale(1./lambda_hist->Integral());

    /*
    lambda_fit->SetParameter(0, 1.);
    lambda_fit->SetParameter(1, lambda_hist->GetMean());
    lambda_fit->SetParameter(2, lambda_hist->GetStdDev());

    lambda_hist->Fit(lambda_fit);
    */

    measures[0] = lambda_hist->GetMean();
    errors[0] = 2. * lambda_hist->GetStdDev();
    // chisqs[0] = lambda_fit->GetChisquare()/lambda_fit->GetNDF();

    mu_hist->Scale(1./mu_hist->Integral());

    /*
    mu_fit->SetParameter(0, 1.);
    mu_fit->SetParameter(1, mu_hist->GetMean());
    mu_fit->SetParameter(2, mu_hist->GetStdDev());

    mu_hist->Fit(mu_fit);
    */

    measures[1] = mu_hist->GetMean();
    errors[1] = 2. * mu_hist->GetStdDev();
    // chisqs[1] = mu_fit->GetChisquare()/mu_fit->GetNDF();

    nu_hist->Scale(1./nu_hist->Integral());

    /*
    nu_fit->SetParameter(0, 1.);
    nu_fit->SetParameter(1, nu_hist->GetMean());
    nu_fit->SetParameter(2, nu_hist->GetStdDev());

    nu_hist->Fit(nu_fit);
    */
    measures[2] = nu_hist->GetMean();
    errors[2] = 2. * nu_hist->GetStdDev();
    // chisqs[2] = nu_fit->GetChisquare()/nu_fit->GetNDF();

    lambda_score->Fill((thetas[0] - measures[0])/errors[0]);
    mu_score->Fill((thetas[1] - measures[1])/errors[1]);
    nu_score->Fill((thetas[2] - measures[2])/errors[2]);

    sigma_lambda->Fill(errors[0]);
    sigma_mu->Fill(errors[1]);
    sigma_nu->Fill(errors[2]);

    /*
    chisq_lambda->Fill(chisqs[0]);
    chisq_mu->Fill(chisqs[1]);
    chisq_nu->Fill(chisqs[2]);
    */

    lambda_true_score->Fill(thetas[0], (thetas[0] - measures[0])/errors[0]);
    mu_true_score->Fill(thetas[1], (thetas[1] - measures[1])/errors[1]);
    nu_true_score->Fill(thetas[2], (thetas[2] - measures[2])/errors[2]);

    lambda_true_error->Fill(thetas[0], errors[0]);
    mu_true_error->Fill(thetas[1], errors[1]);
    nu_true_error->Fill(thetas[2], errors[2]);

    posteriors->Fill();

    if(ii < 50) {
        lambda_score_g1->SetPoint(ii, thetas[0], measures[0]);
        lambda_score_g1->SetPointError(ii, 0., errors[0]);

        mu_score_g1->SetPoint(ii, thetas[1], measures[1]);
        mu_score_g1->SetPointError(ii, 0., errors[1]);

        nu_score_g1->SetPoint(ii, thetas[2], measures[2]);
        nu_score_g1->SetPointError(ii, 0., errors[2]);
    }

    if(ii < 5) {

        lambda_hist->Draw("HIST");

        TLine* lambda_line = new TLine(thetas[0], 0., thetas[0], lambda_hist->GetMaximum());
        lambda_line->SetLineColor(2);
        lambda_line->SetLineWidth(2);
        lambda_line->SetLineStyle(2);

        lambda_line->Draw("SAME");

        TLine* lambda_ci1 = new TLine(measures[0] - errors[0], 0., measures[0] - errors[0], lambda_hist->GetMaximum());
        lambda_ci1->SetLineColor(4);
        lambda_ci1->SetLineWidth(2);
        lambda_ci1->SetLineStyle(1);

        TLine* lambda_ci2 = new TLine(measures[0] + errors[0], 0., measures[0] + errors[0], lambda_hist->GetMaximum());
        lambda_ci2->SetLineColor(4);
        lambda_ci2->SetLineWidth(2);
        lambda_ci2->SetLineStyle(1);

        lambda_ci1->Draw("SAME");
        lambda_ci2->Draw("SAME");

        TString lambda_save = Form("./plots/lambda_%d.png", ii);
        can->Update();
        can->SaveAs(lambda_save.Data());

        mu_hist->Draw("HIST");

        TLine* mu_line = new TLine(thetas[1], 0., thetas[1], mu_hist->GetMaximum());
        mu_line->SetLineColor(2);
        mu_line->SetLineWidth(2);
        mu_line->SetLineStyle(2);

        mu_line->Draw("SAME");

        TLine* mu_ci1 = new TLine(measures[1] - errors[1], 0., measures[1] - errors[1], mu_hist->GetMaximum());
        mu_ci1->SetLineColor(4);
        mu_ci1->SetLineWidth(2);
        mu_ci1->SetLineStyle(1);

        TLine* mu_ci2 = new TLine(measures[1] + errors[1], 0., measures[1] + errors[1], mu_hist->GetMaximum());
        mu_ci2->SetLineColor(4);
        mu_ci2->SetLineWidth(2);
        mu_ci2->SetLineStyle(1);

        mu_ci1->Draw("SAME");
        mu_ci2->Draw("SAME");

        TString mu_save = Form("./plots/mu_%d.png", ii);
        can->Update();
        can->SaveAs(mu_save.Data());

        nu_hist->Draw("HIST");

        TLine* nu_line = new TLine(thetas[2], 0., thetas[2], nu_hist->GetMaximum());
        nu_line->SetLineColor(2);
        nu_line->SetLineWidth(2);
        nu_line->SetLineStyle(2);

        nu_line->Draw("SAME");

        TLine* nu_ci1 = new TLine(measures[2] - errors[2], 0., measures[2] - errors[2], nu_hist->GetMaximum());
        nu_ci1->SetLineColor(4);
        nu_ci1->SetLineWidth(2);
        nu_ci1->SetLineStyle(1);

        TLine* nu_ci2 = new TLine(measures[2] + errors[2], 0., measures[2] + errors[2], nu_hist->GetMaximum());
        nu_ci2->SetLineColor(4);
        nu_ci2->SetLineWidth(2);
        nu_ci2->SetLineStyle(1);

        nu_ci1->Draw("SAME");
        nu_ci2->Draw("SAME");

        TString nu_save = Form("./plots/nu_%d.png", ii);
        can->Update();
        can->SaveAs(nu_save.Data());

        delete lambda_line;
        delete mu_line;
        delete nu_line;

        delete lambda_ci1;
        delete lambda_ci2;
        delete mu_ci1;
        delete mu_ci2;
        delete nu_ci1;
        delete nu_ci2;
    }

    delete lambda_hist;
    delete mu_hist;
    delete nu_hist;
    /*
    delete lambda_fit;
    delete mu_fit;
    delete nu_fit;
    */
}


void fit2D::run(TTree* tree, TRandom3* generator, int ndata) {

    tree->SetBranchAddress("Xs", Xs);
    tree->SetBranchAddress("thetas", thetas);

    lambda_score = new TH1D("lambda_score", "; #frac{#lambda_{true} - #lambda_{meas}}{#sigma_{#lambda}}; counts", 30, -5., 5.);
    mu_score = new TH1D("mu_score", "; #frac{#mu_{true} - #mu_{meas}}{#sigma_{#mu}}; counts", 30, -5., 5.);
    nu_score = new TH1D("nu_score", "; #frac{#nu_{true} - #nu_{meas}}{#sigma_{#nu}}; counts", 30, -5., 5.);

    sigma_lambda = new TH1D("sigma_lambda", "; #sigma_{#lambda}; counts", 30, 0., 2.);
    sigma_mu = new TH1D("sigma_mu", "; #sigma_{#mu}; counts", 30, 0., 2.);
    sigma_nu = new TH1D("sigma_nu", "; #sigma_{#nu}; counts", 30, 0., 2.);

    /*
    chisq_lambda = new TH1D("chisq_lambda", "; #chi^{2}/DOF; counts", 30, 0., 2.);
    chisq_mu = new TH1D("chisq_mu", "; #chi^{2}/DOF; counts", 30, 0., 2.);
    chisq_nu = new TH1D("chisq_nu", "; #chi^{2}/DOF; counts", 30, 0., 2.);
    */

    lambda_score_g1 = new TGraphErrors();
    mu_score_g1 = new TGraphErrors();
    nu_score_g1 = new TGraphErrors();

    lambda_true_error = new TH2D("lambda_true_error", "; #lambda_{true}; #lambda_{error}", 30, -1., 1., 30, 0., 1.);
    mu_true_error = new TH2D("mu_true_error", "; #mu_{true}; #mu_{error}", 30, -0.5, 0.5, 30, 0., 1.);
    nu_true_error = new TH2D("nu_true_error", "; #nu_{true}; #nu_{error}", 30, -0.5, 0.5, 30, 0., 1.);

    lambda_true_score = new TH2D("lambda_true_score", "; #lambda_{true}; #frac{#lambda_{true} - #lambda_{meas}}{#lmabda_{error}}", 30, -1., 1., 30, 0., 2.);
    mu_true_score = new TH2D("mu_true_score", "; #mu_{true}; #frac{#mu_{true} - #mu_{meas}}{#mu_{error}}", 30, -0.5, 0.5, 30, 0., 2.);
    nu_true_score = new TH2D("nu_true_score", "; #nu_{true}; #frac{#nu_{true} - #nu_{meas}}{#nu_{error}}", 30, -0.5, 0.5, 30, 0., 2.);

    for(int ii = 0; ii < 50; ii++) {
        tree->GetEntry(ii);
        eval(generator, ndata);
        gSystem->Exec("python inference.py");
        fit(ii);
    }
}


void fit2D::plots() {

    lambda_score->Draw("HIST");
    can->Update();
    can->SaveAs("./plots/lambda_norm.png");

    mu_score->Draw("HIST");
    can->Update();
    can->SaveAs("./plots/mu_norm.png");

    nu_score->Draw("HIST");
    can->Update();
    can->SaveAs("./plots/nu_norm.png");

    sigma_lambda->Draw("HIST");
    can->Update();
    can->SaveAs("./plots/sigma_lambda.png");

    sigma_mu->Draw("HIST");
    can->Update();
    can->SaveAs("./plots/sigma_mu.png");

    sigma_nu->Draw("HIST");
    can->Update();
    can->SaveAs("./plots/sigma_nu.png");

    lambda_true_error->Draw("COLZ");
    can->Update();
    can->SaveAs("./plots/lambda_true_error.png");

    mu_true_error->Draw("COLZ");
    can->Update();
    can->SaveAs("./plots/mu_true_error.png");

    nu_true_error->Draw("COLZ");
    can->Update();
    can->SaveAs("./plots/nu_true_error.png");

    lambda_true_score->Draw("COLZ");
    can->Update();
    can->SaveAs("./plots/lambda_true_score.png");

    mu_true_score->Draw("COLZ");
    can->Update();
    can->SaveAs("./plots/mu_true_score.png");

    nu_true_score->Draw("COLZ");
    can->Update();
    can->SaveAs("./plots/nu_true_score.png");

    lambda_score_g1->SetNameTitle("lmabda_score_g1", "; #lambda_{true}; #lambda_{meas}");
    lambda_score_g1->SetMarkerColor(2);
    lambda_score_g1->SetMarkerStyle(21);

    TLine* lambda_line = new TLine(-1., -1., 1., 1.);
    lambda_line->SetLineColor(2);
    lambda_line->SetLineWidth(2);
    lambda_line->SetLineStyle(2);

    lambda_score_g1->Draw("APE1");
    lambda_line->Draw("SAME");
    can->Update();
    can->SaveAs("./plots/lambda_score_graph.png");

    mu_score_g1->SetNameTitle("mu_score_g1", "; #mu_{true}; #mu_{meas}");
    mu_score_g1->SetMarkerColor(2);
    mu_score_g1->SetMarkerStyle(21);

    TLine* mu_line = new TLine(-0.5, -0.5, 0.5, 0.5);
    mu_line->SetLineColor(2);
    mu_line->SetLineWidth(2);
    mu_line->SetLineStyle(2);

    mu_score_g1->Draw("APE1");
    mu_line->Draw("SAME");
    can->Update();
    can->SaveAs("./plots/mu_score_graph.png");

    nu_score_g1->SetNameTitle("nu_score_g1", "; #nu_{true}; #nu_{meas}");
    nu_score_g1->SetMarkerColor(2);
    nu_score_g1->SetMarkerStyle(21);

    TLine* nu_line = new TLine(-0.5, -0.5, 0.5, 0.5);
    nu_line->SetLineColor(2);
    nu_line->SetLineWidth(2);
    nu_line->SetLineStyle(2);

    nu_score_g1->Draw("APE1");
    nu_line->Draw("SAME");
    can->Update();
    can->SaveAs("./plots/nu_score_graph.png");

    delete lambda_line;
    delete mu_line;
    delete nu_line;
}
