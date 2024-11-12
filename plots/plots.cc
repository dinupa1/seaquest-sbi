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

#include "plots.h"


plots_reader::plots_reader(TFile* inputs) {

    prior = (TTree*)inputs->Get("prior");
    n_prior = prior->GetEntries();

    prior->SetBranchAddress("theta_0", theta_0);

    hist_lambda = new TH1D("hist_lambda", "; #lambda; p(#lambda | x)", 20, -1., 1.);
    hist_mu = new TH1D("hist_mu", "; #mu; p(#mu | x)", 20, -0.5, 0.5);
    hist_nu = new TH1D("hist_nu", "; #nu; p(#nu | x)", 20, -0.5, 0.5);

    lambda_meas = new TGraphErrors();
    mu_meas = new TGraphErrors();
    nu_meas = new TGraphErrors();
}


void plots_reader::fill(double theta[3], double weights[10000]) {

    hist_lambda->Reset();
    hist_mu->Reset();
    hist_nu->Reset();

    for(int ii = 0; ii < n_prior; ii++) {
        prior->GetEntry(ii);
        hist_lambda->Fill(theta_0[0], weights[ii]);
        hist_mu->Fill(theta_0[1], weights[ii]);
        hist_nu->Fill(theta_0[2], weights[ii]);
    }

    meas[0] = hist_lambda->GetMean();
    meas[1] = hist_mu->GetMean();
    meas[2] = hist_nu->GetMean();

    error[0] = hist_lambda->GetStdDev();
    error[1] = hist_mu->GetStdDev();
    error[2] = hist_nu->GetStdDev();

    score[0] = (theta[0] - meas[0])/error[0];
    score[1] = (theta[1] - meas[1])/error[1];
    score[2] = (theta[2] - meas[2])/error[2];
}


void plots_reader::plot_one(double theta, double meas, double error, TH1D* hist, TGraphErrors* graph, TString pname) {

    double y_max = hist->GetMaximum();

    hist->Scale(1./y_max);

    graph->SetPoint(0, meas, 0.75);
    graph->SetPointError(0, error, 0.);

    graph->SetMarkerColor(4);
    graph->SetMarkerStyle(21);

    TLine* ll = new TLine(theta, 0., theta, 1.);
    ll->SetLineColor(2);
    ll->SetLineWidth(2);
    ll->SetLineStyle(2);

    hist->Draw("HIST");
    graph->Draw("SAME PE1");
    ll->Draw("SAME");

    can->Update();
    can->SaveAs(pname.Data());

    delete ll;
}


void plots_reader::plot(double theta[3], int ii) {

    TString pic_lambda = Form("./plots/lambda_%d.png", ii);
    plot_one(theta[0], meas[0], error[0], hist_lambda, lambda_meas, pic_lambda);

    TString pic_mu = Form("./plots/mu_%d.png", ii);
    plot_one(theta[1], meas[1], error[1], hist_mu, mu_meas, pic_mu);

    TString pic_nu = Form("./plots/nu_%d.png", ii);
    plot_one(theta[2], meas[2], error[2], hist_nu, nu_meas, pic_nu);
}


void plots_reader::histograms(double theta[3], TH1D* lambda_score, TH1D* mu_score, TH1D* nu_score, TH1D* lambda_error, TH1D* mu_error, TH1D* nu_error, TH2D* lambda_true_score, TH2D* mu_true_score, TH2D* nu_true_score, TH2D* lambda_true_error, TH2D* mu_true_error, TH2D* nu_true_error) {

    lambda_score->Fill(score[0]);
    mu_score->Fill(score[1]);
    nu_score->Fill(score[2]);

    lambda_error->Fill(error[0]);
    mu_error->Fill(error[1]);
    nu_error->Fill(error[2]);

    lambda_true_score->Fill(theta[0], score[0]);
    mu_true_score->Fill(theta[1], score[1]);
    nu_true_score->Fill(theta[2], score[2]);

    lambda_true_error->Fill(theta[0], error[0]);
    mu_true_error->Fill(theta[1], error[1]);
    nu_true_error->Fill(theta[2], error[2]);
}


void plots_reader::graphs(double theta[3], TGraphErrors* lambda_graph, TGraphErrors* mu_graph, TGraphErrors* nu_graph, int ii) {

    lambda_graph->SetPoint(ii, theta[0], meas[0]);
    lambda_graph->SetPointError(ii, 0., error[0]);

    mu_graph->SetPoint(ii, theta[1], meas[1]);
    mu_graph->SetPointError(ii, 0., error[1]);

    nu_graph->SetPoint(ii, theta[2], meas[2]);
    nu_graph->SetPointError(ii, 0., error[2]);
}


ratio_plots::ratio_plots() {

    TFile* inputs = TFile::Open("./data/eval.root", "read");

    tree = (TTree*)inputs->Get("tree");
    n_events = tree->GetEntries();

    tree->SetBranchAddress("theta", theta);
    tree->SetBranchAddress("weights", weights);

    rdr = new plots_reader(inputs);

    lambda_score = new TH1D("lambda_score", "; #frac{#lambda_{true} - #lambda_{meas}}{#lambda_{error}};", 30, -5., 5.);
    mu_score = new TH1D("mu_score", "; #frac{#mu_{true} - #mu_{meas}}{#mu_{error}};", 30, -5., 5.);
    nu_score = new TH1D("nu_score", "; #frac{#nu_{true} - #nu_{meas}}{#nu_{error}};", 30, -5., 5.);

    lambda_error = new TH1D("lambda_error", "; #lambda_{error}; ", 30, 0., 1.);
    mu_error = new TH1D("mu_error", "; #mu_{error}; ", 30, 0., 1.);
    nu_error = new TH1D("nu_error", "; #nu_{error}; ", 30, 0., 1.);

    lambda_graph = new TGraphErrors();
    mu_graph = new TGraphErrors();
    nu_graph = new TGraphErrors();

    lambda_true_score = new TH2D("lambda_true_score", "; #lambda_{true}; #frac{#lambda_{true} - #lambda_{meas}}{#lambda_{error}}", 30, -1., 1., 30, -5., 5.);
    mu_true_score = new TH2D("mu_true_score", "; #mu_{true}; #frac{#mu_{true} - #mu_{meas}}{#mu_{error}}", 30, -0.5, 0.5, 30, -5., 5.);
    nu_true_score = new TH2D("nu_true_score", "; #nu_{true}; #frac{#nu_{true} - #nu_{meas}}{#nu_{error}}", 30, -0.5, 0.5, 30, -5., 5.);

    lambda_true_error = new TH2D("lambda_true_error", "; #lambda_{true}; #lambda_{error}", 30, -1., 1., 30, 0., 1.);
    mu_true_error = new TH2D("mu_true_error", "; #mu_{true}; #mu_{error}", 30, -0.5, 0.5, 30, 0., 1.);
    nu_true_error = new TH2D("nu_true_error", "; #nu_{true}; #nu_{error}", 30, -0.5, 0.5, 30, 0., 1.);

    can = new TCanvas("can", "can", 800, 800);
}


void ratio_plots::fill() {

    for(int ii = 0; ii < n_events; ii++) {
        tree->GetEntry(ii);
        rdr->fill(theta, weights);
        if(ii < 5){rdr->plot(theta, ii);}
        if(ii < 100){rdr->graphs(theta, lambda_graph, mu_graph, nu_graph, ii);}
        rdr->histograms(theta, lambda_score, mu_score, nu_score, lambda_error, mu_error, nu_error, lambda_true_score, mu_true_score, nu_true_score, lambda_true_error, mu_true_error, nu_true_error);
    }
}


void ratio_plots::plot_hist1D(TH1D* hist, TString pname) {

    hist->Draw("HIST");
    can->Update();
    can->SaveAs(pname.Data());
}


void ratio_plots::plot_hist2D(TH2D* hist, TString pname) {

    hist->Draw("COLZ");
    can->Update();
    can->SaveAs(pname.Data());
}


void ratio_plots::plot_graph(TGraphErrors* graph, double xmin, double xmax, TString gname, TString tname, TString pname) {

    graph->SetNameTitle(gname.Data(), tname.Data());
    graph->SetMarkerColor(4);
    graph->SetMarkerStyle(21);

    TLine* ll = new TLine(xmin, xmin, xmax, xmax);
    ll->SetLineColor(2);
    ll->SetLineWidth(2);
    ll->SetLineStyle(2);

    graph->Draw("APE1");
    ll->Draw("SAME");
    can->Update();
    can->SaveAs(pname.Data());

    delete ll;
}

void ratio_plots::plot() {

    plot_hist1D(lambda_score, "./plots/lambda_score.png");
    plot_hist1D(mu_score, "./plots/mu_score.png");
    plot_hist1D(nu_score, "./plots/nu_score.png");

    plot_hist1D(lambda_error,  "./plots/lambda_error.png");
    plot_hist1D(mu_error, "./plots/mu_error.png");
    plot_hist1D(nu_error, "./plots/nu_error.png");

    plot_hist2D(lambda_true_score, "./plots/lambda_true_score.png");
    plot_hist2D(mu_true_score, "./plots/mu_true_score.png");
    plot_hist2D(nu_true_score, "./plots/nu_true_score.png");

    plot_hist2D(lambda_true_error, "./plots/lambda_true_error.png");
    plot_hist2D(mu_true_error, "./plots/mu_true_error.png");
    plot_hist2D(nu_true_error, "./plots/nu_true_error.png");

    plot_graph(lambda_graph, -1., 1., "lambda_graph", "; #lambda_{true}; #lambda_{meas}", "./plots/lambda_graph.png");
    plot_graph(mu_graph, -0.5, 0.5, "mu_graph", "; #mu_{true}; #mu_{meas}", "./plots/mu_graph.png");
    plot_graph(nu_graph, -0.5, 0.5, "nu_graph", "; #nu_{true}; #nu_{meas}", "./plots/nu_graph.png");
}
