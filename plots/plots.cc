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

plots::plots() {

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


void plots::fill(TTree* tree, TTree* priors) {

    double theta_test[3];
    double weights[15000];
    double theta[3];

    tree->SetBranchAddress("theta", theta);
    tree->SetBranchAddress("weights", weights);

    priors->SetBranchAddress("theta_test", theta_test);

    for(int ii = 0; ii < tree->GetEntries(); ii++) {
        tree->GetEntry(ii);

        TH1D* hist_lambda = new TH1D("hist_lambda", "; #lambda; p(x | #lambda)", 30, -1.5, 1.5);
        TH1D* hist_mu = new TH1D("hist_mu", "; #mu; p(x | #mu)", 30, -0.6, 0.6);
        TH1D* hist_nu = new TH1D("hist_nu", "; #nu; p(x | #nu)", 30, -0.6, 0.6);

        for(int jj = 0; jj < priors->GetEntries(); jj++) {
            priors->GetEntry(jj);
            hist_lambda->Fill(theta_test[0], weights[jj]);
            hist_mu->Fill(theta_test[1], weights[jj]);
            hist_nu->Fill(theta_test[2], weights[jj]);
        }

        hist_lambda->Scale(1./hist_lambda->Integral());
        hist_mu->Scale(1./hist_mu->Integral());
        hist_nu->Scale(1./hist_nu->Integral());

        meas[0] = hist_lambda->GetMean();
        meas[1] = hist_mu->GetMean();
        meas[2] = hist_nu->GetMean();

        errors[0] = hist_lambda->GetStdDev();
        errors[1] = hist_mu->GetStdDev();
        errors[2] = hist_nu->GetStdDev();

        score[0] = (theta[0] - meas[0])/errors[0];
        score[1] = (theta[1] - meas[1])/errors[1];
        score[2] = (theta[2] - meas[2])/errors[2];

        lambda_score->Fill(score[0]);
        mu_score->Fill(score[1]);
        nu_score->Fill(score[2]);

        lambda_error->Fill(errors[0]);
        mu_error->Fill(errors[1]);
        nu_error->Fill(errors[2]);

        lambda_true_score->Fill(theta[0], score[0]);
        mu_true_score->Fill(theta[1], score[1]);
        nu_true_score->Fill(theta[2], score[2]);

        lambda_true_error->Fill(theta[0], errors[0]);
        mu_true_error->Fill(theta[1], errors[1]);
        nu_true_error->Fill(theta[2], errors[2]);

        if(ii < 100) {
            lambda_graph->SetPoint(ii, theta[0], meas[0]);
            lambda_graph->SetPointError(ii, 0., errors[0]);

            mu_graph->SetPoint(ii, theta[1], meas[1]);
            mu_graph->SetPointError(ii, 0., errors[1]);

            nu_graph->SetPoint(ii, theta[2], meas[2]);
            nu_graph->SetPointError(ii, 0., errors[2]);
        }

        if(ii < 5) {

            hist_lambda->Draw("HIST");

            TLine* ll1 = new TLine(theta[0], 0., theta[0], hist_lambda->GetMaximum());
            ll1->SetLineColor(2);
            ll1->SetLineWidth(2);
            ll1->SetLineStyle(2);

            ll1->Draw("SAME");

            TLine* ll2 = new TLine(hist_lambda->GetMean() - hist_lambda->GetStdDev(), 0., hist_lambda->GetMean() - hist_lambda->GetStdDev(), hist_lambda->GetMaximum());
            ll2->SetLineColor(4);
            ll2->SetLineWidth(2);
            ll2->SetLineStyle(1);

            ll2->Draw("SAME");

            TLine* ll3 = new TLine(hist_lambda->GetMean() + hist_lambda->GetStdDev(), 0., hist_lambda->GetMean() + hist_lambda->GetStdDev(), hist_lambda->GetMaximum());
            ll3->SetLineColor(4);
            ll3->SetLineWidth(2);
            ll3->SetLineStyle(1);

            ll3->Draw("SAME");

            TString lname = Form("./plots/lambda_%d.png", ii);
            can->Update();
            can->SaveAs(lname.Data());

            hist_mu->Draw("HIST");

            TLine* ml1 = new TLine(theta[1], 0., theta[1], hist_mu->GetMaximum());
            ml1->SetLineColor(2);
            ml1->SetLineWidth(2);
            ml1->SetLineStyle(2);

            ml1->Draw("SAME");

            TLine* ml2 = new TLine(hist_mu->GetMean() - hist_mu->GetStdDev(), 0., hist_mu->GetMean() - hist_mu->GetStdDev(), hist_mu->GetMaximum());
            ml2->SetLineColor(4);
            ml2->SetLineWidth(2);
            ml2->SetLineStyle(1);

            ml2->Draw("SAME");

            TLine* ml3 = new TLine(hist_mu->GetMean() + hist_mu->GetStdDev(), 0., hist_mu->GetMean() + hist_mu->GetStdDev(), hist_mu->GetMaximum());
            ml3->SetLineColor(4);
            ml3->SetLineWidth(2);
            ml3->SetLineStyle(1);

            ml3->Draw("SAME");

            TString mname = Form("./plots/mu_%d.png", ii);
            can->Update();
            can->SaveAs(mname.Data());

            hist_nu->Draw("HIST");

            TLine* nl1 = new TLine(theta[2], 0., theta[2], hist_nu->GetMaximum());
            nl1->SetLineColor(2);
            nl1->SetLineWidth(2);
            nl1->SetLineStyle(2);

            nl1->Draw("SAME");

            TLine* nl2 = new TLine(hist_nu->GetMean() - hist_nu->GetStdDev(), 0., hist_nu->GetMean() - hist_nu->GetStdDev(), hist_nu->GetMaximum());
            nl2->SetLineColor(4);
            nl2->SetLineWidth(2);
            nl2->SetLineStyle(1);

            nl2->Draw("SAME");

            TLine* nl3 = new TLine(hist_nu->GetMean() + hist_nu->GetStdDev(), 0., hist_nu->GetMean() + hist_nu->GetStdDev(), hist_nu->GetMaximum());
            nl3->SetLineColor(4);
            nl3->SetLineWidth(2);
            nl3->SetLineStyle(1);

            nl3->Draw("SAME");

            TString nname = Form("./plots/nu_%d.png", ii);
            can->Update();
            can->SaveAs(nname.Data());

            delete ll1;
            delete ll2;
            delete ll3;
            delete ml1;
            delete ml2;
            delete ml3;
            delete nl1;
            delete nl2;
            delete nl3;
        }

        delete hist_lambda;
        delete hist_mu;
        delete hist_nu;
    }
}


void plots::plot() {

    lambda_score->Draw("HIST");
    can->Update();
    can->SaveAs("./plots/lambda_score.png");

    mu_score->Draw("HIST");
    can->Update();
    can->SaveAs("./plots/mu_score.png");

    nu_score->Draw("HIST");
    can->Update();
    can->SaveAs("./plots/nu_score.png");

    lambda_error->Draw("HIST");
    can->Update();
    can->SaveAs("./plots/lambda_error.png");

    mu_error->Draw("HIST");
    can->Update();
    can->SaveAs("./plots/mu_error.png");

    nu_error->Draw("HIST");
    can->Update();
    can->SaveAs("./plots/nu_error.png");

    lambda_true_score->Draw("COLZ");
    can->Update();
    can->SaveAs("./plots/lambda_true_score.png");

    mu_true_score->Draw("COLZ");
    can->Update();
    can->SaveAs("./plots/mu_true_score.png");

    nu_true_score->Draw("COLZ");
    can->Update();
    can->SaveAs("./plots/nu_true_score.png");

    lambda_true_error->Draw("COLZ");
    can->Update();
    can->SaveAs("./plots/lambda_true_error.png");

    mu_true_error->Draw("COLZ");
    can->Update();
    can->SaveAs("./plots/mu_true_error.png");

    nu_true_error->Draw("COLZ");
    can->Update();
    can->SaveAs("./plots/nu_true_error.png");

    lambda_graph->SetNameTitle("lambda_graph", "; #lambda_{true}; #lambda_{meas}");
    lambda_graph->SetMarkerColor(4);
    lambda_graph->SetMarkerStyle(21);

    TLine* l1 = new TLine(-1., -1., 1., 1.);
    l1->SetLineColor(2);
    l1->SetLineWidth(2);
    l1->SetLineStyle(2);

    lambda_graph->Draw("APE1");
    l1->Draw("SAME");
    can->Update();
    can->SaveAs("./plots/lambda_graph.png");

    mu_graph->SetNameTitle("mu_graph", "; #mu_{true}; #mu_{meas}");
    mu_graph->SetMarkerColor(4);
    mu_graph->SetMarkerStyle(21);

    TLine* l2 = new TLine(-0.5, -0.5, 0.5, 0.5);
    l2->SetLineColor(2);
    l2->SetLineWidth(2);
    l2->SetLineStyle(2);

    mu_graph->Draw("APE1");
    l2->Draw("SAME");
    can->Update();
    can->SaveAs("./plots/mu_graph.png");

    nu_graph->SetNameTitle("nu_graph", "; #nu_{true}; #nu_{meas}");
    nu_graph->SetMarkerColor(4);
    nu_graph->SetMarkerStyle(21);

    TLine* l3 = new TLine(-0.5, -0.5, 0.5, 0.5);
    l3->SetLineColor(2);
    l3->SetLineWidth(2);
    l3->SetLineStyle(2);

    nu_graph->Draw("APE1");
    l3->Draw("SAME");
    can->Update();
    can->SaveAs("./plots/nu_graph.png");

    delete l1;
    delete l2;
    delete l3;
}
