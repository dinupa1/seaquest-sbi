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

#include "plots3D.h"

plots3D::plots3D() {

    for(int ii = 0; ii < 4; ii++) {

        TString lambda_score_name = Form("lambda_score_%d", ii);
        TString mu_score_name = Form("mu_score_%d", ii);
        TString nu_score_name = Form("nu_score_%d", ii);

        lambda_score[ii] = new TH1D(lambda_score_name.Data(), "; #frac{#lambda_{true} - #lambda_{meas}}{#lambda_{error}};", 30, -5., 5.);
        mu_score[ii] = new TH1D(mu_score_name.Data(), "; #frac{#mu_{true} - #mu_{meas}}{#mu_{error}};", 30, -5., 5.);
        nu_score[ii] = new TH1D(nu_score_name.Data(), "; #frac{#nu_{true} - #nu_{meas}}{#nu_{error}};", 30, -5., 5.);

        TString lambda_error_name = Form("lambda_error_%d", ii);
        TString mu_error_name = Form("mu_error_%d", ii);
        TString nu_error_name = Form("nu_error_%d", ii);

        lambda_error[ii] = new TH1D(lambda_error_name.Data(), "; #lambda_{error}; ", 30, 0., 1.);
        mu_error[ii] = new TH1D(mu_error_name.Data(), "; #mu_{error}; ", 30, 0., 1.);
        nu_error[ii] = new TH1D(nu_error_name.Data(), "; #nu_{error}; ", 30, 0., 1.);

        lambda_graph[ii] = new TGraphErrors();
        mu_graph[ii] = new TGraphErrors();
        nu_graph[ii] = new TGraphErrors();

        TString lambda_true_name = Form("lambda_true_score_%d", ii);
        TString mu_true_name = Form("mu_true_score_%d", ii);
        TString nu_true_name = Form("nu_true_score_%d", ii);

        lambda_true_score[ii] = new TH2D(lambda_true_name.Data(), "; #lambda_{true}; #frac{#lambda_{true} - #lambda_{meas}}{#lambda_{error}}", 30, -1., 1., 30, -5., 5.);
        mu_true_score[ii] = new TH2D(mu_true_name.Data(), "; #mu_{true}; #frac{#mu_{true} - #mu_{meas}}{#mu_{error}}", 30, -0.5, 0.5, 30, -5., 5.);
        nu_true_score[ii] = new TH2D(nu_true_name.Data(), "; #nu_{true}; #frac{#nu_{true} - #nu_{meas}}{#nu_{error}}", 30, -0.5, 0.5, 30, -5., 5.);

        TString lambda_true_error_name = Form("lambda_true_error_%d", ii);
        TString mu_true_error_name = Form("mu_true_error_%d", ii);
        TString nu_true_error_name = Form("nu_true_error_%d", ii);

        lambda_true_error[ii] = new TH2D(lambda_true_error_name.Data(), "; #lambda_{true}; #lambda_{error}", 30, -1., 1., 30, 0., 1.);
        mu_true_error[ii] = new TH2D(mu_true_error_name.Data(), "; #mu_{true}; #mu_{error}", 30, -0.5, 0.5, 30, 0., 1.);
        nu_true_error[ii] = new TH2D(nu_true_error_name.Data(), "; #nu_{true}; #nu_{error}", 30, -0.5, 0.5, 30, 0., 1.);
    }

    can = new TCanvas("can", "can", 800, 800);
}


void plots3D::fill(TTree* tree, TTree* prior) {

    double theta[12];
    double weights[15000];
    double theta_test[12];

    tree->SetBranchAddress("theta", theta);
    tree->SetBranchAddress("weights", weights);

    prior->SetBranchAddress("theta_test", theta_test);


    for(int mm = 0; mm < 4; mm++) {

        for(int ii = 0; ii < tree->GetEntries(); ii++) {

            tree->GetEntry(ii);

            TH1D* hist_lambda = new TH1D("hist_lambda", "; #lambda; p(x | #lambda)", 30, -1.5, 1.5);
            TH1D* hist_mu = new TH1D("hist_mu", "; #mu; p(x | #mu)", 30, -0.6, 0.6);
            TH1D* hist_nu = new TH1D("hist_nu", "; #nu; p(x | #nu)", 30, -0.6, 0.6);

            for(int jj = 0; jj < prior->GetEntries(); jj++) {
                prior->GetEntry(jj);
                hist_lambda->Fill(theta_test[mm + 0], weights[jj]);
                hist_mu->Fill(theta_test[mm + 1], weights[jj]);
                hist_nu->Fill(theta_test[mm + 2], weights[jj]);
            }

            hist_lambda->Scale(1./hist_lambda->Integral());
            hist_mu->Scale(1./hist_mu->Integral());
            hist_nu->Scale(1./hist_nu->Integral());

            meas[mm + 0] = hist_lambda->GetMean();
            meas[mm + 1] = hist_mu->GetMean();
            meas[mm + 2] = hist_nu->GetMean();

            errors[mm + 0] = 2. * hist_lambda->GetStdDev();
            errors[mm + 1] = 2. * hist_mu->GetStdDev();
            errors[mm + 2] = 2. * hist_nu->GetStdDev();

            score[mm + 0] = (theta[mm + 0] - meas[mm + 0])/errors[mm + 0];
            score[mm + 1] = (theta[mm + 1] - meas[mm + 1])/errors[mm + 1];
            score[mm + 2] = (theta[mm + 2] - meas[mm + 2])/errors[mm + 2];

            lambda_score[mm]->Fill(score[mm + 0]);
            mu_score[mm]->Fill(score[mm + 1]);
            nu_score[mm]->Fill(score[mm + 2]);

            lambda_error[mm]->Fill(errors[mm + 0]);
            mu_error[mm]->Fill(errors[mm + 1]);
            nu_error[mm]->Fill(errors[mm + 2]);

            lambda_true_score[mm]->Fill(theta[mm + 0], score[mm + 0]);
            mu_true_score[mm]->Fill(theta[mm + 1], score[mm + 1]);
            nu_true_score[mm]->Fill(theta[mm + 2], score[mm + 2]);

            lambda_true_error[mm]->Fill(theta[mm + 0], errors[mm + 0]);
            mu_true_error[mm]->Fill(theta[mm + 1], errors[mm + 1]);
            nu_true_error[mm]->Fill(theta[mm + 2], errors[mm + 2]);

            if(ii < 100) {
                lambda_graph[mm]->SetPoint(ii, theta[mm + 0], meas[mm + 0]);
                lambda_graph[mm]->SetPointError(ii, 0., errors[mm + 0]);

                mu_graph[mm]->SetPoint(ii, theta[mm + 1], meas[mm + 1]);
                mu_graph[mm]->SetPointError(ii, 0., errors[mm + 1]);

                nu_graph[mm]->SetPoint(ii, theta[mm + 2], meas[mm + 2]);
                nu_graph[mm]->SetPointError(ii, 0., errors[mm + 2]);
            }

            if(ii < 5) {

                hist_lambda->Draw("HIST");

                TLine* ll1 = new TLine(theta[mm + 0], 0., theta[mm + 0], hist_lambda->GetMaximum());
                ll1->SetLineColor(2);
                ll1->SetLineWidth(2);
                ll1->SetLineStyle(2);

                ll1->Draw("SAME");

                TLine* ll2 = new TLine(hist_lambda->GetMean() - 2. * hist_lambda->GetStdDev(), 0., hist_lambda->GetMean() - 2. * hist_lambda->GetStdDev(), hist_lambda->GetMaximum());
                ll2->SetLineColor(4);
                ll2->SetLineWidth(2);
                ll2->SetLineStyle(1);

                ll2->Draw("SAME");

                TLine* ll3 = new TLine(hist_lambda->GetMean() + 2. * hist_lambda->GetStdDev(), 0., hist_lambda->GetMean() + 2. * hist_lambda->GetStdDev(), hist_lambda->GetMaximum());
                ll3->SetLineColor(4);
                ll3->SetLineWidth(2);
                ll3->SetLineStyle(1);

                ll3->Draw("SAME");

                TString lname = Form("./plots/lambda_%d_%d.png", mm, ii);
                can->Update();
                can->SaveAs(lname.Data());

                hist_mu->Draw("HIST");

                TLine* ml1 = new TLine(theta[mm + 1], 0., theta[mm + 1], hist_mu->GetMaximum());
                ml1->SetLineColor(2);
                ml1->SetLineWidth(2);
                ml1->SetLineStyle(2);

                ml1->Draw("SAME");

                TLine* ml2 = new TLine(hist_mu->GetMean() - 2. * hist_mu->GetStdDev(), 0., hist_mu->GetMean() - 2. * hist_mu->GetStdDev(), hist_mu->GetMaximum());
                ml2->SetLineColor(4);
                ml2->SetLineWidth(2);
                ml2->SetLineStyle(1);

                ml2->Draw("SAME");

                TLine* ml3 = new TLine(hist_mu->GetMean() + 2. * hist_mu->GetStdDev(), 0., hist_mu->GetMean() + 2. * hist_mu->GetStdDev(), hist_mu->GetMaximum());
                ml3->SetLineColor(4);
                ml3->SetLineWidth(2);
                ml3->SetLineStyle(1);

                ml3->Draw("SAME");

                TString mname = Form("./plots/mu_%d_%d.png", mm, ii);
                can->Update();
                can->SaveAs(mname.Data());

                hist_nu->Draw("HIST");

                TLine* nl1 = new TLine(theta[mm + 2], 0., theta[mm + 2], hist_nu->GetMaximum());
                nl1->SetLineColor(2);
                nl1->SetLineWidth(2);
                nl1->SetLineStyle(2);

                nl1->Draw("SAME");

                TLine* nl2 = new TLine(hist_nu->GetMean() - 2. * hist_nu->GetStdDev(), 0., hist_nu->GetMean() - 2. * hist_nu->GetStdDev(), hist_nu->GetMaximum());
                nl2->SetLineColor(4);
                nl2->SetLineWidth(2);
                nl2->SetLineStyle(1);

                nl2->Draw("SAME");

                TLine* nl3 = new TLine(hist_nu->GetMean() + 2. * hist_nu->GetStdDev(), 0., hist_nu->GetMean() + 2. * hist_nu->GetStdDev(), hist_nu->GetMaximum());
                nl3->SetLineColor(4);
                nl3->SetLineWidth(2);
                nl3->SetLineStyle(1);

                nl3->Draw("SAME");

                TString nname = Form("./plots/nu_%d_%d.png", mm, ii);
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
}


void plots3D::plots() {

    for(int ii = 0; ii < 4; ii++) {

        TString n_lambda_score = Form("./plots/lambda_score_%d.png", ii);
        lambda_score[ii]->Draw("HIST");
        can->Update();
        can->SaveAs(n_lambda_score.Data());

        TString n_mu_score = Form("./plots/mu_score_%d.png", ii);
        mu_score[ii]->Draw("HIST");
        can->Update();
        can->SaveAs(n_mu_score.Data());

        TString n_nu_score = Form("./plots/nu_score_%d.png", ii);
        nu_score[ii]->Draw("HIST");
        can->Update();
        can->SaveAs(n_nu_score.Data());

        TString n_lambda_error = Form("./plots/lambda_error_%d.png", ii);
        lambda_error[ii]->Draw("HIST");
        can->Update();
        can->SaveAs(n_lambda_error.Data());

        TString n_mu_error = Form("./plots/mu_error_%d.png", ii);
        mu_error[ii]->Draw("HIST");
        can->Update();
        can->SaveAs(n_mu_error.Data());

        TString n_nu_error = Form("./plots/nu_error_%d.png", ii);
        nu_error[ii]->Draw("HIST");
        can->Update();
        can->SaveAs(n_nu_error.Data());

        TString n_lambda_true_score = Form("./plots/lambda_true_score_%d.png", ii);
        lambda_true_score[ii]->Draw("COLZ");
        can->Update();
        can->SaveAs(n_lambda_true_score.Data());

        TString n_mu_true_score = Form("./plots/mu_true_score_%d.png", ii);
        mu_true_score[ii]->Draw("COLZ");
        can->Update();
        can->SaveAs(n_mu_true_score.Data());

        TString n_nu_true_score = Form("./plots/nu_true_score_%d.png", ii);
        nu_true_score[ii]->Draw("COLZ");
        can->Update();
        can->SaveAs(n_nu_true_score.Data());

        TString n_lambda_true_error = Form("./plots/lambda_true_error_%d.png", ii);
        lambda_true_error[ii]->Draw("COLZ");
        can->Update();
        can->SaveAs(n_lambda_true_error.Data());

        TString n_mu_true_error = Form("./plots/mu_true_error_%d.png", ii);
        mu_true_error[ii]->Draw("COLZ");
        can->Update();
        can->SaveAs(n_mu_true_error.Data());

        TString n_nu_true_error = Form("./plots/nu_true_error_%d.png", ii);
        nu_true_error[ii]->Draw("COLZ");
        can->Update();
        can->SaveAs(n_nu_true_error.Data());


        TString t_lambda_graph = Form("lambda_graph_%d", ii);
        lambda_graph[ii]->SetNameTitle(t_lambda_graph.Data(), "; #lambda_{true}; #lambda_{meas}");
        lambda_graph[ii]->SetMarkerColor(4);
        lambda_graph[ii]->SetMarkerStyle(21);

        TLine* l1 = new TLine(-1., -1., 1., 1.);
        l1->SetLineColor(2);
        l1->SetLineWidth(2);
        l1->SetLineStyle(2);

        TString n_lambda_graph = Form("./plots/lambda_graph_%d.png", ii);
        lambda_graph[ii]->Draw("APE1");
        l1->Draw("SAME");
        can->Update();
        can->SaveAs(n_lambda_graph.Data());

        TString t_mu_graph = Form("mu_graph_%d", ii);
        mu_graph[ii]->SetNameTitle(t_mu_graph.Data(), "; #mu_{true}; #mu_{meas}");
        mu_graph[ii]->SetMarkerColor(4);
        mu_graph[ii]->SetMarkerStyle(21);

        TLine* l2 = new TLine(-0.5, -0.5, 0.5, 0.5);
        l2->SetLineColor(2);
        l2->SetLineWidth(2);
        l2->SetLineStyle(2);

        TString n_mu_graph = Form("./plots/mu_graph_%d.png", ii);
        mu_graph[ii]->Draw("APE1");
        l2->Draw("SAME");
        can->Update();
        can->SaveAs(n_mu_graph.Data());

        TString t_nu_graph = Form("mu_graph_%d", ii);
        nu_graph[ii]->SetNameTitle(t_nu_graph, "; #nu_{true}; #nu_{meas}");
        nu_graph[ii]->SetMarkerColor(4);
        nu_graph[ii]->SetMarkerStyle(21);

        TLine* l3 = new TLine(-0.5, -0.5, 0.5, 0.5);
        l3->SetLineColor(2);
        l3->SetLineWidth(2);
        l3->SetLineStyle(2);

        TString n_nu_graph = Form("./plots/nu_graph_%d.png", ii);
        nu_graph[ii]->Draw("APE1");
        l3->Draw("SAME");
        can->Update();
        can->SaveAs(n_nu_graph.Data());

        delete l1;
        delete l2;
        delete l3;
    }
}
