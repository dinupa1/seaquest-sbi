#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TH3D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TString.h>
#include <TSystem.h>
#include <iostream>

#include "simulator.h"


double cross_section(double lambda, double mu, double nu, double phi, double costh) {
    double weight = 1. + lambda* costh* costh + 2.* mu* costh* sqrt(1. - costh* costh) *cos(phi) + 0.5* nu* (1. - costh* costh)* cos(2.* phi);
    return weight/(1. + costh* costh);
}


simulator2D::simulator2D(TString tname) {
    tree = new TTree(tname.Data(), tname.Data());
    
    tree->Branch("X", X, "X[1][10][10]/D");
    tree->Branch("theta", theta, "theta[3]/D");
    tree->Branch("theta0", theta0, "theta0[3]/D");
}

void simulator2D::train_samples(TTree* inputs, TTree* prior, TRandom3* generator) {

    int n_data = 14999;
    
    double pT, phi, costh, true_phi, true_costh;

    inputs->SetBranchAddress("phi", &phi);
    inputs->SetBranchAddress("costh", &costh);
    inputs->SetBranchAddress("true_phi", &true_phi);
    inputs->SetBranchAddress("true_costh", &true_costh);

    prior->SetBranchAddress("theta0", theta0);
    prior->SetBranchAddress("theta", theta);
    
    // draw samples from histograms
    for(int ii = 0; ii < prior->GetEntries(); ii++) {

        prior->GetEntry(ii);

        int fill = 0;
        
        TH2D* hist = new TH2D("hist", "", 10, -pi, pi, 10, -0.4, 0.4);
        
        for(int jj = 0; jj < inputs->GetEntries(); jj++) {
            if(fill > n_data) break;
            if(generator->Uniform(0., 1.) < generator->Uniform(0., 1.)) continue;
            inputs->GetEntry(jj);
            hist->Fill(phi, costh, cross_section(theta[0], theta[1], theta[2], true_phi, true_costh));
            fill++;
        }
        /*
        std::cout << "===> lambda0, mu0, nu0 " << theta0[0] << " , " << theta0[1] << " , " << theta0[2] << std::endl;
        std::cout << "===> lambda1, mu1, nu1 " << theta[0] << " , " << theta[1] << " , " << theta[2] << std::endl;
        std::cout << "===> events filled " << fill << std::endl;
        std::cout << "===> final event " << jj_last << std::endl;
        */
        
        hist->Scale(1./hist->Integral());
        
        for(int jj = 0; jj < 10; jj++) {
            for(int kk = 0; kk < 10; kk++) {
                X[0][jj][kk] = hist->GetBinContent(jj+1, kk+1);
            }
        }
        
        tree->Fill();
        delete hist;
        if(ii%10000==0){std::cout << "[ ===> " << ii << " events are done ]" << std::endl;}
    }
}


void simulator2D::test_samples(TTree* inputs, TTree* prior, TRandom3* generator) {

    int n_data = 14999;

    double pT, phi, costh, true_phi, true_costh;

    inputs->SetBranchAddress("phi", &phi);
    inputs->SetBranchAddress("costh", &costh);
    inputs->SetBranchAddress("true_phi", &true_phi);
    inputs->SetBranchAddress("true_costh", &true_costh);

    prior->SetBranchAddress("theta0", theta0);
    prior->SetBranchAddress("theta", theta);

    int events = 0;

    // draw samples from histograms
    for(int ii = 0; ii < prior->GetEntries(); ii++) {

        if(events > n_data) break;

        prior->GetEntry(ii);

        if(std::abs(theta[0]) > 1.) continue;
        if(std::abs(theta[1]) > 0.5) continue;
        if(std::abs(theta[2]) > 0.5) continue;

        int fill = 0;

        TH2D* hist = new TH2D("hist", "", 10, -pi, pi, 10, -0.4, 0.4);

        for(int jj = 0; jj < inputs->GetEntries(); jj++) {

            if(fill > n_data) break;

            if(generator->Uniform(0., 1.) < generator->Uniform(0., 1.)) continue;

            inputs->GetEntry(jj);
            hist->Fill(phi, costh, cross_section(theta[0], theta[1], theta[2], true_phi, true_costh));
            fill++;
        }
        /*
        std::cout << "===> lambda0, mu0, nu0 " << theta0[0] << " , " << theta0[1] << " , " << theta0[2] << std::endl;
        std::cout << "===> lambda1, mu1, nu1 " << theta[0] << " , " << theta[1] << " , " << theta[2] << std::endl;
        std::cout << "===> events filled " << fill << std::endl;
        std::cout << "===> final event " << jj_last << std::endl;
        */

        hist->Scale(1./hist->Integral());

        for(int jj = 0; jj < 10; jj++) {
            for(int kk = 0; kk < 10; kk++) {
                X[0][jj][kk] = hist->GetBinContent(jj+1, kk+1);
            }
        }

        events++;

        tree->Fill();
        delete hist;
        if(ii%10000==0){std::cout << "[ ===> " << ii << " events are done ]" << std::endl;}
    }
}


simulator3D::simulator3D(TString tname) {
    tree = new TTree(tname.Data(), tname.Data());

    tree->Branch("Xs", Xs, "Xs[4][10][10]/D");
    tree->Branch("thetas", thetas, "thetas[12]/D");
    tree->Branch("thetas0", thetas0, "thetas0[12]/D");

    double pi = TMath::Pi();

    for(int ii = 0; ii < 11; ii++) {
        phi_edges[ii] = -pi + (2. * pi/10.) * ii ;
        costh_edges[ii] = -0.5 + (1./10.) * ii;
    }
}


void simulator3D::samples(TTree* inputs, TRandom3* generator, int events, int ndata) {

    double pT, phi, costh, true_pT, true_phi, true_costh;

    inputs->SetBranchAddress("pT", &pT);
    inputs->SetBranchAddress("phi", &phi);
    inputs->SetBranchAddress("costh", &costh);
    inputs->SetBranchAddress("true_pT", &true_pT);
    inputs->SetBranchAddress("true_phi", &true_phi);
    inputs->SetBranchAddress("true_costh", &true_costh);

    // draw samples from histograms
    for(int ii = 0; ii < events; ii++) {

        // lambda, mu, nu
        for(int jj = 0; jj < 4; jj++) {
            thetas[0+jj] = generator->Uniform(-1., 1.);
            thetas[4+jj] = generator->Uniform(-0.5, 0.5);
            thetas[8+jj] = generator->Uniform(-0.5, 0.5);
        }

        int fill = 0;

        TH3D* hist = new TH3D("hist", "", 4, pT_edges, 10, phi_edges, 10, costh_edges);

        for(int jj = 0; jj < inputs->GetEntries(); jj++) {
            if(fill >= ndata) break;
            if(0.25 < generator->Uniform(0., 1.)) continue;
            inputs->GetEntry(jj);
            for(int kk = 0; kk < 4; kk++) {
                if(pT_edges[kk] < true_pT && true_pT <= pT_edges[kk+1]) {
                    hist->Fill(pT, phi, costh, cross_section(thetas[0+kk], thetas[4+kk], thetas[8+kk], true_phi, true_costh));
                    fill ++;
                    break;
                }
            }
        }

        hist->Scale(1./hist->Integral());

        for(int jj = 0; jj < 4; jj++) {
            for(int kk = 0; kk < 10; kk++) {
                for(int mm = 0; mm < 10; mm++) {
                     Xs[jj][kk][mm] = hist->GetBinContent(jj+1, kk+1, mm+1);
                }
            }
        }

        for(int jj = 0; jj < 4; jj++) {
            thetas0[0+jj] = generator->Uniform(-1., 1.);
            thetas0[4+jj] = generator->Uniform(-0.5, 0.5);
            thetas0[8+jj] = generator->Uniform(-0.5, 0.5);
        }

        tree->Fill();
        delete hist;
        if(ii%10000==0){std::cout << "[ ===> " << ii << " events are done ]" << std::endl;}
    }
}


void forward_simulation(int seed) {

    std::cout << "[ ===> forward simulation ]" << std::endl;

    gSystem->Exec("python ./simulators/generator.py");

    // generator
    TRandom3* generator = new TRandom3(seed);

    // inputs
    TFile* infile = TFile::Open("./data/generator.root", "read");

    TTree* X_train = (TTree*)infile->Get("X_train");
    TTree* X_val = (TTree*)infile->Get("X_val");
    TTree* X_test = (TTree*)infile->Get("X_test");

    TTree* theta_train = (TTree*)infile->Get("theta_train");
    TTree* theta_val = (TTree*)infile->Get("theta_val");
    TTree* theta_test = (TTree*)infile->Get("theta_test");

    TFile* outfile = new TFile("./data/outputs.root", "recreate");

    // train events
    simulator2D* sim1 = new simulator2D("train_tree");
    sim1->train_samples(X_train, theta_train, generator);
    sim1->save();

    simulator2D* sim2 = new simulator2D("val_tree");
    sim2->train_samples(X_val, theta_val, generator);
    sim2->save();

    // test events
    simulator2D* sim3 = new simulator2D("test_tree");
    sim3->test_samples(X_test, theta_test, generator);
    sim3->save();

    outfile->Write();
    outfile->Close();
}
