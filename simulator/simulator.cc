#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TH3D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TString.h>
#include <iostream>

#include "simulator.h"


double cross_section(double lambda, double mu, double nu, double phi, double costh) {
    double weight = 1. + lambda* costh* costh + 2.* mu* costh* sqrt(1. - costh* costh) *cos(phi) + 0.5* nu* (1. - costh* costh)* cos(2.* phi);
    return weight/(1. + costh* costh);
}


simulator::simulator(TString tname) {
    tree = new TTree(tname.Data(), tname.Data());
    
    tree->Branch("X", X, "X[1][10][10]/D");
    tree->Branch("theta", theta, "theta[3]/D");
}

void simulator::samples(TTree* inputs, TRandom3* prior, int events, int ndata) {
    
    double pT, phi, costh, true_phi, true_costh;
    
    inputs->SetBranchAddress("pT", &pT);
    inputs->SetBranchAddress("phi", &phi);
    inputs->SetBranchAddress("costh", &costh);
    inputs->SetBranchAddress("true_phi", &true_phi);
    inputs->SetBranchAddress("true_costh", &true_costh);
    
    double pi = TMath::Pi();
    
    // draw samples from histograms
    for(int ii = 0; ii < events; ii++) {
        theta[0] = prior->Uniform(-1., 1.);
        theta[1] = prior->Uniform(-0.5, 0.5);
        theta[2] = prior->Uniform(-0.5, 0.5);
        
        int fill = 0;
        
        TH2D* hist = new TH2D("hist", "", 10, -pi, pi, 10, -0.5, 0.5);
        
        for(int jj = 0; jj < inputs->GetEntries(); jj++) {
            inputs->GetEntry(jj);
            if(0.5 < prior->Uniform(0., 1.)) {
                hist->Fill(phi, costh, cross_section(theta[0], theta[1], theta[2], true_phi, true_costh));
                fill += 1;
            }
            if(fill == ndata){break;}
        }
        
        hist->Scale(1./hist->Integral());
        
        for(int jj = 0; jj < 10; jj++) {
            for(int kk = 0; kk < 10; kk++) {
                X[0][jj][kk] = hist->GetBinContent(jj+1, kk+1);
            }
        }
        
        tree->Fill();
        delete hist;
        if(ii%10000==0){std::cout << "===> " << ii << " events are done" << std::endl;}
    }
}

// void simulator::samples(TTree* inputs, TRandom3* prior, double lambda, double mu, double nu) {
//
//     double pT, phi, costh, true_phi, true_costh;
//
//     inputs->SetBranchAddress("pT", &pT);
//     inputs->SetBranchAddress("phi", &phi);
//     inputs->SetBranchAddress("costh", &costh);
//     inputs->SetBranchAddress("true_phi", &true_phi);
//     inputs->SetBranchAddress("true_costh", &true_costh);
//
//     double pi = TMath::Pi();
//
//     TH2D* hist = new TH2D("hist", "", 10, -pi, pi, 10, -0.5, 0.5);
//
//     theta[0] = lambda;
//     theta[1] = mu;
//     theta[2] = nu;
//
//     int events = inputs->GetEntries();
//
//     for(int ii = 0; ii < events; ii++) {
//         inputs->GetEntry(ii);
//         hist->Fill(phi, costh, cross_section(lambda, mu, nu, true_phi, true_costh));
//     }
//
//     hist->Scale(1./hist->Integral());
//
//     for(int ii = 0; ii < 10; ii++) {
//         for(int jj = 0; jj < 10; jj++) {
//             X[0][ii][jj] = hist->GetBinContent(ii+1, jj+1);
//         }
//     }
//
//     tree->Fill();
//
//     delete hist;
// }


void forward_simulation(int seed, int events, int ndata) {

    std::cout << "[===> forward simulation]" << std::endl;

    // generator
    TRandom3* prior = new TRandom3(seed);

    // inputs
    TFile* infile = TFile::Open("./data/generator.root", "read");
    TTree* train = (TTree*)infile->Get("train");
    TTree* test = (TTree*)infile->Get("test");

    TFile* outfile = new TFile("./data/outputs.root", "recreate");

    // train events
    simulator* sim1 = new simulator("train_tree");
    sim1->samples(train, prior, events, ndata);
    sim1->save();

    // test events
    simulator* sim2 = new simulator("test_tree");
    sim2->samples(test, prior, 20000, ndata);
    sim2->save();

    outfile->Write();
    outfile->Close();
}
