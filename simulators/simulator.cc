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


simulator2D::simulator2D(TString tname) {
    tree = new TTree(tname.Data(), tname.Data());
    
    tree->Branch("Xs", Xs, "Xs[1][10][10]/D");
    tree->Branch("thetas", thetas, "thetas[3]/D");
    tree->Branch("thetas0", thetas0, "thetas0[3]/D");
}

void simulator2D::samples(TTree* inputs, TRandom3* generator, int events, int ndata) {
    
    double pT, phi, costh, true_phi, true_costh;

    inputs->SetBranchAddress("phi", &phi);
    inputs->SetBranchAddress("costh", &costh);
    inputs->SetBranchAddress("true_phi", &true_phi);
    inputs->SetBranchAddress("true_costh", &true_costh);
    
    double pi = TMath::Pi();
    
    // draw samples from histograms
    for(int ii = 0; ii < events; ii++) {
        thetas[0] = generator->Uniform(-1., 1.);
        thetas[1] = generator->Uniform(-0.5, 0.5);
        thetas[2] = generator->Uniform(-0.5, 0.5);
        
        int fill = 0;
        
        TH2D* hist = new TH2D("hist", "", 10, -pi, pi, 10, -0.5, 0.5);
        
        for(int jj = 0; jj < inputs->GetEntries(); jj++) {
            if(fill >= ndata) break;
            if(0.25 < generator->Uniform(0., 1.)) continue;
            inputs->GetEntry(jj);
            hist->Fill(phi, costh, cross_section(thetas[0], thetas[1], thetas[2], true_phi, true_costh));
            fill ++;
        }
        
        hist->Scale(1./hist->Integral());
        
        for(int jj = 0; jj < 10; jj++) {
            for(int kk = 0; kk < 10; kk++) {
                Xs[0][jj][kk] = hist->GetBinContent(jj+1, kk+1);
            }
        }

        thetas0[0] = generator->Uniform(-1., 1.);
        thetas0[1] = generator->Uniform(-0.5, 0.5);
        thetas0[2] = generator->Uniform(-0.5, 0.5);
        
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

/*
void forward_simulation(int seed, int train_size, int ndata, int test_size) {

    std::cout << "[ ===> forward simulation ]" << std::endl;

    // generator
    TRandom3* prior = new TRandom3(seed);

    // inputs
    TFile* infile = TFile::Open("./data/generator.root", "read");
    TTree* train = (TTree*)infile->Get("train");
    TTree* test = (TTree*)infile->Get("test");

    TFile* outfile = new TFile("./data/outputs.root", "recreate");

    // train events
    simulator3D* sim1 = new simulator3D("train_tree");
    sim1->samples(train, prior, train_size, ndata);
    sim1->save();

    // test events
    simulator3D* sim2 = new simulator3D("test_tree");
    sim2->samples(test, prior, test_size, ndata);
    sim2->save();

    outfile->Write();
    outfile->Close();
}
*/
