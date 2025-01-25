#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TMath.h>
#include <iostream>


void phi_costheta() {

    double pi = TMath::Pi();

    TFile* infile = TFile::Open("./data/RS67_LH2_data.root", "read");
    TTree* intree = (TTree*)infile->Get("tree");
    int num_events = intree->GetEntries();

    float mass;
    float pT;
    float xF;
    float phi;
    float costh;
    double weight;

    intree->SetBranchAddress("mass", &mass);
    intree->SetBranchAddress("pT", &pT);
    intree->SetBranchAddress("xF", &xF);
    intree->SetBranchAddress("phi", &phi);
    intree->SetBranchAddress("costh", &costh);
    intree->SetBranchAddress("weight", &weight);

    TFile* outputs = new TFile("./data/RS67_LH2_hist.root", "recreate");

    TH2D* hist = new TH2D("hist", "", 12, -pi, pi, 12, -0.4, 0.4);

    for(int ii = 0; ii < num_events; ii++) {
        intree->GetEntry(ii);
        hist->Fill(phi, costh, weight);
    }

    hist->Scale(1./hist->GetMaximum());

    hist->Write();
    outputs->Close();
}
