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
    Long64_t D1;

    intree->SetBranchAddress("mass", &mass);
    intree->SetBranchAddress("pT", &pT);
    intree->SetBranchAddress("xF", &xF);
    intree->SetBranchAddress("phi", &phi);
    intree->SetBranchAddress("costh", &costh);
    intree->SetBranchAddress("D1", &D1);
    intree->SetBranchAddress("weight", &weight);

    TH2D* hist = new TH2D("hist", "", 12, -pi, pi, 12, -0.4, 0.4);

    for(int ii = 0; ii < num_events; ii++) {
        intree->GetEntry(ii);
        hist->Fill(phi, costh, weight);
    }

    hist->Scale(1./hist->GetMaximum());

    std::cout << hist->GetEffectiveEntries() << std::endl;
    std::cout << hist->GetEntries() << std::endl;

    TFile* outputs = new TFile("./data/RS67_LH2_hist.root", "recreate");
    TTree* out_tree = new TTree("out_tree", "out_tree");

    double X[1][12][12];

    out_tree->Branch("X", X, "X[1][12][12]/D");

    for(int ii = 0; ii < 12; ii++) {
        for(int jj = 0; jj < 12; jj++) {
            X[0][ii][jj] = hist->GetBinContent(ii+1, jj+1);
        }
    }

    out_tree->Fill();

    out_tree->Write();
    outputs->Close();
}
