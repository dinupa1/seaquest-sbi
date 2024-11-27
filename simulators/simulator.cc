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


sim_reader::sim_reader(TFile* infile, TString tname) {
    tree = (TTree*)infile->Get(tname.Data());

    num_events = tree->GetEntries();

    tree->SetBranchAddress("pT", &pT);
    tree->SetBranchAddress("phi", &phi);
    tree->SetBranchAddress("costh", &costh);
    tree->SetBranchAddress("true_pT", &true_pT);
    tree->SetBranchAddress("true_phi", &true_phi);
    tree->SetBranchAddress("true_costh", &true_costh);
}


void sim_reader::fill(double theta[9], std::unique_ptr<TH2D> &hist_0, std::unique_ptr<TH2D> &hist_1, std::unique_ptr<TH2D> &hist_2, std::unique_ptr<TRandom3> &generator) {

    for(int ii = 0; ii < num_data; ii++) {
        int event = generator->Integer(num_events);
        tree->GetEntry(event);
        double weight = 0.;
        for(int jj = 0; jj < 3; jj++) {
            if(pT_edges[jj] < true_pT && true_pT <= pT_edges[jj+1]) {
                weight = cross_section(theta[jj], theta[3 + jj], theta[6 + jj], true_phi, true_costh);
            }
        }
        hist_0->Fill(phi, costh, weight);
        hist_1->Fill(pT, phi, weight);
        hist_2->Fill(pT, costh, weight);
    }
}


simulator::simulator():generator(std::make_unique<TRandom3>(42)) {

    gSystem->Exec("python ./simulators/generator.py");

    TFile* inputs = TFile::Open("./data/generator.root", "read");

    rdr = new sim_reader(inputs, "tree");

    outputs = new TFile("./data/outputs.root", "recreate");

    out_tree = new TTree("out_tree", "out_tree");
    out_tree->Branch("X", X, "X[3][12][12]/D");
    out_tree->Branch("theta", theta, "theta[9]/D");
}


void simulator::read(double X[3][12][12], std::unique_ptr<TH2D> &hist_0, std::unique_ptr<TH2D> &hist_1, std::unique_ptr<TH2D> &hist_2) {

    hist_0->Scale(1./hist_0->GetMaximum());
    hist_1->Scale(1./hist_1->GetMaximum());
    hist_2->Scale(1./hist_2->GetMaximum());

    for(int ii = 0; ii < num_bins; ii++) {
        for(int jj = 0; jj < num_bins; jj++) {
            X[0][ii][jj] = hist_0->GetBinContent(ii+1, jj+1);
            X[1][ii][jj] = hist_1->GetBinContent(ii+1, jj+1);
            X[2][ii][jj] = hist_2->GetBinContent(ii+1, jj+1);
        }
    }
}


void simulator::samples(int num_samples) {

    for(int ii = 0; ii < num_samples; ii++) {

        for(int jj = 0; jj < 3; jj++) {
            theta[jj] = generator->Uniform(-1.5, 1.5);
            theta[3 + jj] = generator->Uniform(-0.6, 0.6);
            theta[6 + jj] = generator->Uniform(-0.6, 0.6);
        }

        std::unique_ptr<TH2D> hist_0(new TH2D("hist_0", "", num_bins, -pi, pi, num_bins, -0.4, 0.4));
        std::unique_ptr<TH2D> hist_1(new TH2D("hist_1", "", num_bins, 0., 2.5, num_bins, -pi, pi));
        std::unique_ptr<TH2D> hist_2(new TH2D("hist_2", "", num_bins, 0., 2.5, num_bins, -0.4, 0.4));

        rdr->fill(theta, hist_0, hist_1, hist_2, generator);
        read(X, hist_0, hist_1, hist_2);
        out_tree->Fill();

        if(ii%1024==0){std::cout << "[ ===> " << ii << " samples are done ]" << std::endl;}
    }
}


void simulator::save() {

    outputs->cd();
    out_tree->Write();
    outputs->Close();
}
