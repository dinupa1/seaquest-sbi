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


void sim_reader::fill(double theta[9], std::unique_ptr<TH3D> &hist, std::unique_ptr<TRandom3> &generator) {

    for(int ii = 0; ii < num_data; ii++) {
        int event = generator->Integer(num_events);
        tree->GetEntry(event);
        for(int jj = 0; jj < 3; jj++) {
            if(pT_edges[jj] < true_pT && true_pT <= pT_edges[jj+1]) {
                hist->Fill(pT, phi, costh, cross_section(theta[3 * jj + 0], theta[3 * jj + 1], theta[3 * jj + 2], true_phi, true_costh));
            }
        }
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


void simulator::read(double X[3][12][12], std::unique_ptr<TH3D> &hist) {

    hist->Scale(1./hist->GetMaximum());

    for(int ii = 0; ii < 12; ii++) {
        for(int jj = 0; jj < 12; jj++) {
            X[0][ii][jj] = hist->GetBinContent(1, ii+1, jj+1);
            X[1][ii][jj] = hist->GetBinContent(2, ii+1, jj+1);
            X[2][ii][jj] = hist->GetBinContent(3, ii+1, jj+1);
        }
    }
}


void simulator::samples(int num_samples) {

    for(int ii = 0; ii < num_samples; ii++) {

        for(int jj = 0; jj < 3; jj++) {
            theta[3 * jj + 0] = generator->Uniform(-2., 2.);
            theta[3 * jj + 1] = generator->Uniform(-0.8, 0.8);
            theta[3 * jj + 2] = generator->Uniform(-0.8, 0.8);
        }

        std::unique_ptr<TH3D> hist(new TH3D("hist", "", 3, 0., 2.5, 12, -pi, pi, 12, -0.4, 0.4));

        rdr->fill(theta, hist, generator);
        read(X, hist);
        out_tree->Fill();

        if(ii%1024==0){std::cout << "[ ===> " << ii << " samples are done ]" << std::endl;}
    }
}


void simulator::save() {

    outputs->cd();
    out_tree->Write();
    outputs->Close();
}
