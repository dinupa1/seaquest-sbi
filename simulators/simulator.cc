#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TH3D.h>
#include <TF1.h>
#include <TF2.h>
#include <TMath.h>
#include <TString.h>
#include <TSystem.h>
#include <iostream>

#include "simulator.h"

double cross_section(double lambda, double mu, double nu, double phi, double costh) {
    double weight = 1. + lambda* costh* costh + 2.* mu* costh* sqrt(1. - costh* costh) *cos(phi) + 0.5* nu* (1. - costh* costh)* cos(2.* phi);
    return (4./(lambda + 3))* weight/(1. + costh* costh);
}


reader::reader(TFile* infile, TString tname) {
    tree = (TTree*)infile->Get(tname.Data());

    num_events = tree->GetEntries();

    tree->SetBranchAddress("phi", &phi);
    tree->SetBranchAddress("costh", &costh);
    tree->SetBranchAddress("true_phi", &true_phi);
    tree->SetBranchAddress("true_costh", &true_costh);
}


void reader::fill(double theta[3], std::unique_ptr<TH2D> &hist, std::unique_ptr<TRandom3> &generator) {

    int i_start = generator->Integer(num_events/2);

    for(int ii = i_start; ii < i_start + num_entries; ii++) {
        tree->GetEntry(ii);
        hist->Fill(phi, costh, cross_section(theta[0], theta[1], theta[2], true_phi, true_costh));
    }

    // std::cout << "===> effective entries " << hist->GetEffectiveEntries() << std::endl;
    // std::cout << "===> total entries " << hist->GetEntries() << std::endl;
}


simulator::simulator():generator(std::make_unique<TRandom3>(42)) {

    TFile* inputs = TFile::Open("./data/generator.root", "read");

    train_rdr = new reader(inputs, "train_tree");
    val_rdr = new reader(inputs, "val_tree");
    test_rdr = new reader(inputs, "test_tree");

    outputs = new TFile("./data/outputs.root", "recreate");

    train_tree = new TTree("train_tree", "train_tree");
    train_tree->Branch("X", X, "X[1][12][12]/D");
    train_tree->Branch("theta", theta, "theta[3]/D");

    val_tree = new TTree("val_tree", "val_tree");
    val_tree->Branch("X", X, "X[1][12][12]/D");
    val_tree->Branch("theta", theta, "theta[3]/D");

    test_tree = new TTree("test_tree", "test_tree");
    test_tree->Branch("X", X, "X[1][12][12]/D");
    test_tree->Branch("theta", theta, "theta[3]/D");
}


void simulator::read(double X[1][12][12], std::unique_ptr<TH2D> &hist) {

    hist->Scale(1./hist->GetMaximum());

    for(int ii = 0; ii < 12; ii++) {
        for(int jj = 0; jj < 12; jj++) {
            X[0][ii][jj] = hist->GetBinContent(ii+1, jj+1);
        }
    }
}


void simulator::samples(int train_samples, int val_samples, int test_samples) {

    std::cout << "[===> training samples]" << std::endl;

    for(int ii = 0; ii < train_samples; ii++) {

        theta[0] = generator->Uniform(lambda_min, lambda_max);
        theta[1] = generator->Uniform(mu_min, mu_max);
        theta[2] = generator->Uniform(nu_min, nu_max);

        std::unique_ptr<TH2D> hist(new TH2D("hist", "", 12, -pi, pi, 12, -0.45, 0.45));

        train_rdr->fill(theta, hist, generator);
        read(X, hist);
        train_tree->Fill();

        if(ii%10000==0){std::cout << "[===> " << ii << " samples are done ]" << std::endl;}
    }

    std::cout << "[===> validation samples]" << std::endl;

    for(int ii = 0; ii < val_samples; ii++) {

        theta[0] = generator->Uniform(lambda_min, lambda_max);
        theta[1] = generator->Uniform(mu_min, mu_max);
        theta[2] = generator->Uniform(nu_min, nu_max);

        std::unique_ptr<TH2D> hist(new TH2D("hist", "", 12, -pi, pi, 12, -0.45, 0.45));

        val_rdr->fill(theta, hist, generator);
        read(X, hist);
        val_tree->Fill();

        if(ii%10000==0){std::cout << "[===> " << ii << " samples are done ]" << std::endl;}
    }

    std::cout << "[===> test samples]" << std::endl;

    for(int ii = 0; ii < test_samples; ii++) {

        theta[0] = generator->Uniform(lambda_min, lambda_max);
        theta[1] = generator->Uniform(mu_min, mu_max);
        theta[2] = generator->Uniform(nu_min, nu_max);

        std::unique_ptr<TH2D> hist(new TH2D("hist", "", 12, -pi, pi, 12, -0.45, 0.45));

        test_rdr->fill(theta, hist, generator);
        read(X, hist);
        test_tree->Fill();

        if(ii%10000==0){std::cout << "[===> " << ii << " samples are done ]" << std::endl;}
    }
}


void simulator::save() {

    outputs->cd();
    train_tree->Write();
    val_tree->Write();
    test_tree->Write();
    outputs->Close();
}
