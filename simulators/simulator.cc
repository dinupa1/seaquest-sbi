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

    n_events = tree->GetEntries();

    tree->SetBranchAddress("pT", &pT);
    tree->SetBranchAddress("phi", &phi);
    tree->SetBranchAddress("costh", &costh);
    tree->SetBranchAddress("true_pT", &true_pT);
    tree->SetBranchAddress("true_phi", &true_phi);
    tree->SetBranchAddress("true_costh", &true_costh);
}


void sim_reader::fill(double theta[3], std::unique_ptr<TH2D> &hist_0, std::unique_ptr<TH2D> &hist_1, std::unique_ptr<TH2D> &hist_2, std::unique_ptr<TRandom3> &generator) {

    int n_fill = 0;
    double threshold = generator->Uniform(0., 0.5);

    for(int ii = 0; ii < n_events; ii++) {
        if(n_fill == n_data) break;
        if(generator->Uniform(0., 1.) < threshold) continue;
        tree->GetEntry(ii);
        hist_0->Fill(phi, costh, cross_section(theta[0], theta[1], theta[2], true_phi, true_costh));
        hist_1->Fill(cos(phi), costh, cross_section(theta[0], theta[1], theta[2], true_phi, true_costh));
        hist_2->Fill(cos(2. * phi), costh, cross_section(theta[0], theta[1], theta[2], true_phi, true_costh));
        n_fill++;
    }
    if(n_fill < n_data){std::cout << "[ ===> filled with " << n_fill << " events ] " << std::endl;}
}


simulator::simulator():generator(std::make_unique<TRandom3>(42)) {

    std::cout << "[ ===> e906 messy MC ]" << std::endl;

    gSystem->Exec("python ./simulators/generator.py");

    TFile* inputs = TFile::Open("./data/generator.root", "read");

    train_reader = new sim_reader(inputs, "X_train");
    test_reader = new sim_reader(inputs, "X_test");

    outputs = new TFile("./data/outputs.root", "recreate");

    train_tree = new TTree("train_tree", "train_tree");
    train_tree->Branch("X", X, "X[3][10][10]/D");
    train_tree->Branch("theta", theta, "theta[3]/D");

    test_tree = new TTree("test_tree", "test_tree");
    test_tree->Branch("X", X, "X[3][10][10]/D");
    test_tree->Branch("theta", theta, "theta[3]/D");
}


void simulator::read(double X[3][10][10], std::unique_ptr<TH2D> &hist_0, std::unique_ptr<TH2D> &hist_1, std::unique_ptr<TH2D> &hist_2) {

    hist_0->Scale(1./hist_0->GetMaximum());
    hist_1->Scale(1./hist_1->GetMaximum());
    hist_2->Scale(1./hist_2->GetMaximum());

    for(int ii = 0; ii < 10; ii++) {
        for(int jj = 0; jj < 10; jj++) {
            X[0][ii][jj] = hist_0->GetBinContent(ii+1, jj+1);
            X[1][ii][jj] = hist_1->GetBinContent(ii+1, jj+1);
            X[2][ii][jj] = hist_2->GetBinContent(ii+1, jj+1);
            // std::cout << "[ ===> " << X[0][ii][jj] << " ]" << std::endl;
        }
    }
}


void simulator::samples(int n_train=1024000, int n_test=100) {

    std::cout << "[ ===> train events ]" << std::endl;

    for(int ii = 0; ii < n_train; ii++) {

        theta[0] = generator->Uniform(-1.5, 1.5);
        theta[1] = generator->Uniform(-0.6, 0.6);
        theta[2] = generator->Uniform(-0.6, 0.6);

        std::unique_ptr<TH2D> hist_0(new TH2D("hist_0", "", 10, -pi, pi, 10, -0.4, 0.4));
        std::unique_ptr<TH2D> hist_1(new TH2D("hist_1", "", 10, -1., 1., 10, -0.4, 0.4));
        std::unique_ptr<TH2D> hist_2(new TH2D("hist_2", "", 10, -1., 1., 10, -0.4, 0.4));

        // std::cout << "*****************" << std::endl;

        train_reader->fill(theta, hist_0, hist_1, hist_2, generator);
        read(X, hist_0, hist_1, hist_2);
        train_tree->Fill();

        if(ii%10000==0){std::cout << "[ ===> " << ii << " train events are done ]" << std::endl;}
    }

    std::cout << "[ ===> test events ]" << std::endl;

    for(int ii = 0; ii < n_test; ii++) {

        theta[0] = -1. + (2./(n_test -1.)) * ii;
        theta[1] = -0.4 + (0.8/(n_test -1.)) * ii;
        theta[2] = -0.4 + (0.8/(n_test -1.)) * ii;

        std::unique_ptr<TH2D> hist_0(new TH2D("hist_0", "", 10, -pi, pi, 10, -0.4, 0.4));
        std::unique_ptr<TH2D> hist_1(new TH2D("hist_1", "", 10, -1., 1., 10, -0.4, 0.4));
        std::unique_ptr<TH2D> hist_2(new TH2D("hist_2", "", 10, -1., 1., 10, -0.4, 0.4));

        // std::cout << "*****************" << std::endl;

        test_reader->fill(theta, hist_0, hist_1, hist_2, generator);
        read(X, hist_0, hist_1, hist_2);
        test_tree->Fill();

        if(ii%10==0){std::cout << "[ ===> " << ii << " test events are done ]" << std::endl;}
    }
}


void simulator::save() {

    train_tree->Write();
    test_tree->Write();

    outputs->Close();
}
