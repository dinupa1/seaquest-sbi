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


void sim_reader::fill(double theta[3], TH2D* hist, double threshold, TRandom3* generator) {

    hist->Reset();
    int n_fill = 0;

    for(int ii = 0; ii < n_events; ii++) {
        if(n_fill == n_data) break;
        if(generator->Uniform(0., 1.) < threshold) continue;
        tree->GetEntry(ii);
        hist->Fill(phi, costh, cross_section(theta[0], theta[1], theta[2], true_phi, true_costh));
        n_fill++;
    }
    if(n_fill < n_data){std::cout << "[ ===> filled with " << n_fill << " events ] " << std::endl;}
}


simulator::simulator() {

    std::cout << "[ ===> e906 messy MC ]" << std::endl;

    gSystem->Exec("python ./simulators/generator.py");

    TFile* inputs = TFile::Open("./data/generator.root", "read");

    train_reader = new sim_reader(inputs, "X_train");
    val_reader = new sim_reader(inputs, "X_val");
    test_reader = new sim_reader(inputs, "X_test");

    outputs = new TFile("./data/outputs.root", "recreate");

    generator_0 = new TRandom3(42);
    generator_1 = new TRandom3(52);

    prior_tree = new TTree("prior_tree", "prior_tree");
    prior_tree->Branch("theta_0", theta_0, "theta_0[3]/D");

    train_tree = new TTree("train_tree", "train_tree");
    train_tree->Branch("X", X, "X[1][10][10]/D");
    train_tree->Branch("theta", theta, "theta[3]/D");
    train_tree->Branch("theta_0", theta_0, "theta_0[3]/D");

    val_tree = new TTree("val_tree", "val_tree");
    val_tree->Branch("X", X, "X[1][10][10]/D");
    val_tree->Branch("theta", theta, "theta[3]/D");
    val_tree->Branch("theta_0", theta_0, "theta_0[3]/D");

    test_tree = new TTree("test_tree", "test_tree");
    test_tree->Branch("X", X, "X[1][10][10]/D");
    test_tree->Branch("theta", theta, "theta[3]/D");
}


void simulator::prior(double theta[3], TRandom3* generator, double lambda_min, double lambda_max, double mu_min, double mu_max, double nu_min, double nu_max) {

    theta[0] = generator->Uniform(lambda_min, lambda_max);
    theta[1] = generator->Uniform(mu_min, mu_max);
    theta[2] = generator->Uniform(nu_min, nu_max);
    // std::cout << "[ ===> ii "<< 3 * ii << " lambda = " << theta[3* ii + 0] << " mu : " << theta[3* ii + 1] << " nu: " << theta[3* ii + 2] << " ]" << std::endl;
}


void simulator::read(double X[1][10][10], TH2D* hist) {

    hist->Scale(1./hist->GetMaximum());

    for(int ii = 0; ii < 10; ii++) {
        for(int jj = 0; jj < 10; jj++) {
            X[0][ii][jj] = hist->GetBinContent(ii+1, jj+1);
            // std::cout << "[ ===> " << X[0][ii][jj] << " ]" << std::endl;
        }
    }
}


void simulator::samples(int n_train, int n_val, int n_test) {

    std::cout << "[ ===> prior distribution ]" << std::endl;

    for(int ii = 0; ii < n_data; ii++) {
        prior(theta_0, generator_0, -1.5, 1.5, -0.6, 0.6, -0.6, 0.6);
        prior_tree->Fill();
    }

    std::cout << "[ ===> train events ]" << std::endl;

    TH2D* train_hist = new TH2D("train_hist", "", 10, -pi, pi, 10, -0.4, 0.4);

    for(int ii = 0; ii < n_train; ii++) {

        prior(theta_0, generator_0, -1.5, 1.5, -0.6, 0.6, -0.6, 0.6);
        prior(theta, generator_1, -1.5, 1.5, -0.6, 0.6, -0.6, 0.6);

        // std::cout << "*****************" << std::endl;

        double threshold = generator_0->Uniform(0., 0.5);
        train_reader->fill(theta, train_hist, threshold, generator_1);
        read(X, train_hist);
        train_tree->Fill();

        if(ii%10000==0){std::cout << "[ ===> " << ii << " events are done ]" << std::endl;}
    }

    std::cout << "[ ===> val events ]" << std::endl;

    TH2D* val_hist = new TH2D("val_hist", "", 10, -pi, pi, 10, -0.4, 0.4);

    for(int ii = 0; ii < n_val; ii++) {

        prior(theta_0, generator_0, -1.5, 1.5, -0.6, 0.6, -0.6, 0.6);
        prior(theta, generator_1, -1.5, 1.5, -0.6, 0.6, -0.6, 0.6);

        // std::cout << "*****************" << std::endl;

        double threshold = generator_0->Uniform(0., 0.5);
        val_reader->fill(theta, val_hist, threshold, generator_1);
        read(X, val_hist);
        val_tree->Fill();

        if(ii%10000==0){std::cout << "[ ===> " << ii << " events are done ]" << std::endl;}
    }

    std::cout << "[ ===> test events ]" << std::endl;

    TH2D* test_hist = new TH2D("test_hist", "", 10, -pi, pi, 10, -0.4, 0.4);

    for(int ii = 0; ii < n_test; ii++) {

        prior(theta, generator_1, -1., 1., -0.5, 0.5, -0.5, 0.5);

        // std::cout << "*****************" << std::endl;

        double threshold = generator_0->Uniform(0., 0.5);
        test_reader->fill(theta, test_hist, threshold, generator_1);
        read(X, test_hist);
        test_tree->Fill();

        if(ii%10000==0){std::cout << "[ ===> " << ii << " events are done ]" << std::endl;}
    }

    delete train_hist;
    delete val_hist;
    delete test_hist;
}


void simulator::save() {

    train_tree->Write();
    val_tree->Write();
    test_tree->Write();
    prior_tree->Write();

    outputs->Close();
}
