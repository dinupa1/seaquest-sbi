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

/*
double cross_section(double lambda, double mu, double nu, double phi, double costh) {
    double weight = 1. + lambda* costh* costh + 2.* mu* costh* sqrt(1. - costh* costh) *cos(phi) + 0.5* nu* (1. - costh* costh)* cos(2.* phi);
    return weight/(1. + costh* costh);
}
*/


sim_reader::sim_reader(TFile* infile, TString tname) {
    tree = (TTree*)infile->Get(tname.Data());

    num_events = tree->GetEntries();

    tree->SetBranchAddress("pT", &pT);
    tree->SetBranchAddress("phi", &phi);
    tree->SetBranchAddress("costh", &costh);
    tree->SetBranchAddress("true_pT", &true_pT);
    tree->SetBranchAddress("true_phi", &true_phi);
    tree->SetBranchAddress("true_costh", &true_costh);

    cross_section = new TF2("cross_section",
                            [](double* x, double* theta) {
                                double phi = x[0];
                                double costh = x[1];
                                double lambda = theta[0];
                                double mu = theta[1];
                                double nu = theta[2];
                                double numerator = 1. + lambda* costh* costh + 2.* mu* costh* sqrt(1. - costh* costh)* cos(phi) + 0.5* nu* (1. - costh* costh)* cos(2* phi);
                                double denominator = 1. + costh* costh;
                                return numerator/denominator;},
                            -pi, pi, -1., 1., 3);
}


void sim_reader::fill(double theta[3], std::unique_ptr<TH2D> &hist, std::unique_ptr<TRandom3> &generator) {

    cross_section->SetParameter(0, theta[0]);
    cross_section->SetParameter(1, theta[1]);
    cross_section->SetParameter(2, theta[2]);

    double weight_max = cross_section->GetMaximum(-pi, pi);
    double weight_min = cross_section->GetMinimum(-pi, pi);

    int accepted = 0;

    for(int ii = 0; ii < num_events; ii++) {
        if(accepted == num_data) {break;}
        tree->GetEntry(ii);
        double threshold = cross_section->Eval(true_phi, true_costh);
        if(threshold > generator->Uniform(weight_min, weight_max)) {
            hist->Fill(phi, costh);
            accepted++;
        }
    }
}


simulator::simulator():generator(std::make_unique<TRandom3>(42)) {

    gSystem->Exec("python ./simulators/generator.py");

    TFile* inputs = TFile::Open("./data/generator.root", "read");

    rdr = new sim_reader(inputs, "tree");

    outputs = new TFile("./data/outputs.root", "recreate");

    out_tree = new TTree("out_tree", "out_tree");
    out_tree->Branch("X", X, "X[1][12][12]/D");
    out_tree->Branch("theta", theta, "theta[3]/D");
}


void simulator::read(double X[1][12][12], std::unique_ptr<TH2D> &hist) {

    hist->Scale(1./hist->GetMaximum());

    for(int ii = 0; ii < 12; ii++) {
        for(int jj = 0; jj < 12; jj++) {
            X[0][ii][jj] = hist->GetBinContent(ii+1, jj+1);
            // std::cout << "[ ===> " << X[0][ii][jj] << " ]" << std::endl;
        }
    }
}


void simulator::samples(int num_samples) {

    for(int ii = 0; ii < num_samples; ii++) {

        theta[0] = generator->Uniform(-1.5, 1.5);
        theta[1] = generator->Uniform(-0.6, 0.6);
        theta[2] = generator->Uniform(-0.6, 0.6);

        std::unique_ptr<TH2D> hist(new TH2D("hist", "", 12, -pi, pi, 12, -0.4, 0.4));

        // std::cout << "*****************" << std::endl;

        rdr->fill(theta, hist, generator);
        read(X, hist);
        out_tree->Fill();

        if(ii%10240==0){std::cout << "[ ===> " << ii << " samples are done ]" << std::endl;}
    }
}


void simulator::save() {

    outputs->cd();
    out_tree->Write();
    outputs->Close();
}
