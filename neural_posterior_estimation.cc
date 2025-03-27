#include "simulations.h"


double cross_section_ratio(double phi, double costh, double lambda, double mu, double nu) {

    double cos_theta2 = costh* costh;
    double sin_theta2 = 1. - cos_theta2;
    double sin_theta = sqrt(sin_theta2);
    double cos_phi = cos(phi);
    double cos_2phi = cos(2.* phi);
    double sin_2theta = 2.* sin_theta* costh;

    double numerator = 1. + lambda* cos_theta2 + mu* sin_2theta* cos_phi + 0.5* nu* sin_theta2* cos_2phi;
    double denominator = 1. + cos_theta2;
    double coef = 4./(lambda + 3);

    return coef* numerator/denominator;
}


void prior_sample(double lambda, double mu, double nu) {
    theta[0] =  events->Uniform(-lambda, lambda);
    theta[1] =  events->Uniform(-mu, mu);
    theta[2] =  events->Uniform(-nu, nu);
}


void simulation_sample(TTree* tree, TH2D* hist) {

    int num = tree->GetEntries();

    int i_start = int(events->Uniform(0., num-num_data));
    int i_end = i_start + num_data;

    for(int ii = i_start; ii < i_end; ii++) {
        tree->GetEntry(ii);
        weight1 = cross_section_ratio(true_phi, true_costh, theta[0], theta[1], theta[2]);
        hist->Fill(phi, costh, weight* weight1);
    }

    hist->Scale(1./hist->GetMaximum());

    // std::cout << "===> number of effective entries " << hist->GetEffectiveEntries() << std::endl;
}


void likelihood_sample(TH2D* hist) {

    for(int ii = 0; ii < num_bins; ii++) {
        for(int jj = 0; jj < num_bins; jj++) {
            X[0][ii][jj] = hist->GetBinContent(ii+1, jj+1);
        }
    }
}


void input_tree(TTree* tree) {
    tree->SetBranchAddress("mass",            &mass);
    tree->SetBranchAddress("pT",              &pT);
    tree->SetBranchAddress("xF",              &xF);
    tree->SetBranchAddress("phi",             &phi);
    tree->SetBranchAddress("costh",           &costh);
    tree->SetBranchAddress("true_phi",        &true_phi);
    tree->SetBranchAddress("true_costh",      &true_costh);
    tree->SetBranchAddress("true_mass",       &true_mass);
    tree->SetBranchAddress("true_pT",         &true_pT);
    tree->SetBranchAddress("true_xF",         &true_xF);
    tree->SetBranchAddress("weight",          &weight);
}


void out_tree(TTree* tree) {
    tree->Branch("theta",      theta,      "theta[3]/D");
    tree->Branch("X",          X,          "X[1][12][12]/D");
}


int neural_posterior_estimation() {

    //
    // forward simulation
    //

    //
    // train, val, test split
    //
    gSystem->Exec("python generation.py");

    //
    // events with weights
    //
    // gSystem->Exec("python events_with_weights.py");

    TFile* input_file = TFile::Open("./data/generation.root", "read");

    train_tree = (TTree*)input_file->Get("train_tree");

    input_tree(train_tree);

    val_tree = (TTree*)input_file->Get("val_tree");

    input_tree(val_tree);

    test_tree = (TTree*)input_file->Get("test_tree");

    input_tree(test_tree);


    TFile* outfile = new TFile("./data/outfile.root", "recreate");

    events = new TRandom3(42);

    //
    // train samples
    //
    std::cout << "===> simulating train events" << std::endl;

    train_out = new TTree("train_out", "train_out");

    out_tree(train_out);

    for(int ii = 0; ii < train_samples* base_size; ii++) {

        prior_sample(lambda_limits, mu_limits, nu_limits);

        TH2D* phi_costh = new TH2D("phi_costh", "", num_bins, -pi, pi, num_bins, -costh_limits, costh_limits);

        simulation_sample(train_tree, phi_costh);

        likelihood_sample(phi_costh);

        train_out->Fill();
        delete phi_costh;
        if(ii%1024 == 0){std::cout << "[===> " << double(ii)/double(train_samples* base_size) << " samples are done ]" << std::endl;}
    }

    //
    // val samples
    //
    std::cout << "===> simulating validation events" << std::endl;

    val_out = new TTree("val_out", "val_out");

    out_tree(val_out);

    for(int ii = 0; ii < val_samples* base_size; ii++) {

        prior_sample(lambda_limits, mu_limits, nu_limits);

        TH2D* phi_costh = new TH2D("phi_costh", "", num_bins, -pi, pi, num_bins, -costh_limits, costh_limits);

        simulation_sample(val_tree, phi_costh);

        likelihood_sample(phi_costh);

        val_out->Fill();
        delete phi_costh;
        if(ii%1024 == 0){std::cout << "[===> " << double(ii)/double(val_samples* base_size) << " samples are done ]" << std::endl;}
    }

    //
    // test samples
    //
    std::cout << "===> simulating test events" << std::endl;

    test_out = new TTree("test_out", "test_out");

    out_tree(test_out);

    for(int ii = 0; ii < test_samples* base_size; ii++) {

        prior_sample(1.0, 0.5, 0.5); // test in sub range

        TH2D* phi_costh = new TH2D("phi_costh", "", num_bins, -pi, pi, num_bins, -costh_limits, costh_limits);

        simulation_sample(test_tree, phi_costh);

        likelihood_sample(phi_costh);

        test_out->Fill();
        delete phi_costh;
        if(ii%1024 == 0){std::cout << "[===> " << double(ii)/double(test_samples* base_size) << " samples are done ]" << std::endl;}
    }

    train_out->Write();
    val_out->Write();
    test_out->Write();

    outfile->Close();

    //
    // inference
    //
    gSystem->Exec("python inference.py");

    //
    // bootstrapping
    //
    // gSystem->Exec("python bootstrapping.py");

    return 0;
}
