#include <ROOT/RDataFrame.hxx>
#include <TH2D.h>
#include <TMath.h>
#include <iostream>

void small_tree()
{
    gStyle->SetOptStat(0);
    gStyle->SetOptFit();
    gStyle->SetPalette(71);

    ROOT::EnableImplicitMT();

    double PI = TMath::Pi();

    ROOT::RDataFrame df("save", "/seaquest/users/knagai/Public/data/E906/simple_str/4pi/67_LH2_4pi/data.root", {"mass", "pT", "x1", "x2", "xF", "phi", "costh"});

    auto df1 = df.Filter("mass > 4.5");

    auto hist = df1.Histo2D({"hist", "; phi [rad]; costh", 10, -PI, PI, 10, -1., 1.}, "phi", "costh");

    auto can = new TCanvas("can", "can", 800, 800);

    auto fit = new TF2("fit", "[0]* (1. + [1]* y* y + 2.* [2]* y* sqrt(1. - y* y) *cos(x) + 0.5* [3]* (1. - y* y)* cos(2.* x))");
    fit->SetParLimits(0, 0., 1.);
    fit->SetParLimits(1, -1., 1.);
    fit->SetParLimits(2, -0.5, 0.5);
    fit->SetParLimits(3, -0.5, 0.5);
    fit->SetParNames("A", "#lambda", "#mu", "#nu");

    hist->Scale(1./hist->Integral());

    hist->Fit("fit");
    hist->Draw("COLZ");
    can->SaveAs("particle.png");

    df1.Snapshot("save", "small.root");
}