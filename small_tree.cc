#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <TMath.h>
#include <iostream>

void small_tree()
{
    gStyle->SetOptStat(0);
    gStyle->SetOptFit();

    ROOT::EnableImplicitMT();

    double PI = TMath::Pi();

    ROOT::RDataFrame df("save", "/seaquest/users/knagai/Public/data/E906/simple_str/4pi/67_LH2_4pi/data.root", {"mass", "pT", "x1", "x2", "xF", "phi", "costh"});

    auto df1 = df.Filter("mass > 4.5");

    auto hist = df1.Histo2D({"hist", "; phi [rad]; costh", 10, -PI, PI, 10, -1., 1.}, "phi", "costh");

    auto can = new Tcanvas("can", "can", 800, 800);

    auto fit = new TF2("fit", "[0]* (1. + [1]* y* y + 2.* [2]* y* sqrt(1. - y* y) *cos(x) + 0.5* [3]* (1. - y* y)* cos(2.* x))");

    hist->Fit("fit");



    df1.Snapshot("save", "small.root");
}