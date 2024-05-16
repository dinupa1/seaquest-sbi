#include <TFile.h>
#include <TTree.h>
#include <iostream>

void simple_tree()
{
    ROOT::EnableImplicitMT();

    ROOT::RDataFrame df("save", "/seaquest/users/knagai/Public/data/E906/simple_str/4pi/67_LH2_4pi/data.root", {"mass", "pT", "x1", "x2", "xF", "phi", "costh"});

    auto df1 = df.Filter("mass > 4.5");

    df1..Snapshot("save", "small.root");
}