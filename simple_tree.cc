#include <TFile.h>
#include <TTree.h>
#include <iostream>

void simple_tree()
{
    TFile oldfile("/seaquest/users/knagai/Public/data/E906/simple_str/4pi/67_LH2_4pi/data.root", "READ");
    TTree *oldtree;
    oldfile.GetObject("save", oldtree);

    oldtree->SetBranchStatus("*", 0);

    for(auto activeBranchName : {"mass", "pT", "x1", "x2", "xF", "phi", "costh"}) {oldtree->SetBranchStatus(activeBranchName, 1);}

    TFile newfile("small.root", "recreate");
    auto newtree = oldtree->CloneTree();

    auto newtree = oldtree->CloneTree();

    newtree->Print();
    newfile.Write();
}