import ROOT as rt
import os
rt.gROOT.SetBatch(True)

rt.gStyle.SetOptStat(0)

def save_plot(hist, tree, feature):
    tree.Draw(feature+">>"+hist.GetName())
    cv = rt.TCanvas("cv","",700,600)
    hist.SetLineWidth(3)
    hist.Draw()
    save_dir = '/eos/user/q/qnguyen/www/InclusiveFeature_SMS_T2qq_950_925/'
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    cv.SaveAs(save_dir+"/"+feature+".png")

signal_file = rt.TFile.Open("/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_4Oct2017_Inclusive/SignalFastsim/SMS-T2qq_950_925.root")
signal_tree = signal_file.Get("InclusiveSignalRegion")

signal_Rsq = rt.TH1F("signal_Rsq",";R^{2}",100,0,1.2)
signal_MR = rt.TH1F("signal_MR",";M_{R}",100,0,2000)
signal_dPhiRazor = rt.TH1F("signal_dPhiRazor",";#Delta#phi_{Hemispheres}",100,-3.4,3.4)
signal_dPhiMinJetMET = rt.TH1F("signal_dPhiMinJetMET",";min(#Delta#phi_{jet,MET})",100,0,3.4)
signal_MT2 = rt.TH1F("signal_MT2",";M_{T2}",100,0,800)
signal_MHT = rt.TH1F("signal_MHT",";H_{T}^{miss}",100,0,1200)
signal_MET = rt.TH1F("signal_MET",";E_{T}^{miss}",100,0,1200)
signal_alphaT = rt.TH1F("signal_alphaT",";#alpha_{T}",100,0,1.2)
signal_HT = rt.TH1F("signal_HT",";H_{T}",100,0,1500)
signal_nJets = rt.TH1F("signal_nJets",";n_{Jets}",10,0,10)
signal_leadingJetPt = rt.TH1F("signal_leadingJetPt",";Leading Jet PT",100,80,500)
signal_subleadingJetPt = rt.TH1F("signal_subleadingJetPt",";Subleading Jet PT",100,0,500)

save_plot(signal_Rsq, signal_tree, "Rsq")
save_plot(signal_MR, signal_tree, "MR")
save_plot(signal_MT2, signal_tree, "MT2")
save_plot(signal_MHT, signal_tree, "MHT")
save_plot(signal_MET, signal_tree, "MET")
save_plot(signal_alphaT, signal_tree, "alphaT")
save_plot(signal_HT, signal_tree, "HT")
save_plot(signal_nJets, signal_tree, "nSelectedJets")
save_plot(signal_dPhiRazor, signal_tree, "dPhiRazor")
save_plot(signal_dPhiMinJetMET, signal_tree, "dPhiMinJetMET")
save_plot(signal_leadingJetPt, signal_tree, "leadingJetPt")
save_plot(signal_subleadingJetPt, signal_tree, "subleadingJetPt")
