import ROOT as rt
import os

rt.gROOT.SetBatch(True)

rt.gStyle.SetOptStat(0)
COLORS = {
        "WJets":rt.kRed+1, 
        "WJetsInv":rt.kRed+1, 
        "DYJets":rt.kBlue+1, 
        "DYJetsInv":rt.kBlue+1, 
        "TTJets":rt.kGreen+2, 
        "TTJets1L":rt.kGreen+2, 
        "TTJets2L":rt.kGreen+3, 
        "ZInv":rt.kCyan+1, 
        "QCD":rt.kMagenta, 
        "SingleTop":rt.kOrange-3, 
        "VV":rt.kViolet+3, 
        "TTV":rt.kGreen-7, 
        "DYJetsLow":rt.kBlue+1, 
        "GJets":rt.kOrange, 
        "GJetsInv":rt.kOrange, 
        "GJetsFrag":(rt.kOrange+4), 
        "Other":rt.kAzure+4
        }

def draw_plot(hist, tree, feature, sample):
    cr = rt.TFile("ControlRegionPlots.root","update")
    lumi = '35.9'
    if sample is "Data": lumi = '1.'
    cv = rt.TCanvas("cv","",700,600)
    #tree.Draw(feature+">>"+hist.GetName(),"weight")
    tree.Draw(feature+">>"+hist.GetName(),"weight*"+lumi+"*(lep1MT<100)")
    hist.SetLineWidth(1)
    #if sample is not 'Data': hist.SetFillColor(COLORS[sample])
    hist.Draw()
    save_dir = '/eos/user/q/qnguyen/www/InclusiveFeature_CR/'
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    cv.SaveAs(save_dir+"/"+hist.GetName()+".png")
    hist.Write()
    cr.Close()
    return hist

SAMPLES = {}
filedir = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/RunTwoInclusiveControlRegion/2016/V3p15_4Oct2017_Inclusive/OneLeptonFull/'
SAMPLES['WJets'] = {'file': filedir+"/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_WJets_1pb_weighted.root"}
SAMPLES['TTJets'] = {'file': filedir+"InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_TTJets_1pb_weighted.root"}
SAMPLES['Other'] = {'file': filedir+"InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_Other_1pb_weighted.root"}
SAMPLES['QCD'] = {'file': filedir+"InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_QCD_1pb_weighted.root"}
SAMPLES['DYJets'] = {'file': filedir+"InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_DYJets_1pb_weighted.root"}
SAMPLES['SingleTop'] = {'file': filedir+"InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_SingleTop_1pb_weighted.root"}
SAMPLES['Data'] = {'file': filedir+"InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_Data_NoDuplicates_RazorSkim_GoodLumiGolden.root"}

for sample in SAMPLES:
    #SAMPLES[sample]['tree'] = rt.TFile.Open(SAMPLES[sample]['file']).Get('InclusiveControlRegion')
    _file = rt.TFile.Open(SAMPLES[sample]['file'])
    _tree = _file.Get('InclusiveControlRegion')
    SAMPLES[sample]['Rsq'] =  rt.TH1F(sample+"_Rsq",";R^{2}",100,0,1.2)
    SAMPLES[sample]['MR'] =  rt.TH1F(sample+"_MR",";M_{R}",100,0,2000)
    SAMPLES[sample]['dPhiRazor'] =  rt.TH1F(sample+"_dPhiRazor",";#Delta#phi_{Hemispheres}",100,-3.4,3.4)
    SAMPLES[sample]['dPhiMinJetMET'] =  rt.TH1F(sample+"_dPhiMinJetMET",";min(#Delta#phi_{jet,MET})",100,0,3.4)
    SAMPLES[sample]['MT2'] =  rt.TH1F(sample+"_MT2",";M_{T2}",100,0,300)
    SAMPLES[sample]['MHT'] =  rt.TH1F(sample+"_MHT",";H_{T}^{miss}",100,0,500)
    SAMPLES[sample]['MET'] =  rt.TH1F(sample+"_MET",";E_{T}^{miss}",100,0,300)
    SAMPLES[sample]['alphaT'] =  rt.TH1F(sample+"_alphaT",";#alpha_{T}",100,0,1.2)
    SAMPLES[sample]['HT'] =  rt.TH1F(sample+"_HT",";H_{T}",100,0,1500)
    SAMPLES[sample]['nJets'] =  rt.TH1F(sample+"_nJets",";n_{Jets}",10,0,10)
    SAMPLES[sample]['leadingJetPt'] =  rt.TH1F(sample+"_leadingJetPt",";Leading Jet PT",100,80,400)
    SAMPLES[sample]['subleadingJetPt'] =  rt.TH1F(sample+"_subleadingJetPt",";Subleading Jet PT",100,20,250)
    
    SAMPLES[sample]['Rsq'] = draw_plot(SAMPLES[sample]['Rsq'], _tree, "Rsq", sample) 
    SAMPLES[sample]['MR'] = draw_plot(SAMPLES[sample]['MR'], _tree, "MR", sample) 
    SAMPLES[sample]['MT2'] = draw_plot(SAMPLES[sample]['MT2'], _tree, "MT2", sample) 
    SAMPLES[sample]['MHT'] = draw_plot(SAMPLES[sample]['MHT'], _tree, "MHT", sample) 
    SAMPLES[sample]['MET'] = draw_plot(SAMPLES[sample]['MET'], _tree, "MET", sample) 
    SAMPLES[sample]['alphaT'] = draw_plot(SAMPLES[sample]['alphaT'], _tree, "alphaT", sample) 
    SAMPLES[sample]['HT'] = draw_plot(SAMPLES[sample]['HT'], _tree, "HT", sample) 
    SAMPLES[sample]['nJets'] = draw_plot(SAMPLES[sample]['nJets'], _tree, "nSelectedJets", sample) 
    SAMPLES[sample]['dPhiRazor'] = draw_plot(SAMPLES[sample]['dPhiRazor'], _tree, "dPhiRazor", sample) 
    SAMPLES[sample]['dPhiMinJetMET'] = draw_plot(SAMPLES[sample]['dPhiMinJetMET'], _tree, "dPhiMinJetMET", sample) 
    SAMPLES[sample]['leadingJetPt'] = draw_plot(SAMPLES[sample]['leadingJetPt'], _tree, "leadingJetPt", sample) 
    SAMPLES[sample]['subleadingJetPt'] = draw_plot(SAMPLES[sample]['subleadingJetPt'], _tree, "subleadingJetPt", sample) 

STACKS = {}
for feature in SAMPLES['Data']:
    if feature is 'file': continue
    STACKS[feature] = rt.THStack(feature+"_stack","")

for sample in SAMPLES:
    for feature in SAMPLES['Data']:
        if feature is 'file': continue
        if sample is not 'Data': STACKS[feature].Add(SAMPLES[sample][feature])

for feature in SAMPLES['Data']:
    if feature is 'file': continue
    cv = rt.TCanvas("cv","",700,600)
    STACKS[feature].Draw()
    SAMPLES['Data'][feature].Draw("same")
    save_dir = '/eos/user/q/qnguyen/www/InclusiveFeature_CR/'
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    cv.SaveAs(save_dir+"/"+feature+".png")


