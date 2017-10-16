import ROOT as rt
import os
import random

rt.gROOT.SetBatch(True)
rt.TH1.AddDirectory(rt.kFALSE)
rt.gStyle.SetOptStat(0)
rt.gStyle.SetLegendBorderSize(0)

SAVEFILE = 'ControlRegionPlots.root'
newfile = rt.TFile(SAVEFILE,"recreate")
newfile.Close()

SAVEDIR = '/eos/user/q/qnguyen/www/InclusiveFeature_CR_NoMTcut/'

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
    cv = rt.TCanvas("cv","",700,600)
    lumi = "35.9*1000"
    if sample is "Data": lumi = "1."
    hist.SetLineWidth(3)
    if sample is not 'Data': 
        hist.SetFillColor(COLORS[sample])
        hist.SetLineColor(COLORS[sample])
    tree.Draw(feature+">>"+hist.GetName(),"weight*"+lumi+"*((abs(lep1Type) == 11 || abs(lep1Type) == 13) && lep1PassTight && ((abs(lep1Type) == 11 && lep1.Pt() > 30) || (abs(lep1Type) == 13 && lep1.Pt() > 25)) && MET > 100)")
    if not os.path.isdir(SAVEDIR): os.makedirs(SAVEDIR)
    if hist.Integral() > 0:
        cv.SaveAs(SAVEDIR+"/"+hist.GetName()+".png")
    del cv

def save_plot(hist, tree, feature, sample):
    print hist.Integral()
    cr = rt.TFile.Open(SAVEFILE,"update")
    hist.Write()
    cr.Close()

SAMPLES = {}
filedir = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/RunTwoInclusiveControlRegion/2016/V3p15_13Oct2017_Inclusive/OneLeptonFull/'
SAMPLES['WJets'] = {'file': filedir+"/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_WJets_1pb_weighted.root"}
SAMPLES['TTJets'] = {'file': filedir+"InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_TTJets_1pb_weighted.root"}
SAMPLES['Other'] = {'file': filedir+"InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_Other_1pb_weighted.root"}
SAMPLES['QCD'] = {'file': filedir+"InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_QCD_1pb_weighted.root"}
SAMPLES['DYJets'] = {'file': filedir+"InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_DYJets_1pb_weighted.root"}
SAMPLES['SingleTop'] = {'file': filedir+"InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_SingleTop_1pb_weighted.root"}
SAMPLES['Data'] = {'file': filedir+"InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_Data_NoDuplicates_RazorSkim_GoodLumiGolden.root"}

# Test files for quick and dirty check
SAMPLES['WJets']['test'] = filedir+'jobs/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_WJetsToLNu_Pt-250To400_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8.Job0of13.root'
SAMPLES['TTJets']['test'] = filedir+'jobs/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_TTJets_HT-600to800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job15of19.root'
SAMPLES['Other']['test'] = filedir+'jobs/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_WWTo2L2Nu_13TeV-powheg.Job0of2.root'
SAMPLES['QCD']['test'] = filedir+'jobs/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job49of61.root'
SAMPLES['DYJets']['test'] = filedir+'jobs/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_DYJetsToLL_M-50_HT-200to400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job24of251.root'
SAMPLES['SingleTop']['test'] = filedir+'jobs/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_ST_t-channel_antitop_4f_inclusiveDecays_13TeV-powhegV2-madspin-pythia8_TuneCUETP8M1.Job121of678.root'
SAMPLES['Data']['test'] = filedir+'/jobs/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_SingleElectron_2016B_03Feb2017.Job145of199.root'

#print SAMPLES['Data']['test']

for sample in SAMPLES:
    SAMPLES[sample]['feature'] = {}
    _file = rt.TFile.Open(SAMPLES[sample]['file'])
    _tree = _file.Get('InclusiveControlRegion')

    SAMPLES[sample]['feature']['MR'] =  rt.TH1F(sample+"_MR",sample+";M_{R}",100,0,2000)
    SAMPLES[sample]['feature']['Rsq'] =  rt.TH1F(sample+"_Rsq",sample+";R^{2}",100,0,1.2)
    SAMPLES[sample]['feature']['dPhiMinJetMET'] =  rt.TH1F(sample+"_dPhiMinJetMET",sample+";min(#Delta#phi_{jet,MET})",100,0,3.4)
    SAMPLES[sample]['feature']['MT2'] =  rt.TH1F(sample+"_MT2",sample+";M_{T2}",100,0,300)
    SAMPLES[sample]['feature']['alphaT'] =  rt.TH1F(sample+"_alphaT",sample+";#alpha_{T}",100,0,1.2)
    SAMPLES[sample]['feature']['MHT'] =  rt.TH1F(sample+"_MHT",sample+";H_{T}^{miss}",100,0,500)
    SAMPLES[sample]['feature']['MET'] =  rt.TH1F(sample+"_MET",sample+";E_{T}^{miss}",100,0,300)
    SAMPLES[sample]['feature']['HT'] =  rt.TH1F(sample+"_HT",sample+";H_{T}",100,0,1500)
    SAMPLES[sample]['feature']['nSelectedJets'] =  rt.TH1F(sample+"_nSelectedJets",sample+";n_{Jets}",10,0,10)
    SAMPLES[sample]['feature']['dPhiRazor'] =  rt.TH1F(sample+"_dPhiRazor",sample+";#Delta#phi_{Hemispheres}",100,-3.4,3.4)
    SAMPLES[sample]['feature']['leadingJetPt'] =  rt.TH1F(sample+"_leadingJetPt",sample+";Leading Jet PT",100,80,400)
    SAMPLES[sample]['feature']['subleadingJetPt'] =  rt.TH1F(sample+"_subleadingJetPt",sample+";Subleading Jet PT",100,20,250)
  
    features = list(SAMPLES[sample]['feature'])
    #random.shuffle(features)
    for feature in features:
        _hist = SAMPLES[sample]['feature'][feature]
        _hist.SetDirectory(rt.gROOT)
        draw_plot(_hist, _tree, feature, sample) 
        save_plot(_hist, _tree, feature, sample)

    for feature in features:
        _hist = SAMPLES[sample]['feature'][feature]
        if _hist.Integral() == 0:
            print "Redone {}_{} for a stupid pyROOT pointer/ownership problem that does not fill in the first histogram".format(sample,feature)
            draw_plot(_hist, _tree, feature, sample) 
            save_plot(_hist, _tree, feature, sample)

STACKS = {}
for feature in SAMPLES['Data']['feature']:
    STACKS[feature] = rt.THStack(feature+"_stack","")

savefile = rt.TFile.Open(SAVEFILE,"read")
for sample in SAMPLES:
    for feature in SAMPLES['Data']['feature']:
        if sample is not 'Data': 
            _hist = savefile.Get(sample+"_"+feature)
            _hist.SetFillColor(COLORS[sample])
            _hist.SetLineColor(COLORS[sample])
            STACKS[feature].Add(_hist)

for feature in SAMPLES['Data']['feature']:
    cv = rt.TCanvas("cv","",700,600)
    STACKS[feature].Draw("hist")
    _hist = savefile.Get("Data_"+feature)
    _hist.SetLineWidth(2)
    _hist.SetMarkerStyle(rt.kFullCircle);
    _hist.Draw("E1 SAME")
    leg = cv.BuildLegend(0.5,0.7,0.85,0.88)
    leg.SetNColumns(2)
    if not os.path.isdir(SAVEDIR): os.makedirs(SAVEDIR)
    cv.SaveAs(SAVEDIR+"/_"+feature+".png")

