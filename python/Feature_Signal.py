import ROOT as rt
import os
import random

rt.gROOT.SetBatch(True)
rt.TH1.AddDirectory(rt.kFALSE)
rt.gStyle.SetOptStat(0)
rt.gStyle.SetLegendBorderSize(0)

SAVEFILE = 'SignalRegionPlots.root'
newfile = rt.TFile(SAVEFILE,"recreate")
newfile.Close()

SAVEDIR = '/eos/user/q/qnguyen/www/InclusiveFeature_Signal_TriggerCutOR'
if not os.path.isdir(SAVEDIR): 
    os.mkdir(SAVEDIR)
    print("Making {}".format(SAVEDIR))
CUT = 'leadingJetPt>100 && (box==21 || box==22) && ((HT > 100 && jet1MT > 100 && MET > 200 && MHT > 200) || MT2 > 200 || (Rsq > 0.2 && MR > 200))'

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
    lumi = "35.9"
    if 'T2qq' in sample: lumi = "1."
    hist.SetLineWidth(3)
    if 'T2qq' not in sample: 
        hist.SetFillColor(COLORS[sample])
        hist.SetLineColor(COLORS[sample])
    tree.Draw(feature+">>"+hist.GetName(),"weight*"+lumi+"*("+CUT+")")
    if not os.path.isdir(SAVEDIR): os.makedirs(SAVEDIR)
    if hist.Integral() > 0:
        cv.SaveAs(SAVEDIR+"/"+hist.GetName()+".png")
    del cv

def save_plot(hist, tree, feature, sample):
    print hist.Integral()
    cr = rt.TFile.Open(SAVEFILE,"update")
    hist.Write()
    cr.Close()

def sum_stack(stack):
    _stack = stack.GetStack().Last();
    return _stack.Integral()

SAMPLES = {}
filedir = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_13Oct2017_Inclusive/Signal/'
SAMPLES['WJets'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_WJets_1pb_weighted.root"}
SAMPLES['TTJets'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_TTJetsHTBinned_1pb_weighted.root"}
SAMPLES['Other'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_Other_1pb_weighted.root"}
SAMPLES['QCD'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_QCD_1pb_weighted.root"}
SAMPLES['DYJets'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_DYJets_1pb_weighted.root"}
SAMPLES['SingleTop'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_SingleTop_1pb_weighted.root"}
SAMPLES['ZInv'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_ZInv_1pb_weighted.root"}

SAMPLES['T2qq_1450_1400'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_1450_1400.root"}
SAMPLES['T2qq_450_425'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_450_425.root"}
SAMPLES['T2qq_900_100'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_900_100.root"}

# Test files for quick and dirty check
SAMPLES['WJets']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_WJetsToLNu_Pt-250To400_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8.Job0of13.root'
SAMPLES['TTJets']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_TTJets_HT-600to800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job15of19.root'
SAMPLES['Other']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_WWTo2L2Nu_13TeV-powheg.Job0of2.root'
SAMPLES['QCD']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job49of61.root'
SAMPLES['DYJets']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_DYJetsToLL_M-50_HT-200to400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job24of251.root'
SAMPLES['SingleTop']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_ST_t-channel_antitop_4f_inclusiveDecays_13TeV-powhegV2-madspin-pythia8_TuneCUETP8M1.Job121of678.root'
SAMPLES['ZInv']['test'] = filedir+"/jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_ZJetsToNuNu_HT-200To400_13TeV-madgraph.Job201of512.root"
SAMPLES['T2qq_1450_1400']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_1450_1400.root"
SAMPLES['T2qq_450_425']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_450_425.root"
SAMPLES['T2qq_900_100']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_900_100.root"
   
print SAMPLES['WJets']['test']
print SAMPLES['ZInv']['file']

for sample in SAMPLES:
        _file = rt.TFile.Open(SAMPLES[sample]['file'])
        _tree = _file.Get('InclusiveSignalRegion')
        
        SAMPLES[sample]['feature'] = {}

        SAMPLES[sample]['feature']['MR'] =  rt.TH1F(sample+"_MR",sample+";M_{R}",100,0,2500)
        SAMPLES[sample]['feature']['Rsq'] =  rt.TH1F(sample+"_Rsq",sample+";R^{2}",100,0,1.2)
        SAMPLES[sample]['feature']['dPhiMinJetMET'] =  rt.TH1F(sample+"_dPhiMinJetMET",sample+";min(#Delta#phi_{jet,MET})",100,0,3.4)
        SAMPLES[sample]['feature']['MT2'] =  rt.TH1F(sample+"_MT2",sample+";M_{T2}",100,0,1000)
        SAMPLES[sample]['feature']['alphaT'] =  rt.TH1F(sample+"_alphaT",sample+";#alpha_{T}",100,0,1.2)
        SAMPLES[sample]['feature']['MHT'] =  rt.TH1F(sample+"_MHT",sample+";H_{T}^{miss}",100,0,1200)
        SAMPLES[sample]['feature']['MET'] =  rt.TH1F(sample+"_MET",sample+";E_{T}^{miss}",100,0,1200)
        SAMPLES[sample]['feature']['HT'] =  rt.TH1F(sample+"_HT",sample+";H_{T}",100,0,2000)
        SAMPLES[sample]['feature']['nSelectedJets'] =  rt.TH1F(sample+"_nSelectedJets",sample+";n_{Jets}",10,0,10)
        SAMPLES[sample]['feature']['dPhiRazor'] =  rt.TH1F(sample+"_dPhiRazor",sample+";#Delta#phi_{Hemispheres}",100,-3.4,3.4)
        SAMPLES[sample]['feature']['leadingJetPt'] =  rt.TH1F(sample+"_leadingJetPt",sample+";Leading Jet PT",100,80,1000)
        SAMPLES[sample]['feature']['subleadingJetPt'] =  rt.TH1F(sample+"_subleadingJetPt",sample+";Subleading Jet PT",100,20,1000)
        SAMPLES[sample]['feature']['leadingJetCISV'] =  rt.TH1F(sample+"_leadingJetCISV",sample+";Leading Jet's CISV",100,0.,1.)
        SAMPLES[sample]['feature']['jet1MT'] =  rt.TH1F(sample+"_jet1MT",sample+";m^{T}(jet_{1},E_{T}^{miss})",100,0.,2000.)
      
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
                print ("Redone {}_{} for a stupid pyROOT pointer/ownership problem that does not fill in the first histogram".format(sample,feature))
                draw_plot(_hist, _tree, feature, sample) 
                save_plot(_hist, _tree, feature, sample)
    
#    # Draw sms signal
#    elif sample is "Signal":
#        for sms in SAMPLES['Signal']:
#            SAMPLES['Signal'][sms]['feature'] = {}
#            _file = rt.TFile.Open(SAMPLES['Signal'][sms]['file'])
#            assert(_file)
#            _tree = _file.Get('InclusiveSignalRegion')
#            
#            for feature in features:
#                SAMPLES['Signal'][sms]['feature'][feature] = SAMPLES[SAMPLES.keys()[0]]['feature'][feature].Clone()
#                SAMPLES['Signal'][sms]['feature'][feature].SetNameTitle(sms+"_"+feature, sms)
#                SAMPLES['Signal'][sms]['feature'][feature].SetDirectory(rt.gROOT)
#
#                draw_plot(SAMPLES['Signal'][sms]['feature'][feature], _tree, feature, sms)
#                save_plot(SAMPLES['Signal'][sms]['feature'][feature], _tree, feature, sms)
#            
#            for feature in features:
#                _thishist = SAMPLES['Signal'][sms]['feature'][feature]
#                if _thishist.Integral() == 0:
#                    print "Redone {}_{} for a stupid pyROOT pointer/ownership problem that does not fill in the first histogram".format(sms,feature)
#                    draw_plot(_thishist, _tree, feature, sms) 
#                    save_plot(_thishist, _tree, feature, sms)
#

STACKS = {}
NORM_STACKS = {}
for feature in SAMPLES['WJets']['feature']:
    STACKS[feature] = rt.THStack(feature+"_stack","")
    NORM_STACKS[feature] = rt.THStack(feature+"_normstack","")

savefile = rt.TFile.Open(SAVEFILE,"read")
for sample in SAMPLES:
    for feature in SAMPLES['WJets']['feature']:
        if 'T2qq' not in sample: 
            _myHist = savefile.Get(sample+"_"+feature)
            _myHist.SetLineColor(COLORS[sample])
            _myHist.SetFillColor(COLORS[sample])
            _myHist.SetMarkerColor(COLORS[sample])
            STACKS[feature].Add(_myHist)

for sample in SAMPLES:
    for feature in SAMPLES['WJets']['feature']:
        if 'T2qq' not in sample: 
            _myHist = savefile.Get(sample+"_"+feature)
            _myHist.Scale(1./sum_stack(STACKS[feature]))
            _myHist.SetLineColor(COLORS[sample])
            _myHist.SetFillColor(COLORS[sample])
            _myHist.SetMarkerColor(COLORS[sample])
            NORM_STACKS[feature].Add(_myHist)

for feature in SAMPLES['WJets']['feature']:
    cv = rt.TCanvas("cv","",700,600)
    NORM_STACKS[feature].Draw("hist")
    maxy = []
    maxy.append(NORM_STACKS[feature].GetMaximum())
    for i,sms in enumerate([key for key in SAMPLES.keys() if 'T2qq' in key]):
        _myHist = savefile.Get(sms+"_"+feature)
        _myHist.SetLineWidth(2)
        _myHist.SetLineStyle(i+2)
        _myHist.SetLineColor(i+1)
        #_myHist.SetMarkerStyle(rt.kFullCircle)
        _myHist.Scale(1./_myHist.Integral())
        _myHist.Draw("SAME HIST")
        maxy.append(_myHist.GetMaximum())
    NORM_STACKS[feature].SetMaximum(max(maxy)*1.2)
    leg = cv.BuildLegend(0.5,0.7,0.85,0.88)
    leg.SetNColumns(2)
    if not os.path.isdir(SAVEDIR): os.makedirs(SAVEDIR)
    cv.SaveAs(SAVEDIR+"/_"+feature+".png")

