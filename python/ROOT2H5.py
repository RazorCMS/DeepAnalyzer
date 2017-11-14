import ROOT as rt
from root_numpy import root2array, tree2array
import os
import argparse
import numpy as np
import h5py
import numpy.lib.recfunctions as nlr

SAVEDIR = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_13Oct2017_Inclusive/h5/'
if not os.path.isdir(SAVEDIR): os.makedirs(SAVEDIR)

CUT = 'leadingJetPt>100 && MET>100 && MHT>100 && (box==21 || box==22)'

SAMPLES = {}
filedir = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_13Oct2017_Inclusive/Signal/'
SAMPLES['WJets'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_WJets_1pb_weighted.root"}
SAMPLES['TTJets'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_TTJetsHTBinned_1pb_weighted.root"}
SAMPLES['Other'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_Other_1pb_weighted.root"}
SAMPLES['QCD'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_QCD_1pb_weighted.root"}
SAMPLES['DYJets'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_DYJets_1pb_weighted.root"}
SAMPLES['SingleTop'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_SingleTop_1pb_weighted.root"}
SAMPLES['ZInv'] = {'file': filedir+"InclusiveSignalRegion_Razor2016_MoriondRereco_ZInv_1pb_weighted.root"}
SAMPLES['T2qq_900_850'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_900_850_1pb_weighted.root"}
SAMPLES['T2qq_850_800'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_850_800_1pb_weighted.root"}
SAMPLES['T2qq_800_750'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_800_750_1pb_weighted.root"}
SAMPLES['T2qq_750_700'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_750_700_1pb_weighted.root"}
SAMPLES['T2qq_700_650'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_700_650_1pb_weighted.root"}
SAMPLES['T2qq_650_600'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_650_600_1pb_weighted.root"}
SAMPLES['T2qq_600_550'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_600_550_1pb_weighted.root"}
SAMPLES['T2qq_550_500'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_550_500_1pb_weighted.root"}
SAMPLES['T2qq_500_450'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_500_450_1pb_weighted.root"}
SAMPLES['T2qq_450_400'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_450_400_1pb_weighted.root"}

# Test files for quick and dirty check
SAMPLES['WJets']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_WJetsToLNu_Pt-250To400_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8.Job0of13.root'
SAMPLES['TTJets']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_TTJets_HT-600to800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job15of19.root'
SAMPLES['Other']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_WWTo2L2Nu_13TeV-powheg.Job0of2.root'
SAMPLES['QCD']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job49of61.root'
SAMPLES['DYJets']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_DYJetsToLL_M-50_HT-200to400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job24of251.root'
SAMPLES['SingleTop']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_ST_t-channel_antitop_4f_inclusiveDecays_13TeV-powhegV2-madspin-pythia8_TuneCUETP8M1.Job121of678.root'
SAMPLES['ZInv']['test'] = filedir+"/jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_ZJetsToNuNu_HT-200To400_13TeV-madgraph.Job201of512.root"
SAMPLES['T2qq_900_850']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_900_850_1pb_weighted.root"
SAMPLES['T2qq_850_800']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_850_800_1pb_weighted.root"
SAMPLES['T2qq_800_750']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_800_750_1pb_weighted.root"
SAMPLES['T2qq_750_700']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_750_700_1pb_weighted.root"
SAMPLES['T2qq_700_650']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_700_650_1pb_weighted.root"
SAMPLES['T2qq_650_600']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_650_600_1pb_weighted.root"
SAMPLES['T2qq_600_550']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_600_550_1pb_weighted.root"
SAMPLES['T2qq_550_500']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_550_500_1pb_weighted.root"
SAMPLES['T2qq_500_450']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_500_450_1pb_weighted.root"
SAMPLES['T2qq_450_400']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_450_400_1pb_weighted.root"

def convert(tree, sample=''):
    print("Transforming {} events from {}".format(tree.GetEntries(), sample))
    feature = tree2array(tree,
            branches = ['weight','alphaT','dPhiMinJetMET','dPhiRazor','HT','jet1MT','leadingJetCISV','leadingJetPt','MET','MHT','MR','MT2','nSelectedJets','Rsq','subleadingJetPt'],
            selection = CUT)
    if 'T2qq' in sample:
        label = np.ones(shape=(feature.shape), dtype = [('label','f4')])
    else:
        label = np.zeros(shape=(feature.shape), dtype = [('label','f4')])
    data = nlr.merge_arrays([label,feature], flatten=True) 
    print("{} selected events converted to h5py".format(data.shape[0]))
    return data

def saveh5(sample,loca):
    print(SAVEDIR+'/'+sample+'.h5')
    outh5 = h5py.File(SAVEDIR+'/'+sample+'.h5','w')
    _file = rt.TFile.Open(SAMPLES[sample][loca])
    _tree = _file.Get('InclusiveSignalRegion')
    outh5['Data'] = convert(_tree, sample)
    outh5.close()
    _file.Close()
    print("Save to {}".format(SAVEDIR+'/'+sample+'.h5'))

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-s','--sample', help='Sample to process (WJets, TTJets, Signal, etc.)', choices=['WJets','TTJets','Other','QCD','DYJets','SingleTop','ZInv','Signal'])
group.add_argument('-a','--all', action='store_true', help='Run all samples')
parser.add_argument('-t','--test', action='store_true', help='Run a very small test sample')

args = parser.parse_args()

if args.test: 
    print("Using small test samples")
    loca = 'test'
else:
    loca = 'file'

if args.all: # 
    print("Processing all files...")
    for sample in SAMPLES:
        saveh5(sample, loca)
elif "Signal" in args.sample:
    print("Processing Signal only...")
    for sample in SAMPLES:
        if "T2qq" in sample:
            saveh5(sample, loca)
else:
    print("Processing {}...".format(args.sample)) 
    saveh5(args.sample, loca)

