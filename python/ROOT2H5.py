import ROOT as rt
from root_numpy import root2array, tree2array
import os
import argparse
import numpy as np
import h5py
import numpy.lib.recfunctions as nlr
from normalizeFastsimSMS import makeFileLists

SIGNAL_DIR = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_13Oct2017_Inclusive/SignalFastsim/weighted/'
BACKGROUND_DIR = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_13Oct2017_Inclusive/Signal/'
SAVEDIR = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_13Oct2017_Inclusive/h5/'
if not os.path.isdir(SAVEDIR): os.makedirs(SAVEDIR)

CUT = 'leadingJetPt>100 && MET>100 && MHT>100 && (box==21 || box==22)'

def makeFileLists(inDir, smsName, OneDScan=False):
    """
    inDir: directory to search 
    smsName: name of signal model
    OneDScan: parse only gluino mass, not LSP mass, from filename
    
    Returns: dictionary in which keys are (mGluino, mLSP) pairs
        and values are lists of ntuple files for the corresponding mass point
    """
    inFiles = os.listdir(inDir)

    #build dict of files associated with the different signal mass points
    fileLists = {}
    for f in inFiles:

        #skip files not corresponding to selected smsName
        if smsName not in f:
            continue

        #parse filename to get gluino and LSP masses
        if '.root' not in f: 
            print "Skipping non-ROOT file/directory",f
            continue
        splitF = f.replace('.root','').split('_')
        #check sanity
        if len(splitF) < 2:
            print "Unexpected file",f,": skipping"
            continue

        if not OneDScan:
            try:
                int(splitF[1])
                mGluino = splitF[1]
            except ValueError:
                print "Cannot parse gluino mass from",f,": skipping"
                continue

            try:
                int(splitF[2])
                mLSP = splitF[2]
            except ValueError:
                print "Cannot parse LSP mass from",f,": skipping"
                continue

            pair = (mGluino, mLSP)

            #add to dictionary if not present
            if pair not in fileLists:
                fileLists[pair] = []

            #add this file to appropriate list
            fileLists[pair].append(f)

        else:
            try:
                int(splitF[-1])
                mGluino = splitF[-1]
            except ValueError:
                print "Cannot parse gluino mass from",f,": skipping"
                continue

            if mGluino not in fileLists:
                fileLists[mGluino] = []

            #add this file to appropriate list
            fileLists[mGluino].append(f)
    
    return fileLists

SAMPLES = {}
SAMPLES['WJets'] = {'file': BACKGROUND_DIR+"InclusiveSignalRegion_Razor2016_MoriondRereco_WJets_1pb_weighted.root"}
SAMPLES['TTJets'] = {'file': BACKGROUND_DIR+"InclusiveSignalRegion_Razor2016_MoriondRereco_TTJetsHTBinned_1pb_weighted.root"}
SAMPLES['Other'] = {'file': BACKGROUND_DIR+"InclusiveSignalRegion_Razor2016_MoriondRereco_Other_1pb_weighted.root"}
SAMPLES['QCD'] = {'file': BACKGROUND_DIR+"InclusiveSignalRegion_Razor2016_MoriondRereco_QCD_1pb_weighted.root"}
SAMPLES['DYJets'] = {'file': BACKGROUND_DIR+"InclusiveSignalRegion_Razor2016_MoriondRereco_DYJets_1pb_weighted.root"}
SAMPLES['SingleTop'] = {'file': BACKGROUND_DIR+"InclusiveSignalRegion_Razor2016_MoriondRereco_SingleTop_1pb_weighted.root"}
SAMPLES['ZInv'] = {'file': BACKGROUND_DIR+"InclusiveSignalRegion_Razor2016_MoriondRereco_ZInv_1pb_weighted.root"}
SignalDict = makeFileLists(SIGNAL_DIR, 'T2qq')
for signal in SignalDict:
    SAMPLES['T2qq_{}_{}'.format(signal[0],signal[1])] = {'file': SIGNAL_DIR+SignalDict[signal][0]}

# Test files for quick and dirty check
SAMPLES['WJets']['test'] = BACKGROUND_DIR+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_WJetsToLNu_Pt-250To400_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8.Job0of13.root'
SAMPLES['TTJets']['test'] = BACKGROUND_DIR+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_TTJets_HT-600to800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job15of19.root'
SAMPLES['Other']['test'] = BACKGROUND_DIR+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_WWTo2L2Nu_13TeV-powheg.Job0of2.root'
SAMPLES['QCD']['test'] = BACKGROUND_DIR+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job49of61.root'
SAMPLES['DYJets']['test'] = BACKGROUND_DIR+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_DYJetsToLL_M-50_HT-200to400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job24of251.root'
SAMPLES['SingleTop']['test'] = BACKGROUND_DIR+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_ST_t-channel_antitop_4f_inclusiveDecays_13TeV-powhegV2-madspin-pythia8_TuneCUETP8M1.Job121of678.root'
SAMPLES['ZInv']['test'] = BACKGROUND_DIR+"/jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_ZJetsToNuNu_HT-200To400_13TeV-madgraph.Job201of512.root"
for signal in SignalDict:
    SAMPLES['T2qq_{}_{}'.format(signal[0],signal[1])]['test'] = SIGNAL_DIR+SignalDict[signal][0]

def convert(tree, sample=''):
    print("Transforming {} events from {}".format(tree.GetEntries(), sample))
    feature = tree2array(tree,
            branches = ['weight','alphaT','dPhiMinJetMET','dPhiRazor','HT','jet1MT','leadingJetCISV','leadingJetPt','MET','MHT','MR','MT2','nSelectedJets','Rsq','subleadingJetPt'],
            selection = CUT)
    if 'T2qq' in sample:
        label = np.ones(shape=(feature.shape), dtype = [('label','f4')])
        mSquark = int(sample.split('_')[1])
        mLSP = int(sample.split('_')[2])
        ms = np.full(shape=(feature.shape), fill_value=mSquark, dtype = [('mSquark','f4')])
        ml = np.full(shape=(feature.shape), fill_value=mLSP, dtype = [('mLSP','f4')])
    else:
        label = np.zeros(shape=(feature.shape), dtype = [('label','f4')])
        print ("Feature shape = {}".format(feature.shape))
        rand_ms = np.random.randint(0, 2000, size=(feature.shape))
        rand_ml = np.random.randint(0, 1000, size=(feature.shape))
        ms = np.zeros(shape=(feature.shape), dtype = [('mSquark','f4')])
        ms['mSquark'] = rand_ms
        ml = np.zeros(shape=(feature.shape), dtype = [('mLSP','f4')])
        ml['mLSP'] = rand_ml

    data = nlr.merge_arrays([label,feature,ms,ml], flatten=True) 
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

