import ROOT as rt
from root_numpy import root2array, tree2array
import os
import argparse
import numpy as np
import h5py
import numpy.lib.recfunctions as nlr
from normalizeFastsimSMS import makeFileLists
from utils.TriggerManager import TriggerManager

SIGNAL_DIR = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_13Oct2017_Inclusive/SignalFastsim/weighted/'
BACKGROUND_DIR = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_13Oct2017_Inclusive/Signal/'

# box: MultiJet = 21, MonoJet = 22
MT2Trigger = TriggerManager('MT2')
MHTTrigger = TriggerManager('MHT')
RazorTrigger = TriggerManager('Razor')
TriggerCut = MT2Trigger.appendTriggerCuts() + ' || ' + MHTTrigger.appendTriggerCuts() + ' || ' + RazorTrigger.appendTriggerCuts()
BASELINE_MT2 = '(((HT > 1000 && MET > 30) || (HT > 250 && MET > 250)) && (( MT2 > 200 || (MT2 > 400 && HT > 1500)) && nSelectedJets > 1))'
BASELINE_MHT = '(MHT > 250 && HT > 300)'
BASELINE_RAZOR = '((MR > 650 && Rsq > 0.3) || (MR > 1600 && Rsq > 0.2))'
#BASELINE_CUT = 'leadingJetPt>100 && HT > 100 && jet1MT > 100 && MET > 200 && MHT > 200 && MR > 200 && MT2 > 200 && Rsq > 0.2 &&' + TriggerCut
BASELINE_CUT = 'leadingJetPt>100 &&' + TriggerCut + '&& (' + BASELINE_MT2 + '||' + BASELINE_MHT + '||' + BASELINE_RAZOR + ')'
CUT_MONOJET = BASELINE_CUT + ' && box==22 && (subleadingJetPt<60 || nSelectedJets == 1)'
CUT_DIJET = BASELINE_CUT + ' && box==21 && subleadingJetPt>60 && nSelectedJets > 1 && nSelectedJets < 4'
CUT_FOURJET = BASELINE_CUT + ' && box==21 && subleadingJetPt>60 && nSelectedJets > 3 && nSelectedJets < 7'
CUT_SEVENJET = BASELINE_CUT + ' && box==21 && subleadingJetPt>60 && nSelectedJets > 6'
CUT_MULTIJET = BASELINE_CUT + ' && box ==21 && subleadingJetPt>60'

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

def convert(tree, sample='', box=1, start=None, stop=None):
    if box==1:
        print("Using monojet box with selection: {}".format(CUT_MONOJET))
        CUT = CUT_MONOJET
    elif box == 2 or box == 3:
        print("Using dijet box with selection: {}".format(CUT_DIJET))
        CUT = CUT_DIJET
    elif box == 4 or box == 5 or box == 6:
        print("Using fourjet box with selection: {}".format(CUT_FOURJET))
        CUT = CUT_FOURJET
    elif box == 7:
        print("Using sevenjet box with selection: {}".format(CUT_SEVENJET))
        CUT = CUT_SEVENJET
    else:
        print("Using multijet box with selection: {}".format(CUT_MULTIJET))
        CUT = CUT_MULTIJET
    #NEvents = tree.GetEntries()
    print("Transforming {} events from {}".format(tree.GetEntries(), sample))
    feature = tree2array(tree,
                branches = ['weight','alphaT','dPhiMinJetMET','dPhiRazor','HT','jet1MT','leadingJetCISV','leadingJetPt','MET','MHT','MR','MT2','nSelectedJets','Rsq','subleadingJetPt'],
                selection = CUT, start=start, stop=stop)
    if 'T2qq' in sample:
        label = np.ones(shape=(feature.shape), dtype = [('label','f4')])
        mSquark = int(sample.split('_')[1])
        mLSP = int(sample.split('_')[2])
        ms = np.full(shape=(feature.shape), fill_value=mSquark, dtype = [('mSquark','f4')])
        ml = np.full(shape=(feature.shape), fill_value=mLSP, dtype = [('mLSP','f4')])
    else:
        label = np.zeros(shape=(feature.shape), dtype = [('label','f4')])
        print ("Feature shape = {}".format(feature.shape))
        ms = np.zeros(shape=(feature.shape), dtype = [('mSquark','f4')])
        ms['mSquark'] = np.random.randint(300, 1800, size=(feature.shape))
        ml = np.zeros(shape=(feature.shape), dtype = [('mLSP','f4')])
        ml['mLSP'] = np.random.randint(0, 1400, size=(feature.shape))
       
    data = nlr.merge_arrays([label,feature,ms,ml], flatten=True) 
    print("{} selected events converted to h5py".format(data.shape[0]))
    return data

def saveh5(sample,loca,box=1):
    #SAVEDIR = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_13Oct2017_Inclusive/h5/'
    SAVEDIR = '/eos/cms/store/group/dpg_hcal/comm_hcal/qnguyen/H5/OR_CUT/'
    if box == 1:
        SAVEDIR += '/MonoJet/'
    elif box == 2 or box == 3:
        SAVEDIR += '/DiJet/'
    elif box == 4 or box == 5 or box == 6:
        SAVEDIR += '/FourJet/'
    elif box == 7:
        SAVEDIR += '/SevenJet/'
    else:
        SAVEDIR += '/MultiJet/'
    if not os.path.isdir(SAVEDIR):
        os.makedirs(SAVEDIR)

    print(SAVEDIR+'/'+sample+'.h5')
    if os.path.isfile(SAVEDIR+'/'+sample+'.h5'):
        print ("Remove old file")
        os.remove(SAVEDIR+'/'+sample+'.h5')
    _file = rt.TFile.Open(SAMPLES[sample][loca])
    _tree = _file.Get('InclusiveSignalRegion')
    NEvents = _tree.GetEntries()
    segments = int(NEvents / 1e7)
    if segments < 2:
        outh5 = h5py.File(SAVEDIR+'/'+sample+'.h5','w')
        outh5['Data'] = convert(_tree, sample, box)
        outh5.close()
        _file.Close()
        print("Save to {}".format(SAVEDIR+'/'+sample+'.h5'))
    else:
        for i in range(segments):
            print("Converting {}/{}".format(i, segments))
            outname = SAVEDIR+'/'+sample+'_'+str(i)+'.h5'
            start = int(i*1e7)
            stop = int((i+1)*1e7)
            if stop > NEvents: stop = NEvents
            outh5 = h5py.File(outname,'w')
            outh5['Data'] = convert(_tree, sample, box, start, stop)
            outh5.close()
            print("Save to {}".format(outname))
        _file.Close()

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-s','--sample', help='Sample to process (WJets, TTJets, Signal, etc.)', choices=['WJets','TTJets','Other','QCD','DYJets','SingleTop','ZInv','Signal'])
group.add_argument('-a','--all', action='store_true', help='Run all samples')
group.add_argument('-b','--background', action='store_true', help='Run all background')
parser.add_argument('-t','--test', action='store_true', help='Run a very small test sample')
parser.add_argument('--box', type=int, default=1, help='1: Monojet box. Else: Multijet box.')


args = parser.parse_args()

if args.test: 
    print("Using small test samples")
    loca = 'test'
else:
    loca = 'file'

if args.all: # 
    print("Processing all files...")
    for sample in SAMPLES:
        saveh5(sample, loca, args.box)
elif args.background:
    print("Processing all backgrounds...")
    for sample in SAMPLES:
        if "T2qq" not in sample:
            saveh5(sample, loca, args.box)
elif "Signal" in args.sample:
    print("Processing Signal only...")
    for sample in SAMPLES:
        if "T2qq" in sample:
            saveh5(sample, loca, args.box)
else:
    print("Processing {}...".format(args.sample)) 
    saveh5(args.sample, loca, args.box)

