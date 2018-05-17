import ROOT as rt
from root_numpy import root2array, tree2array
import os, sys
import argparse
import numpy as np
import h5py
import numpy.lib.recfunctions as nlr
from normalizeFastsimSMS import makeFileLists
from utils.TriggerManager import TriggerManager

SIGNAL_DIR = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_13Oct2017_Inclusive/SignalFastsim/weighted/'
BACKGROUND_DIR = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_13Oct2017_Inclusive/Signal/'
CONTROL_DIR = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/RunTwoInclusiveControlRegion/2016/V3p15_13Oct2017_Inclusive/'

# box: MultiJet = 21, MonoJet = 22
MT2Trigger = TriggerManager('MT2')
MHTTrigger = TriggerManager('MHT')
RazorTrigger = TriggerManager('Razor')
SingleLeptonTrigger = TriggerManager('SingleLepton')

CRTriggerCut = SingleLeptonTrigger.appendTriggerCuts()
SignalTriggerCut = MT2Trigger.appendTriggerCuts() + ' || ' + MHTTrigger.appendTriggerCuts() + ' || ' + RazorTrigger.appendTriggerCuts()
BASELINE_MT2 = '(((HT > 1000 && MET > 30) || (HT > 250 && MET > 250)) && (( MT2 > 200 || (MT2 > 400 && HT > 1500)) && nSelectedJets > 1))'
BASELINE_MHT = '(MHT > 250 && HT > 300)'
BASELINE_RAZOR = '((MR > 650 && Rsq > 0.3) || (MR > 1600 && Rsq > 0.2))'

LOOSE_RAZOR = '((MR > 300 && Rsq > 0.15) || (MR > 1200 && Rsq > 0.1))'
LOOSE_MHT = '(MHT > 150 && HT > 200)'
LOOSE_MT2 = '(((HT > 800 && MET > 30) || (HT > 150 && MET > 150)) && (( MT2 > 100 || (MT2 > 300 && HT > 1200)) && nSelectedJets > 1))'

#BASELINE_CUT = 'leadingJetPt>100 && HT > 100 && jet1MT > 100 && MET > 200 && MHT > 200 && MR > 200 && MT2 > 200 && Rsq > 0.2 &&' + TriggerCut
BASELINE_CUT = '(' + BASELINE_MT2 + '||' + BASELINE_MHT + '||' + BASELINE_RAZOR + ')'
LOOSE_CUT = '(' + LOOSE_MT2 + '||' + LOOSE_MHT + '||' + LOOSE_RAZOR + ')'
CUT_MONOJET =  ' (subleadingJetPt<60 || nSelectedJets == 1)' # && box==22 
CUT_DIJET =  ' box==21 && subleadingJetPt>60 && nSelectedJets > 1 && nSelectedJets < 4'
CUT_FOURJET =  ' box==21 && subleadingJetPt>60 && nSelectedJets > 3 && nSelectedJets < 7'
CUT_SEVENJET =  ' box==21 && subleadingJetPt>60 && nSelectedJets > 6'
CUT_MULTIJET =  ' subleadingJetPt>60' #&& box ==21 

def xstr(s):
    if s is None:
        return ''
    return str(s)

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
SAMPLES['Data'] = {'file': ''}
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
SAMPLES['Data']['test'] = ''
for signal in SignalDict:
    SAMPLES['T2qq_{}_{}'.format(signal[0],signal[1])]['test'] = SIGNAL_DIR+SignalDict[signal][0]

# Control region 1L0B
SAMPLES['WJets']['1L0B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_WJets_1pb_weighted.root'
SAMPLES['TTJets']['1L0B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_TTJetsHTBinned_1pb_weighted.root'
SAMPLES['Other']['1L0B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_Other_1pb_weighted.root'
SAMPLES['QCD']['1L0B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_QCD_1pb_weighted.root'
SAMPLES['DYJets']['1L0B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_DYJets_1pb_weighted.root'
SAMPLES['SingleTop']['1L0B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_SingleTop_1pb_weighted.root'
SAMPLES['ZInv']['1L0B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_ZInv_1pb_weighted.root'
SAMPLES['Data']['1L0B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_Data_NoDuplicates_RazorSkim_GoodLumiGolden.root'

# Control region 1L1B
SAMPLES['WJets']['1L1B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_WJets_1pb_weighted.root'
SAMPLES['TTJets']['1L1B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_TTJetsHTBinned_1pb_weighted.root'
SAMPLES['Other']['1L1B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_Other_1pb_weighted.root'
SAMPLES['QCD']['1L1B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_QCD_1pb_weighted.root'
SAMPLES['DYJets']['1L1B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_DYJets_1pb_weighted.root'
SAMPLES['SingleTop']['1L1B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_SingleTop_1pb_weighted.root'
SAMPLES['ZInv']['1L1B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_ZInv_1pb_weighted.root'
SAMPLES['Data']['1L1B'] = CONTROL_DIR+'/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_Data_NoDuplicates_RazorSkim_GoodLumiGolden.root'

# Control region 1LInv
SAMPLES['WJets']['1LInv'] = CONTROL_DIR+'/OneLeptonAddToMET/InclusiveControlRegion_OneLeptonAddToMetFull_SingleLeptonSkim_Razor2016_MoriondRereco_WJets_1pb_weighted.root'
SAMPLES['TTJets']['1LInv'] = CONTROL_DIR+'/OneLeptonAddToMET/InclusiveControlRegion_OneLeptonAddToMetFull_SingleLeptonSkim_Razor2016_MoriondRereco_TTJetsHTBinned_1pb_weighted.root'
SAMPLES['Other']['1LInv'] = CONTROL_DIR+'/OneLeptonAddToMET/InclusiveControlRegion_OneLeptonAddToMetFull_SingleLeptonSkim_Razor2016_MoriondRereco_Other_1pb_weighted.root'
SAMPLES['QCD']['1LInv'] = CONTROL_DIR+'/OneLeptonAddToMET/InclusiveControlRegion_OneLeptonAddToMetFull_SingleLeptonSkim_Razor2016_MoriondRereco_QCD_1pb_weighted.root'
SAMPLES['DYJets']['1LInv'] = CONTROL_DIR+'/OneLeptonAddToMET/InclusiveControlRegion_OneLeptonAddToMetFull_SingleLeptonSkim_Razor2016_MoriondRereco_DYJets_1pb_weighted.root'
SAMPLES['SingleTop']['1LInv'] = CONTROL_DIR+'/OneLeptonAddToMET/InclusiveControlRegion_OneLeptonAddToMetFull_SingleLeptonSkim_Razor2016_MoriondRereco_SingleTop_1pb_weighted.root'
SAMPLES['ZInv']['1LInv'] = CONTROL_DIR+'/OneLeptonAddToMET/InclusiveControlRegion_OneLeptonAddToMetFull_SingleLeptonSkim_Razor2016_MoriondRereco_ZInv_1pb_weighted.root'
SAMPLES['Data']['1LInv'] = CONTROL_DIR+'/OneLeptonAddToMET/InclusiveControlRegion_OneLeptonAddToMetFull_SingleLeptonSkim_Razor2016_MoriondRereco_Data_NoDuplicates_RazorSkim_GoodLumiGolden.root'

# Control region 2LInv
SAMPLES['WJets']['2LInv'] = CONTROL_DIR+'/DileptonAddToMET/InclusiveControlRegion_DileptonAddToMetFull_DileptonSkim_Razor2016_MoriondRereco_WJets_1pb_weighted.root'
SAMPLES['TTJets']['2LInv'] = CONTROL_DIR+'/DileptonAddToMET/InclusiveControlRegion_DileptonAddToMetFull_DileptonSkim_Razor2016_MoriondRereco_TTJetsHTBinned_1pb_weighted.root'
SAMPLES['Other']['2LInv'] = CONTROL_DIR+'/DileptonAddToMET/InclusiveControlRegion_DileptonAddToMetFull_DileptonSkim_Razor2016_MoriondRereco_Other_1pb_weighted.root'
SAMPLES['QCD']['2LInv'] = CONTROL_DIR+'/DileptonAddToMET/InclusiveControlRegion_DileptonAddToMetFull_DileptonSkim_Razor2016_MoriondRereco_QCD_1pb_weighted.root'
SAMPLES['DYJets']['2LInv'] = CONTROL_DIR+'/DileptonAddToMET/InclusiveControlRegion_DileptonAddToMetFull_DileptonSkim_Razor2016_MoriondRereco_DYJets_1pb_weighted.root'
SAMPLES['SingleTop']['2LInv'] = CONTROL_DIR+'/DileptonAddToMET/InclusiveControlRegion_DileptonAddToMetFull_DileptonSkim_Razor2016_MoriondRereco_SingleTop_1pb_weighted.root'
SAMPLES['ZInv']['2LInv'] = CONTROL_DIR+'/DileptonAddToMET/InclusiveControlRegion_DileptonAddToMetFull_DileptonSkim_Razor2016_MoriondRereco_ZInv_1pb_weighted.root'
SAMPLES['Data']['2LInv'] = CONTROL_DIR+'/DileptonAddToMET/InclusiveControlRegion_DileptonAddToMetFull_DileptonSkim_Razor2016_MoriondRereco_Data_NoDuplicates_RazorSkim_GoodLumiGolden.root'



def convert(tree, sample='', box=1, start=None, stop=None, cr=None, saveReal=False, separatePhi=True):
    if box==1:
        print("Using monojet box")
        CUT = CUT_MONOJET
    elif box == 2 or box == 3:
        print("Using dijet box")
        CUT = CUT_DIJET
    elif box == 4 or box == 5 or box == 6:
        print("Using fourjet box")
        CUT = CUT_FOURJET
    elif box == 7:
        print("Using sevenjet box")
        CUT = CUT_SEVENJET
    else:
        print("Using multijet box")
        CUT = CUT_MULTIJET
    #NEvents = tree.GetEntries()
    if 'T2qq' not in sample:
        if cr == None: CUT += ' && ' + SignalTriggerCut + ' && ' + BASELINE_CUT
        else: CUT += ' && ' + CRTriggerCut + ' && ' + BASELINE_CUT
    else: #T2qq doesn't have trigger
        CUT += ' && ' + BASELINE_CUT
    print("Transforming {} events from {}".format(tree.GetEntries(), sample))
    if cr == '1L0B':
        CUT += ' && lep1MT < 100 && nBJetsMedium==0 && lep1MT > 30 && MET > 30'
        print("Using selection: {}".format(CUT))
    
    elif cr == '1L1B':
        CUT += ' && lep1MT < 100 && nBJetsMedium>0 && lep1MT > 30 && MET > 30'
        print("Using selection: {}".format(CUT))
    
    feature = tree2array(tree,
                branches = ['weight','alphaT','dPhiMinJetMET','dPhiRazor','HT','jet1MT','leadingJetCISV','leadingJetPt','MET','MHT','MR','MT2','nSelectedJets','Rsq','subleadingJetPt'],
                selection = CUT, start=start, stop=stop)

    if cr == '1LInv':
        CUT = CUT.replace("MET","MET_NoW").replace(" HT"," HT_NoW").replace("(HT","(HT_NoW").replace("nSelectedJets","nJets_NoW").replace("MR","MR_NoW").replace("Rsq","Rsq_NoW").replace("MT2","MT2_NoW").replace("alphaT","alphaT_NoW")
        CUT += ' && lep1MT < 100 && nBJetsMedium == 0 && lep1MT > 30 && MET_NoW > 30'
        print("Using selection: {}".format(CUT))
        if not saveReal:
            feature = tree2array(tree,
                branches = ['weight','alphaT_NoW','dPhiMinJetMET','dPhiRazor_NoW','HT_NoW','jet1MT','leadingJetCISV','leadingJetPt','MET_NoW','MHT','MR_NoW','MT2_NoW','nJets_NoW','Rsq_NoW','subleadingJetPt'],
                selection = CUT, start=start, stop=stop)
        else:
            feature = tree2array(tree,
                branches = ['weight','alphaT','dPhiMinJetMET','dPhiRazor','HT','jet1MT','leadingJetCISV','leadingJetPt','MET','MHT','MR','MT2','nSelectedJets','Rsq','subleadingJetPt'],
                selection = CUT, start=start, stop=stop)

    elif cr == '2LInv':
        CUT = CUT.replace("MET","MET_NoZ").replace(" HT"," HT_NoZ").replace("(HT","(HT_NoZ").replace("nSelectedJets","nJets_NoZ").replace("MR","MR_NoZ").replace("Rsq","Rsq_NoZ").replace("MT2","MT2_NoZ").replace("alphaT","alphaT_NoZ")

        CUT += ' && lep1MT < 100 && nBJetsMedium == 0 && lep1MT > 30 && MET_NoZ > 30'
        print("Using selection: {}".format(CUT))
        if not saveReal:
            feature = tree2array(tree,
                branches = ['weight','alphaT_NoZ','dPhiMinJetMET','dPhiRazor_NoZ','HT_NoZ','jet1MT','leadingJetCISV','leadingJetPt','MET_NoZ','MHT','MR_NoZ','MT2_NoZ','nJets_NoZ','Rsq_NoZ','subleadingJetPt'],
                selection = CUT, start=start, stop=stop)
        else:
            feature = tree2array(tree,
                branches = ['weight','alphaT','dPhiMinJetMET','dPhiRazor','HT','jet1MT','leadingJetCISV','leadingJetPt','MET','MHT','MR','MT2','nSelectedJets','Rsq','subleadingJetPt'],
                selection = CUT, start=start, stop=stop)

#    else:
#        feature = tree2array(tree,
#                branches = ['weight','alphaT','dPhiMinJetMET','dPhiRazor','HT','jet1MT','leadingJetCISV','leadingJetPt','MET','MHT','MR','MT2','nSelectedJets','Rsq','subleadingJetPt'],
#                selection = CUT, start=start, stop=stop)
    if separatePhi:
        feature = tree2array(tree,
                branches = ['weight','alphaT','HT','jet1MT','leadingJetCISV','leadingJetPt','MET','MHT','MR','MT2','nSelectedJets','Rsq','subleadingJetPt'],
                selection = CUT, start=start, stop=stop)
        separate = tree2array(tree, branches = ['dPhiMinJetMET','dPhiRazor'], selection=CUT, start=start, stop=stop)

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
    if separatePhi:
        return data, separate
    else:
        return data

def saveh5(sample,loca,box=1, cr = None):
    #SAVEDIR = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_13Oct2017_Inclusive/h5/'
    #SAVEDIR = '/eos/cms/store/group/dpg_hcal/comm_hcal/qnguyen/OR_CUT/'
    #SAVEDIR = '/eos/cms/store/group/dpg_hcal/comm_hcal/qnguyen/H5CR/OR_CUT/1BTagged/'
    #SAVEDIR = '/eos/cms/store/group/dpg_hcal/comm_hcal/qnguyen/H5CR/OR_CUT/'
    #SAVEDIR = '/eos/cms/store/group/dpg_hcal/comm_hcal/qnguyen/H5SR/OR_CUT/'
    if cr == None:
        SAVEDIR = '/afs/cern.ch/work/q/qnguyen/public/DMAnalysis/CMSSW_9_2_7/src/DeepAnalyzer/python/H5SR/'
    else:
        SAVEDIR = '/afs/cern.ch/work/q/qnguyen/public/DMAnalysis/CMSSW_9_2_7/src/DeepAnalyzer/python/H5CR/'
    if cr != None: SAVEDIR+=cr
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
    _treeName = 'InclusiveSignalRegion'
    if loca in ['1L0B','1L1B','2L','1LInv','2LInv','Photon','VetoL','VetoTau']: _treeName = "InclusiveControlRegion"
    try:
        _tree = _file.Get(_treeName)
        NEvents = _tree.GetEntries()
    except:
        print("Problematic file: {}".format(SAMPLES[sample][loca]))
        print("trying to read tree: {}".format(_treeName))
        raise
    segments = int(NEvents / 1e7)
    if segments < 2:
        try:
            if os.path.isfile(SAVEDIR+'/'+sample+'.h5'):
                print ("Remove old file")
                os.remove(SAVEDIR+'/'+sample+'.h5')
            outh5 = h5py.File(SAVEDIR+'/'+sample+'.h5','w')
        except IOError as e:
            print(e, SAVEDIR+'/'+sample+'.h5')
            raise

        outh5['Data'], outh5['Phi'] = convert(_tree, sample, box, cr=cr, saveReal=False, separatePhi=True)
        if 'Inv' in xstr(cr): 
            outh5['Data_Visible'],_ = convert(_tree, sample, box, cr=cr, saveReal=True, separatePhi=True)
        outh5.close()
        _file.Close()
        print("Save to {}".format(SAVEDIR+'/'+sample+'.h5'))
    else:
        for i in range(segments):
            print("Converting {}/{}".format(i, segments))
            outname = SAVEDIR+'/'+sample+'_'+str(i)+'.h5'
            if os.path.isfile(outname):
                print ("Remove old file")
                os.remove(outname)
            start = int(i*1e7)
            stop = int((i+1)*1e7)
            if stop > NEvents: stop = NEvents
            outh5 = h5py.File(outname,'w')
            outh5['Data'], outh5['Phi'] = convert(_tree, sample, box, start, stop, cr = cr, saveReal=False, separatePhi=True)
            if 'Inv' in xstr(cr):
                outh5['Data_Visible'], _ = convert(_tree, sample, box, start, stop, cr = cr, saveReal=True, separatePhi=True)

            outh5.close()
            print("Save to {}".format(outname))
        _file.Close()

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-s','--sample', help='Sample to process (WJets, TTJets, Signal, etc.)', choices=['WJets','TTJets','Other','QCD','DYJets','SingleTop','ZInv','Signal','Data'])
group.add_argument('-a','--all', action='store_true', help='Run all samples')
group.add_argument('-b','--background', action='store_true', help='Run all background')
parser.add_argument('--region', help="Has to be one of ['Signal','1L0B','1L1B','1LInv','2L','2LInv','Photon','VetoL','VetoTau']")
parser.add_argument('-t','--test', action='store_true', help='Run a very small test sample')
parser.add_argument('--box', type=int, default=0, help='1: Monojet box. Else: Multijet box.')


args = parser.parse_args()

cr = None

if args.test: 
    print("Using small test samples from SR")
    loca = 'test'

elif args.region == 'Signal':
    loca = 'file'
elif args.region not in ['1L0B','1L1B','1LInv','2L','2LInv','Photon','VetoL','VetoTau']: 
    sys.exit("Please use a proper region name: has to be 1 of ['Signal','1L0B','1L1B','1LInv','2L','2LInv','Photon','VetoL','VetoTau']")
else: # 1 of Control Region 
    print("Processing {} region".format(args.region))
    loca = args.region
    cr = args.region

if args.background:
    print("Processing all backgrounds...")
    for sample in SAMPLES:
        if "T2qq" not in sample:
            saveh5(sample, loca, args.box, cr=cr)
elif args.all: 
    print("Processing all samples...")
    for sample in SAMPLES:
        saveh5(sample, loca, args.box, cr=cr)

elif args.sample:
    print("Processing {}".format(args.sample))
    if "Signal" in args.sample:
        for sample in SAMPLES:
            if "T2qq" in sample:
                saveh5(sample, loca, args.box, cr=cr)
    else:
        saveh5(args.sample, loca, args.box, cr=cr)
