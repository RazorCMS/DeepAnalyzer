import ROOT as rt
import h5py
import numpy as np
import os
import argparse

# Make a numpy array containing 14 Features
# Features (in order): alphaT, dPhiMinJetMET, dPhiRazor, HT, jet1MT, leadingJetCISV, leadingJetPt, MET, MHT, MR, MT2, nSelectedJets, Rsq, subleadingJetPt

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
SAMPLES['T2qq_900_850'] = {'file': filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_900_850.root"}

# Test files for quick and dirty check
SAMPLES['WJets']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_WJetsToLNu_Pt-250To400_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8.Job0of13.root'
SAMPLES['TTJets']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_TTJets_HT-600to800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job15of19.root'
SAMPLES['Other']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_WWTo2L2Nu_13TeV-powheg.Job0of2.root'
SAMPLES['QCD']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job49of61.root'
SAMPLES['DYJets']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_DYJetsToLL_M-50_HT-200to400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.Job24of251.root'
SAMPLES['SingleTop']['test'] = filedir+'jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_ST_t-channel_antitop_4f_inclusiveDecays_13TeV-powhegV2-madspin-pythia8_TuneCUETP8M1.Job121of678.root'
SAMPLES['ZInv']['test'] = filedir+"/jobs/InclusiveSignalRegion_Razor2016_MoriondRereco_ZJetsToNuNu_HT-200To400_13TeV-madgraph.Job201of512.root"
SAMPLES['T2qq_900_850']['test'] = filedir.replace("Signal/","SignalFastsim/")+"SMS-T2qq_900_850.root"


# Save everything into 1 file
#all_h5 = h5py.File(SAVEDIR+'/SignalInclusive.h5','w')
#all_stack_feature = np.empty(shape=14, dtype=np.float32)
#all_stack_feature[:] = np.NAN
#all_stack_weight = np.empty(shape=1, dtype=np.float32)
#all_stack_label = np.empty(shape=1, dtype=np.int8)

parser = argparse.ArgumentParser()
parser.add_argument('sample', help='Sample to process (WJets, TTJets, etc.)', choices=['WJets','TTJets','Other','QCD','DYJets','SingleTop','ZInv','T2qq_900_850'])
parser.add_argument('-t','--test', action='store_true', help='Run a very small test sample')
#parser.add_argument('-a','--all', help='Run all samples')

args = parser.parse_args()

if args.test: 
    loca = 'test'
else:
    loca = 'file'

#for sample in SAMPLES:
sample = args.sample
outh5 = h5py.File(SAVEDIR+'/'+sample+'.h5','w')

stack_feature = np.empty(shape=14, dtype=np.float32)
stack_feature[:] = np.NAN
stack_weight = np.empty(shape=1, dtype=np.float32)
stack_weight[:] = np.NAN
_file = rt.TFile.Open(SAMPLES[sample][loca])
_tree = _file.Get('InclusiveSignalRegion')

NEntries = _tree.GetEntries()
print "Begin processing {} entries in {}".format(NEntries, sample)

_tree.Draw('>>elist', CUT, 'entrylist')
elist = rt.gDirectory.Get('elist')
NPass = elist.GetN()
print "Total entries passing cuts: {}".format(NPass)

count = 0
while True:
    entry = elist.Next()
    if entry == -1: break
    if count > 0 and count % 10000 == 0: print "Processing entry {}/{}".format(count, NPass)
    _tree.GetEntry(entry)
    count += 1

    # Features (in order): alphaT, dPhiMinJetMET, dPhiRazor, HT, jet1MT, leadingJetCISV, leadingJetPt, MET, MHT, MR, MT2, nSelectedJets, Rsq, subleadingJetPt
    feature = np.empty(shape=14, dtype=np.float32)
    feature[0] = _tree.alphaT
    feature[1] = _tree.dPhiMinJetMET
    feature[2] = _tree.dPhiRazor
    feature[3] = _tree.HT
    feature[4] = _tree.jet1MT
    feature[5] = _tree.leadingJetCISV
    feature[6] = _tree.leadingJetPt
    feature[7] = _tree.MET
    feature[8] = _tree.MHT
    feature[9] = _tree.MR
    feature[10] = _tree.MT2
    feature[11] = _tree.nSelectedJets
    feature[12] = _tree.Rsq
    feature[13] = _tree.subleadingJetPt

    weight = np.array([_tree.weight])
    if np.isnan(stack_feature).any():
        stack_feature = np.copy(feature)
        stack_weight = np.copy(weight)
    else:
        stack_feature = np.vstack((stack_feature, feature))
        stack_weight = np.vstack((stack_weight, weight))

outh5['Feature'] = stack_feature
if 'T2qq' not in sample: 
    stack_label = np.zeros(shape=(stack_feature.shape[0],1))
else: 
    stack_label = np.ones(shape=(stack_feature.shape[0],1))
outh5['Label'] = stack_label
outh5['Weight'] = stack_weight

print "Save to {}".format(SAVEDIR+'/'+sample+'.h5')

## Save to the inclusive file
#if np.isnan(all_stack_feature).any():
#    all_stack_feature = np.copy(stack_feature)
#    all_stack_weight = np.copy(stack_weight)
#    all_stack_label = np.copy(stack_label)
#else:
#    all_stack_feature = np.vstack((all_stack_feature, stack_feature))
#    all_stack_weight = np.vstack((all_stack_weight, stack_weight))
#    all_stack_label = np.vstack((all_stack_label, stack_label))
#

_file.Close()
outh5.close()
del stack_feature, outh5, _tree, feature, stack_weight, weight        

#all_h5.create_dataset('Feature', data = all_stack_feature, compression='gzip')
#all_h5.create_dataset('Weight', data = all_stack_weight, compression='gzip')
#all_h5.create_dataset('Label', data = all_stack_label, dtype='int8')
#all_h5.close()
#print "Save to {}".format(SAVEDIR+'/SignalInclusive.h5')
