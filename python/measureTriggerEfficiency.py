import ROOT as rt
from argparse import ArgumentParser
import os
from utils.TriggerManager import TriggerManager
import time

start_time = time.time()

rt.gROOT.SetBatch()

filein = rt.TFile.Open('/eos/cms/store/group/phys_susy/razor/Run2Analysis/RunTwoInclusiveControlRegion/2016/V3p15_13Oct2017_Inclusive/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_Data_NoDuplicates_RazorSkim_GoodLumiGolden.root')
assert(filein)

mytree = filein.Get("InclusiveControlRegion")
assert(mytree)

def initHist(name, xlow, xhigh):
    return rt.TEfficiency(name,';Offline '+name,25,xlow,xhigh)

HistList = {}
HistList['alphaT'] = initHist('alphaT', 0, 1.2)
HistList['dPhiMinJetMET'] = initHist('dPhiMinJetMET', 0, 3.2)
HistList['dPhiRazor'] = initHist('dPhiRazor', 0, 3.2)
HistList['HT'] = initHist('HT', 0, 1400)
HistList['jet1MT'] = initHist('jet1MT', 0, 800)
HistList['leadingJetPt'] = initHist('leadingJetPt', 0, 800)
HistList['MET'] = initHist('MET', 0, 800)
HistList['MHT'] = initHist('MHT', 0, 800)
HistList['MR'] = initHist('MR', 0, 1000)
HistList['MT2'] = initHist('MT2', 0, 800)
HistList['Rsq'] = initHist('Rsq', 0, 1.0)
HistList['subleadingJetPt'] = initHist('subleadingJetPt', 0, 600)

Threshold = {}
Threshold['alphaT'] = 0.
Threshold['dPhiMinJetMET'] = 0.
Threshold['dPhiRazor'] = 0.
Threshold['HT'] = 100.
Threshold['jet1MT'] = 100
Threshold['leadingJetPt'] = 100.
Threshold['MET'] = 200
Threshold['MHT'] = 200
Threshold['MR'] = 300
Threshold['MT2'] = 200
Threshold['Rsq'] = 0.2
Threshold['subleadingJetPt'] = 80.

DenominatorTrigger = TriggerManager('SingleLepton')
MT2Trigger = TriggerManager('MT2')
MHTTrigger = TriggerManager('MHT')
RazorTrigger = TriggerManager('Razor')

entries = mytree.GetEntries()
print ("NEntries = {}".format(entries))
entries /= 2 # For fast testing
print ("Running on {} events".format(entries))

for i in range(entries):
    mytree.GetEntry(i)
    if (i%10000==0): 
        print ("Get entry {0}/{1} ({2:.2f}% -- Time used: {3:.1f}s)".format(i, entries, float(i)/float(entries)*100, time.time()-start_time))

    if abs(mytree.leadingJetEta) < 2.5 and mytree.nBJetsMedium == 0: # Baseline selection
        cmd = 'if {}: '.format(DenominatorTrigger.appendTriggerCuts(treeName='mytree'))
        for feature in list(HistList):
            cmd += '\n    if '
            first = True
            for threshold in list(Threshold):
                if feature != threshold:
                    if not first: cmd += ' and '
                    cmd += 'mytree.{} > {}'.format(threshold, Threshold[threshold]) 
                    first = False
            cmd += ':\n         passSel = {} or {} or {}'.format(MT2Trigger.appendTriggerCuts(treeName='mytree'), MHTTrigger.appendTriggerCuts(treeName='mytree'), RazorTrigger.appendTriggerCuts(treeName='mytree'))
            cmd += '\n         HistList[\'{}\'].Fill(passSel, mytree.{})'.format(feature, feature)

        exec (cmd)
rt.gStyle.SetOptStat(0)   
c1 = rt.TCanvas("c1","",600,600)
SaveDir = 'TriggerPlots'
if not os.path.isdir(SaveDir): os.makedirs(SaveDir)
for feature in list(HistList):
    HistList[feature].Draw("AP")
    c1.SaveAs("{}/{}.png".format(SaveDir,feature))
