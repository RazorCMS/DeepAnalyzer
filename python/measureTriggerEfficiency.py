import ROOT as rt
from argparse import ArgumentParser
import os
from utils.TriggerManager import TriggerManager

rt.gROOT.SetBatch()

filein = rt.TFile.Open('/eos/cms/store/group/phys_susy/razor/Run2Analysis/RunTwoInclusiveControlRegion/2016/V3p15_13Oct2017_Inclusive/OneLeptonFull/InclusiveControlRegion_OneLeptonFull_SingleLeptonSkim_Razor2016_MoriondRereco_Data_NoDuplicates_RazorSkim_GoodLumiGolden.root')
assert(filein)

mytree = filein.Get("InclusiveControlRegion")
assert(mytree)

def initHist(name, xlow, xhigh):
    return rt.TEfficiency(name,';Offline '+name,60,xlow,xhigh)

HistList = {}
HistList['alphaT'] = initHist('alphaT', 0, 1.2)
HistList['HT'] = initHist('HT', 0, 1400)
HistList['jet1MT'] = initHist('jet1MT', 0, 800)
HistList['leadingJetPt'] = initHist('leadingJetPt', 0, 200)
HistList['MET'] = initHist('MET', 0, 800)
HistList['MHT'] = initHist('MHT', 0, 800)
HistList['MR'] = initHist('MR', 0, 1000)
HistList['MT2'] = initHist('MT2', 0, 500)
HistList['Rsq'] = initHist('Rsq', 0, 1.0)
HistList['subleadingJetPt'] = initHist('subleadingJetPt', 0, 400)
    
DenominatorTrigger = TriggerManager('SingleLepton')
MT2Trigger = TriggerManager('MT2')
MHTTrigger = TriggerManager('MHT')
RazorTrigger = TriggerManager('Razor')

entries = mytree.GetEntries()
print ("NEntries = {}".format(entries))

for i in range(entries):
    mytree.GetEntry(i)
    if (i%10000==0): print ("Get entry {}/{}".format(i, entries))
    if mytree.MET > 100 and mytree.MHT > 100 and abs(mytree.leadingJetEta) < 2.5 and mytree.nBJetsMedium == 0: # Baseline selection
        cmd = 'if {}: '.format(DenominatorTrigger.appendTriggerCuts(treeName='mytree'))
        cmd += '\n    passSel = {} or {} or {}'.format(MT2Trigger.appendTriggerCuts(treeName='mytree'), MHTTrigger.appendTriggerCuts(treeName='mytree'), RazorTrigger.appendTriggerCuts(treeName='mytree'))
        for feature in list(HistList):
            cmd += '\n    HistList[\'{}\'].Fill(passSel, mytree.{})'.format(feature, feature)

        exec (cmd)

rt.gStyle.SetOptStat(0)   
c1 = rt.TCanvas("c1","",600,600)
if not os.path.isdir('TriggerPlots'): os.makedirs('TriggerPlots')
for feature in list(HistList):
    HistList[feature].Draw("AP")
    c1.SaveAs("TriggerPlots/{}.png".format(feature))
