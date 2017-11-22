import sys
import os
import argparse
import ROOT as rt

from RunCombine import exec_me


SIGNAL_DIR = '/eos/cms/store/group/phys_susy/razor/Run2Analysis/InclusiveSignalRegion/2016/V3p15_13Oct2017_Inclusive/SignalFastsim/'

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
        print splitF
        #check sanity
        if len(splitF) < 2:
            print "Unexpected file",f,": skipping"
            continue

        if not OneDScan:
            try:
                int(splitF[-2])
                mGluino = splitF[-2]
            except ValueError:
                print "Cannot parse gluino mass from",f,": skipping"
                continue

            try:
                int(splitF[-1])
                mLSP = splitF[-1]
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

def getTheoryCrossSectionAndError(mGluino=-1, mStop=-1, mSquark=-1):
    thyXsec = -1
    thyXsecErr = -1

    if mGluino!=-1:
        for line in open('../data/gluino13TeV.txt','r'):
            line = line.replace('\n','')
            if str(int(mGluino))==line.split(',')[0]:
                thyXsec = float(line.split(',')[1]) #pb
                thyXsecErr = 0.01*float(line.split(',')[2])
    if mStop!=-1:
        for line in open('../data/stop13TeV.txt','r'):
            line = line.replace('\n','')
            if str(int(mStop))==line.split(',')[0]:
                thyXsec = float(line.split(',')[1]) #pb
                thyXsecErr = 0.01*float(line.split(',')[2]) 
    if mSquark!=-1:
        for line in open('../data/squark13TeV.txt','r'):
            line = line.replace('\n','')
            if str(int(mSquark))==line.split(',')[0]:
                thyXsec = float(line.split(',')[1]) #pb
                thyXsecErr = 0.01*float(line.split(',')[2]) 

    return thyXsec,thyXsecErr

def NormalizeSMS():
    fileList = makeFileLists(SIGNAL_DIR, 'T2qq', OneDScan=False)
    local_dir = os.environ['CMSSW_BASE']+'/src/DeepAnalyzer/'
    
    tempXS = local_dir+'_tempXS.dat'
    tempList = local_dir+'_tempList.txt'
    mainXS = local_dir+'data/xSections.dat'
    placeholder = local_dir+'tmp.tmp'

    with open(tempXS,'w') as out:
        for pair in list(fileList.keys()):
            SMSName = fileList[pair][0].replace('.root','')
            xs, _ = getTheoryCrossSectionAndError(mSquark=pair[0])
            out.write("{}\t{}\n".format(SMSName, xs))
    print("Write temporary cross section file to {}".format(tempXS))

    with open(tempList,'w') as out:
        for pair in list(fileList.keys()):
            SMSName = fileList[pair][0].replace('.root','')
            SMSLocation = SIGNAL_DIR+fileList[pair][0] 
            out.write("{}\t{}\n".format(SMSName, SMSLocation))
    print("Write temporary list of files to normalize to {}".format(tempList))

    # The NormalizeNtuple source code, written in C++, is a hassle to change.
    # Will replace the newly created tempXS file with the main xSection.dat, run the normalization, and swap back
    from shutil import copyfile
    copyfile(mainXS, placeholder)
    copyfile(tempXS, mainXS)

    os.chdir(local_dir)
    from subprocess import call
    call(['./NormalizeNtuple',tempList])

    copyfile(placeholder, mainXS)
    os.remove(tempXS)
    os.remove(tempList)
    os.remove(placeholder)
    print("Removing {}".format(tempXS))
    print("Removing {}".format(tempList))
    print("Removing {}".format(placeholder))

if __name__ == "__main__":
    NormalizeSMS()
