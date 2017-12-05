import os
from TriggerDict import TriggerNames

class TriggerManager(object):
    """Stores and retrieves trigger names"""
    def __init__(self, trigType, tag, isData=False):
        self.trigType = trigType
        self.tag = tag
        if isData:
            self.isDataTag = "Data"
        else:
            self.isDataTag = "MC"

        # get trigger path name/number correspondence
        self.triggerDict = {}
        if tag == 'Razor2015':
            trigFile = os.environ['CMSSW_BASE']+'src/DeepAnalyzer/data/RazorHLTPathnames_2015.dat'
        else:
            trigFile = os.environ['CMSSW_BASE']+'src/DeepAnalyzer/data/RazorHLTPathnames_2016.dat'
        with open(trigFile) as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                self.triggerDict[row[-1]] = int(row[0]) 

        # get trigger names
        if self.trigType in triggerNames:
            self.names = triggerNames[self.trigType]
        else:
            self.badInit(trigType,tag)

        # get trigger numbers
        self.getTriggerNums()

    def badInit(self, trigType, tag):
        print ("Error in triggerUtils: trigger type/tag combination '",trigType,tag,"' not recognized!")
        self.names = None

    def getTriggerNums(self):
        self.nums = []
        for name in self.names:
            if name == 'PASS': # indicates that no trigger cuts should be applied
                self.nums = [-1] 
                return
            elif name in self.triggerDict:
                self.nums.append( self.triggerDict[name] )
            else:
                print "Warning in triggerUtils.getTriggerNums: trigger name",name,"not found!"

    def appendTriggerCuts(self, cutsString):
        """Append a string of the form "(HLTDecision[t1] || HLTDecision[t2] || ... || HLTDecision[tN]) && " 
           to the provided cut string, where t1...tN are the desired trigger numbers"""
        if -1 in self.nums: # trigger requirement not applied
            return cutsString
        return '('+(' || '.join(['HLTDecision['+str(n)+']' for n in self.nums]))+") && "+cutsString

