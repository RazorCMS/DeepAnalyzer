#!/bin/env python

### Submit jobs using NtupleUtils.py, then check jobs' statuses every 30 seconds. 
### When jobs are done, automatically proceed to the next step in the sequence 
### MC: hadd --> normalize --> hadd-final 
### Data: hadd --> skim --> hadd-final --> remove-duplicates --> good-lumi

import os, sys, subprocess
import argparse
import time
import smtplib
from email.mime.text import MIMEText

def check_bjobs(job_name):
# Get the number of running and pending jobs, only proceed if both = 0
    finish = False
    jstatus = subprocess.Popen(["bjobs","-sum","-noheader","-J",job_name], stdout=subprocess.PIPE).communicate()[0]
    jlist = [ x for x in jstatus.split(" ") if x != '' ]
    rjobs = int(jlist[0])
    pjobs = int(jlist[4])
    print "Checking job status: ", job_name
    print "- Running jobs: ", rjobs
    print "- Pending jobs: ", pjobs
    if rjobs == 0 and pjobs == 0: finish = True
    return finish

def send_email(label, tag, data, email, finished=True):
    me = 'Dead Dustin <dustin1111111@gmail.com>'
    msg = MIMEText('.')
    if finished:
        msg['Subject'] = 'Sequence '+label+' '+tag+' '+str(data)+' is finished' 
    else:
        msg['Subject'] = 'Sequence '+label+' '+tag+' '+str(data)+' has zombies. Restarting.' 
    msg['From'] = me
    msg['To'] = email
    s = smtplib.SMTP('localhost')
    s.sendmail(me, email, msg.as_string())
    s.quit()


def sub_sequence(tag, isData=False, submit=False, label='', skipSub=False, force=False, email='', fastSim=False, noZombies=False):
    basedir = os.environ['CMSSW_BASE']+'/src/RazorAnalyzer'
    if not submit: 
        nosub = '--no-sub'
    else:
        nosub = ''
    if isData:
        data = '--data'
    else:
        data = ''
    if force:
        force = '--force'
    else:
        force = ''
    if fastSim:
        fastsim = '--fastsim'
    else:
        fastsim = ''

    if not submit and not skipSub: # --no-sub, only execute the submit command
        cmd_submit = list(filter(None,['python', 'python/ntupling/NtupleUtils.py', tag, '--submit', nosub, '--label', label, data, fastsim]))
        print ' '.join(cmd_submit)
        subprocess.call(cmd_submit)
    
    if submit: 
        if not skipSub: # execute the whole sequence
            cmd_submit = list(filter(None,['python', 'python/ntupling/NtupleUtils.py', tag, '--submit', nosub, '--label', label, data, fastsim]))
            print ' '.join(cmd_submit)
            subprocess.call(cmd_submit)
            job_done = False
            while not job_done:
                time.sleep(30)
                job_done = check_bjobs('*'+label+'*')

        # Before running hadd, unless --no-zombies is specified, we have to check that there are no zombie files in the output.
        # If there are, remove the zombies and restart the process.
        if not noZombies:
            cmd_zombies = list(filter(None,['python', 'python/ntupling/NtupleUtils.py', tag, '--find-zombies', nosub, '--label', label, data, fastsim]))
            print ' '.join(cmd_zombies)
            subprocess.call(cmd_zombies)
            zombieFileName = "Zombies_%s_%s.txt"%(tag, label)
            if isData:
                zombieFileName = zombieFileName.replace(".txt","_Data.txt")
            with open(zombieFileName) as zombieFile:
                if len(list(zombieFile)) > 0:
                    for line in zombieFile:
                        print "Removing {}".format(line)
                        line = line.replace('\n','')
                        os.remove(line)
                    sub_sequence(tag, isData, submit, label, skipSub, email, fastSim)

                    if (email is not None): send_email(label, tag, data, email, finished=False)
                    
        # If skipSub and noZombies are specified, start the sequence at hadd
        cmd_hadd = list(filter(None,['python', 'python/ntupling/NtupleUtils.py', tag, '--hadd', nosub, '--label', label, data, force, fastsim]))
        print ' '.join(cmd_hadd)
        subprocess.call(cmd_hadd)

        if fastSim: sys.exit("No normalization for fastsim at the moment!") # stop the script here

        if not isData:
            cmd_normalize = list(filter(None,['python', 'python/ntupling/NtupleUtils.py', tag, '--normalize', nosub, '--label', label, data, force]))
            print ' '.join(cmd_normalize)
            subprocess.call(cmd_normalize)
        else:
            cmd_skim = list(filter(None,['python', 'python/ntupling/NtupleUtils.py', tag, '--skim', nosub, '--label', label, data, force]))
            print ' '.join(cmd_skim)
            subprocess.call(cmd_skim)
        cmd_hadd_final = list(filter(None,['python', 'python/ntupling/NtupleUtils.py', tag, '--hadd-final', nosub, '--label', label, data, force]))
        print ' '.join(cmd_hadd_final)
        subprocess.call(cmd_hadd_final)
        if isData:
            cmd_remove_duplicates = list(filter(None,['python', 'python/ntupling/NtupleUtils.py', '--remove-duplicates', nosub, '--label', label, data, tag]))
            print ' '.join(cmd_remove_duplicates)
            subprocess.call(cmd_remove_duplicates)

            cmd_good_lumi = list(filter(None,['python', 'python/ntupling/NtupleUtils.py', '--good-lumi', nosub, '--label', label, data, tag]))
            print ' '.join(cmd_good_lumi)
            subprocess.call(cmd_good_lumi)
        
        if (email is not None): send_email(label, tag, data, email, finished=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', help = '1L, 2L, ...', required = True)
    parser.add_argument('--label', help = 'Label for RazorRun', default='Razor2016_MoriondRereco', default='Razor2016_MoriondRereco')
    parser.add_argument('--data', action = 'store_true', help = 'Run on data (MC otherwise)')
    parser.add_argument('--force', action = 'store_true', help = 'Force action if file exists')
    parser.add_argument('--no-sub', dest = 'noSub', action = 'store_true', help = 'Print commands but do not submit')
    parser.add_argument('--email', help = 'Send email notification after sequence is finished')
    parser.add_argument('--skip-sub', dest = 'skipSub', action = 'store_true', help = 'Start the sequence at the hadd step')
    parser.add_argument('--fastsim', action = 'store_true', help = 'Action for fastsim')
    parser.add_argument('--no-zombies', action = 'store_true', dest='noZombies', help = 'We know for sure there is no zombies. Skip --find-zombies step')


    args = parser.parse_args()
    
    sub_sequence(args.tag, args.data, (not args.noSub), args.label, args.skipSub, args.force, args.email, args.fastsim, args.noZombies)
