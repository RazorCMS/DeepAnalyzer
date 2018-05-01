#!/usr/bin/python3
import h5py
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.externals import joblib
import pickle 
import argparse
import os
import sys

plt.switch_backend('agg') # Non-interactive

# Convert to regular numpy arrays
def to_regular_array(struct_array):
    # There is an integer column (nSelectedJets) in the structured array. Need to convert to float before converting to regular array
    dt = struct_array.dtype.descr
    for i in range(len(dt)):
        if 'f4' not in dt[i][1]:
            dt[i] = (dt[i][0],np.float32)
    converted = np.array(struct_array, dtype=dt)
    return converted.view((np.float32, len(converted.dtype.names)))

def clean_dataset(arr):
    print ("Before cleaning: {}".format(arr.shape))
    df = pd.DataFrame(data=arr)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    cleaned = pd.DataFrame.as_matrix(df[indices_to_keep].astype(np.float32))
    print ("After cleaning: {}".format(cleaned.shape))
    return cleaned

def remove_outlier(arr):
    # Will remove outlier according to each feature.
    def clip(arr, lower, sub_lower, upper, sub_upper):
        print("{} arr = {}".format(type(arr),arr))
        print("{} lower = {}".format(type(lower),lower))
        arr[arr < lower] = sub_lower
        arr[arr > upper] = sub_upper
        return arr
    _arr = arr[:]
    appex = ''
    for key in _arr.dtype.names: 
        if 'NoW' in key: appex = '_NoW'
        if 'NoZ' in key: appex = '_NoZ'
    print("appex = {}".format(appex))
    _arr['alphaT'+appex] = clip(arr['alphaT'+appex], 0, 0, 100, 100)
    _arr['dPhiMinJetMET'] = clip(arr['dPhiMinJetMET'], -np.pi, 0, np.pi, 0)
    _arr['dPhiRazor'+appex] = clip(arr['dPhiRazor'+appex], -np.pi, 0, np.pi, 0)
    _arr['HT'+appex] = clip(arr['HT'+appex], 0, 0, 10000, 10000)
    _arr['jet1MT'] = clip(arr['jet1MT'], 0, 0, 10000, 10000)
    _arr['leadingJetCISV'] = clip(arr['leadingJetCISV'], 0, 0, 1, 1)
    _arr['leadingJetPt'] = clip(arr['leadingJetPt'], 0, 0, 5000, 5000)
    _arr['MET'+appex] = clip(arr['MET'+appex], 0, 0, 5000, 5000)
    _arr['MHT'] = clip(arr['MHT'], 0, 0, 5000, 5000)
    _arr['MR'+appex] = clip(arr['MR'+appex], 0, 0, 10000, 10000)
    _arr['MT2'+appex] = clip(arr['MT2'+appex], 0, 0, 5000, 5000)
    _arr['nSelectedJets' if appex=='' else 'nJets'+appex] = clip(arr['nSelectedJets' if appex=='' else 'nJets'+appex], 0, 0, 20, 20)
    _arr['Rsq'+appex] = clip(arr['Rsq'+appex], 0, 0, 5, 5)
    _arr['subleadingJetPt'] = clip(arr['subleadingJetPt'], 0, 0, 4000, 4000)

    return _arr


def multiply_data(data, multiplicity):
    print ("Dataset size before multiplicity: {}".format(data.shape[0]))
    for i in range(multiplicity):
        if i==0: sum_data = np.copy(data)
        else: sum_data = np.hstack((sum_data, data))
    print ("Dataset size after multiplicity: {}".format(sum_data.shape[0]))
    return sum_data

def has_nan(x, name=''):
    if np.isnan(x).any():
        print ("Warning: {} has nan.".format(name))
        return True
    return False

def load_dataset(location, load_type = 0, sample_size = 0, loadReal=False):
    loadfile = h5py.File(location,"r")
    assert(len(loadfile.keys())>0)
    to_load = 'Data'
    if loadReal: to_load='Data_Visible'
    try:
        dat = loadfile[to_load]
    except KeyError as e:
        print("Key error: {}".format(e))
    print("Cleaning dataset {}. LoadReal = {}".format(location,loadReal))
    _dat = clean_dataset(to_regular_array(remove_outlier(dat)))

    if sample_size==0:
        _x = _dat[:,2:]
        _y = _dat[:,0].astype(int)
        _weight = _dat[:,1]
    else:
        print ("Loading sample size {}".format(sample_size))
        _x = _dat[0:int(sample_size),2:]
        _y = _dat[0:int(sample_size),0]
        _weight = _dat[0:int(sample_size),1]
    has_nan(_x)
    has_nan(_y)
    has_nan(_weight)
    #_weight = np.clip(_weight, None, 1e3) # Remove crazy weights
    return _x, _y, _weight

def get_class_weight(label):
    unique, counts = np.unique(label, return_counts=True)
    occur = dict(zip(unique, counts)) # this hopefully returns {0: bkg, 1: sn}
    print ("occur: {}".format(occur))
    class_weight = {list(occur)[0]: 1, list(occur)[1]: occur[list(occur)[0]]/occur[list(occur)[1]]}
    print ("class_weight: {}".format(class_weight))
    return class_weight

def scale_fit(x_train, sample_size=0, scalerfile = None):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    if scalerfile == None: scaler_file = "Scaler/scaler_{}.pkl".format(sample_size)
    else: scaler_file = scalerfile
    joblib.dump(scaler, scaler_file)
    print ("Saving scaler information to {}".format(scaler_file))

def scale_dataset(x_train, sample_size=0, scalerfile=None):
    if scalerfile == None: scaler_file = "Scaler/scaler_{}.pkl".format(sample_size)
    else:
        scaler_file = scalerfile
    print("Loading scaler information from {}".format(scaler_file))
    scaler = joblib.load(scaler_file)
    _x_train = scaler.transform(x_train)
    return _x_train

def inverse_scale(x, sample_size = 0, scalerfile=None):
    if scalerfile == None: scaler_file = "Scaler/scaler_{}.pkl".format(sample_size)
    else:
        scaler_file = scalerfile
    print("Loading inverse scaler information from {}".format(scaler_file))
    scaler = joblib.load(scaler_file)
    inverse_x = scaler.inverse_transform(x)
    return inverse_x

def get_score(file_name='', title='', use_weight=False):
    valrel = h5py.File(file_name,"r")
    assert valrel
    valrel.keys()
    val_pred = valrel['Prediction'][:,1]
    val_truth = valrel['Truth'][:]
    val_weight = valrel['Weight'][:]
    val_sn = val_pred[np.where(val_truth>0.5)]
    val_bkg = val_pred[np.where(val_truth<0.5)]
    weight_sn = val_weight[np.where(val_truth>0.5)]
    weight_bkg = val_weight[np.where(val_truth<0.5)]
    print (weight_sn.shape[:10])
    print (weight_bkg.shape[:10])
    #plt.figure()
    if use_weight:
        n_sn, bins_sn, _ = plt.hist(val_sn, 
                                weights=weight_sn, 
                                bins=30, range=(0,1), histtype='step', color='r', label='Signal', normed=False)
        n_bkg, bins_bkg, _ = plt.hist(val_bkg, 
                                  weights=weight_bkg, 
                                  bins=30, range=(0,1), histtype='step', color='b', label='Background', normed=False)
    else:
        n_sn, bins_sn, _ = plt.hist(val_sn, bins=30, range=(0,1.), histtype='step', color='r', label='Signal', normed=False)
        n_bkg, bins_bkg, _ = plt.hist(val_bkg, bins=30, range=(0,1.), histtype='step', color='b', label='Background', normed=False)
    #plt.title('Neural network score for {}'.format(title))
    #plt.legend(loc='best')
    #plt.yscale('log')
    #plt.show()
    valrel.close()
    return bins_sn, n_sn, n_bkg

def select_mass_point(x_test, y_test, weight_test, mSquark, mLSP):
    #selected_signal = ((abs(x_test[:,-2] - mSquark) < 0.01) & (abs(x_test[:,-1] - mLSP) < 0.01) & (y_test > 0.5))
    #x_signal = x_test[selected_signal]
    #if len(x_signal) < 1: sys.exit("Required mass point not found in FastsimSMS.")
    #y_signal = y_test[selected_signal]
    #weight_signal = weight_test[selected_signal]
    
    x_background = x_test[y_test < 0.5]
    y_background = y_test[y_test < 0.5]
    weight_background = weight_test[y_test < 0.5]
    x_background[:,-2] = float(mSquark)
    x_background[:,-1] = float(mLSP)
    print ("y background shape = {}".format(y_background.shape))
    #x_fin = np.vstack((x_signal, x_background))
    #y_fin = np.concatenate((y_signal, y_background))
    #weight_fin = np.concatenate((weight_signal, weight_background))
    
    #return x_fin, y_fin, weight_fin
    return x_background, y_background, weight_background
            
def predict(sample, label='Default', real=False):
    #import ROOT as rt
    from glob import glob
    from array import array
    SIG_DIR = '/bigdata/shared/analysis/OR_CUT/MonoJet/'
    sample_size = 0 # for history purpose
    SIGNAL = [os.path.basename(x).replace('.h5','') for x in glob(SIG_DIR+'T2qq*')]
    DataLocation = DATA_DIR+"/"+sample
    CheckPoint = 'CheckPoint/CheckPoint{}_{}.h5'.format(sample_size,label)
    TrainLocation = "Result/TrainResult{size}_{label}.h5".format(size=sample_size, label=label)
    scaler_file = "Scaler/scaler_{}_{}.pkl".format(sample_size, label)
    #shape_file = "ShapeOutput/Score{size}_{label}_{btag}B_{sample}".format(size=sample_size, label=label, btag=args.btag, sample=sample) 
    score_file = "ScoredDataset/{}/{}_{}.h5".format(args.region, sample.replace('.h5',''),label) 

    from keras.models import load_model
    print ("Loading the model checkpoint: {}".format(CheckPoint))
    model = load_model(CheckPoint)
    _x_test, _y_test, _weight_test = load_dataset(DataLocation,2,sample_size = sample_size)
    if real: # predict the fully visible system for 1LInv and 2LInv too
        _x_visi, _y_visi, _weight_visi = load_dataset(DataLocation, 2, sample_size=sample_size, loadReal=True)


    for i,signal in enumerate(SIGNAL[-10:]):
        #if i < lower_test or i > upper_test: continue
        mSquark = int(signal.split('_')[1])
        mLSP = int(signal.split('_')[2])
        print("{}/{} Evaluating mSquark = {}, mLSP = {}".format(i, len(SIGNAL)-1, mSquark, mLSP))
    
        x_test, y_test, weight_test = select_mass_point(_x_test, _y_test, _weight_test, mSquark, mLSP)
        x_test = scale_dataset(x_test, sample_size, scaler_file)
        test_pred = model.predict(x_test)
        if real: 
            x_visi, y_visi, weight_visi = select_mass_point(_x_visi, _y_visi, _weight_visi, mSquark, mLSP)
            x_visi = scale_dataset(x_visi, sample_size, scaler_file)
            visi_pred = model.predict(x_visi)

        print("test shape {}".format(test_pred.shape))
        print("weight shape {}".format(weight_test.shape))
        if real:
            print("visible test shape {}".format(visi_pred.shape))
            print("visible weight shape {}".format(weight_visi.shape))

        feature_names = ['alphaT','dPhiMinJetMET','dPhiRazor','HT','jet1MT','leadingJetCISV', 
                    'leadingJetPt', 'MET', 'MHT', 'MR', 'MT2', 'nSelectedJets', 'Rsq', 'subleadingJetPt']
        feature_ascii = [n.encode("ascii", "ignore") for n in feature_names]
        
        if os.path.isfile(score_file):
            with h5py.File(score_file,'a') as out:
                out.create_dataset('Score/{}_{}'.format(int(mSquark), int(mLSP)), data=test_pred[:,1])
                if real:
                    out.create_dataset('VisibleScore/{}_{}'.format(int(mSquark), int(mLSP)), data=visi_pred[:,1])
        else:
            with h5py.File(score_file,'a') as out:
                out.create_dataset('Feature', data=inverse_scale(x_test, sample_size, scaler_file)[:,:-2])
                out.create_dataset('Feature_Name', (len(feature_ascii),1), 'S11', feature_ascii)
                out.create_dataset('Weight', data=weight_test)
                out.create_dataset('Score/{}_{}'.format(int(mSquark), int(mLSP)), data=test_pred[:,1])
                if real:
                    out.create_dataset('VisibleScore/{}_{}'.format(int(mSquark), int(mLSP)), data=visi_pred[:,1])

        print("Save prediction score to {}".format(score_file))              


#        n_sn, bins_sn, _ = plt.hist(test_pred[:,1], 
#                                weights=weight_test, 
#                                bins=30, range=(0,1), histtype='step', color='r', label='Signal', normed=False)
#        
#        out = rt.TFile(shape_file.replace("h5","root","a"))
#        hist = rt.TH1F("{}_{}".format(int(mSquark),int(mLSP)), "", len(bins_sn), 0,1)
#        for i in range(len(bins_sn)):
#            hist.SetBinContent(i+1, n_sn[i])
#        hist.Write()
#        out.Write()
#        out.Close()
        #shape_file = shape_file.replace("h5","hdf5")
#        if os.path.isfile(shape_file):
#            print("Appending {}_{} to {}".format(int(mSquark), int(mLSP), shape_file))
#            out = h5py.File(shape_file,'a')
#            #out.create_dataset("Bins", data=bins_sn)
#            out.create_dataset("{}_{}".format(int(mSquark), int(mLSP)), data=n_sn)
#            out.close()
#        else:
#            print("Creating {}_{} in {}".format(int(mSquark), int(mLSP), shape_file))
#            out = h5py.File(shape_file,"w")
#            out.create_dataset("Bins", data=bins_sn)
#            out.create_dataset("{}_{}".format(int(mSquark), int(mLSP)), data=n_sn)
#            out.close()
#
#        print("Save shape output to {}".format(shape_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--all', action='store_true', help='Test on validation set, evaluate at all mass points')
    parser.add_argument('--mSquark', help='Squark mass for parameterized testing', default=0)
    parser.add_argument('--mLSP', help='LSP mass for parameterized testing', default=0)
    parser.add_argument('-s','--sample', default=None, help='Sample to predict')
    parser.add_argument('-d','--device', default="5", help='GPU device to use')
    parser.add_argument('-nw','--noweight', action='store_true', help='Not use sample weights')
    parser.add_argument('-l','--label', default='', help='Label for benchmark study')
    parser.add_argument('--box',type=int, default=0, help='1: Monojet. Else: Multijet.')
    parser.add_argument('--region',default='1L0B', help='Sample to predict. Choices: 1L0B, 1L1B, 1LInv, 2LInv')

    args = parser.parse_args()
    
    DATA_DIR = '/bigdata/shared/analysis/H5CR/OR_CUT/OR_CUT/'
    if args.region == '1L1B': DATA_DIR += '1L1B/'
    elif args.region == '1L0B': DATA_DIR += '1L0B/'
    elif args.region == '1LInv': DATA_DIR += '1LInv/'
    elif args.region == '2LInv': DATA_DIR += '2LInv/'

    FEATURES = ['alphaT', 'dPhiMinJetMET', 'dPhiRazor', 'HT', 'jet1MT', 'leadingJetCISV', 'leadingJetPt', 'MET', 'MHT', 'MR', 'MT2', 'nSelectedJets', 'Rsq', 'subleadingJetPt']

    #DATA_DIR = '/home/ubuntu/data/'

    if args.box == 1:
        print("Using monojet box")
        DATA_DIR += '/MonoJet/'
    elif args.box == 2:
        DATA_DIR += '/DiJet/'
        print("Using dijet box")
    elif args.box == 4 or args.box==5 or args.box == 6:
        DATA_DIR += '/FourJet/'
        print("Using fourjet box")
    elif args.box == 7:
        DATA_DIR += '/SevenJet/'
        print("Using sevenjet box")
    else:
        DATA_DIR += '/MultiJet/'
        print("Using multijet box")

    if args.noweight: print ("Not using sample weight")
    print ("Using GPU(s):",args.device)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    # if 1LInv or 2LInv, get the prediction score of real system (1L not added to MET) too.
    if '1LInv' in args.region or '2LInv' in args.region: loadReal = True
    else: loadReal = False
    SAMPLES = ['Data','DYJets','Other','QCD','SingleTop','TTJets','WJets','ZInv']
    if args.region=='2LInv': 
        SAMPLES.remove('QCD')
        SAMPLES.remove('ZInv')
    if args.sample == None:
        for sample in SAMPLES:
            predict(sample+'.h5', args.label, real=loadReal)
    else:
        predict(args.sample+'.h5', args.label, real=loadReal)
