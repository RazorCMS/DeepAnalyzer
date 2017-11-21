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

DATA_DIR = '/bigdata/shared/analysis/'
FEATURES = ['alphaT', 'dPhiMinJetMET', 'dPhiRazor', 'HT', 'jet1MT', 'leadingJetCISV', 'leadingJetPt', 'MET', 'MHT', 'MR', 'MT2', 'nSelectedJets', 'Rsq', 'subleadingJetPt']

#DATA_DIR = '/home/ubuntu/data/'

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

def remove_outlier(arr, remove=None):
    # Will remove outlier according to each feature.
    # In order: 'alphaT','dPhiMinJetMET','dPhiRazor','HT','jet1MT','leadingJetCISV','leadingJetPt','MET','MHT','MR','MT2','nSelectedJets','Rsq','subleadingJetPt'
    print ("Removing outlier")
    if remove == None: 
        index_to_remove=None
    else: index_to_remove = FEATURES.index(remove)

    # alphaT
    arr[arr[:,0] < 0] = 0
    arr[arr[:,0] > 10] = 10
    # dPhiMinJetMET
    arr[arr[:,1] < -np.pi] = 0
    arr[arr[:,1] > np.pi] = 0 # unphysical
    # dPhiRazor
    arr[arr[:,2] < -np.pi] = 0 # unphysical
    arr[arr[:,2] > np.pi] = 0
    # HT
    arr[arr[:,3] < 0] = 0
    arr[arr[:,3] > 3000] = 3000
    # jet1MT
    arr[arr[:,4] < 0] = 0
    arr[arr[:,4] > 3000] = 3000
    # leadingJetCISV
    arr[arr[:,5] < 0] = 0
    arr[arr[:,5] > 1.] = 1.
    # leadingJetPT
    arr[arr[:,6] < 0] = 0
    arr[arr[:,6] > 2000] = 2000
    # MET 
    arr[arr[:,7] < 0] = 0
    arr[arr[:,7] > 5000] = 5000
    # MHT
    arr[arr[:,8] < 0] = 0
    arr[arr[:,8] > 2000] = 2000
    # MR
    arr[arr[:,9] < 0] = 0
    arr[arr[:,9] > 5000] = 5000
    # MT2
    arr[arr[:,10] < 0] = 0
    arr[arr[:,10] > 5000] = 5000
    # nSelectedJet
    arr[arr[:,11] < 0] = 0
    arr[arr[:,11] > 20] = 20
    # Rsq
    arr[arr[:,12] < 0] = 0
    arr[arr[:,12] > 2] = 2
    # subleadingJetPt
    arr[arr[:,13] < 0] = 0
    arr[arr[:,13] > 2000] = 2000

    return arr

def multiply_data(data, multiplicity):
    print ("Dataset size before multiplicity: {}".format(data.shape[0]))
    for i in range(multiplicity):
        if i==0: sum_data = np.copy(data)
        else: sum_data = np.hstack((sum_data, data))
    print ("Dataset size after multiplicity: {}".format(sum_data.shape[0]))
    return sum_data

def create_dataset():
    BACKGROUND = ['DYJets','Other','QCD','SingleTop','TTJets','WJets','ZInv']
    SIGNAL = ['T2qq_900_850']
    print ("Creating dataset...")
    for i,bkg in enumerate(BACKGROUND):
        _file = h5py.File(DATA_DIR+'/'+bkg+'.h5','r')
        if i == 0: Background = np.copy(_file['Data'])
        else: Background = np.hstack((Background, _file['Data']))
    Signal = h5py.File(DATA_DIR+SIGNAL[0]+'.h5','r')['Data'][:]

    print ("Background size: {}".format(Background.shape[0]))
    print ("Signal size: {}".format(Signal.shape[0]))

    Signal = multiply_data(Signal, 3483)

    # Get shuffled unified dataset for training
    Dataset = np.hstack((Background, Signal))
    np.random.shuffle(Dataset)

    # 60% training, 20% validation, 20% testing
    data_size = Dataset.shape[0]
    training_index = int(0.6*data_size)
    val_index = training_index + int(0.2*data_size)

    Dataset = to_regular_array(Dataset)
    Dataset = clean_dataset(Dataset)
    
    unique, counts = np.unique(Dataset[:,0], return_counts=True)
    occur = dict(zip(unique, counts)) # this hopefully returns {0: bkg, 1: sn}
    print ("Original label counts: {}".format(occur))


    # Save to files
    combine = h5py.File(DATA_DIR+"/CombinedDataset_Balanced.h5","w")
    combine['Training'] = Dataset[:training_index]
    combine['Validation'] = Dataset[training_index:val_index]
    combine['Test'] = Dataset[val_index:]
    print ("Save divided datasets to {}/CombinedDataset_Balanced.h5".format(DATA_DIR))
    combine.close()

def has_nan(x, name=''):
    if np.isnan(x).any():
        print ("Warning: {} has nan.".format(name))
        return True
    return False

def load_dataset(location, load_type = 0, sample_size = 0):
    loadfile = h5py.File(location,"r")
    assert(loadfile)

    def decode(load_type):
        if load_type == 0: return "Training"
        elif load_type == 1: return "Validation"
        else: return "Test"

    dat = loadfile[decode(load_type)]
    if sample_size==0:
        _x = dat[:,2:]
        _y = dat[:,0].astype(int)
        _weight = dat[:,1] * 1e5
    else:
        print ("Loading sample size {}".format(sample_size))
        _x = dat[0:int(sample_size),2:]
        _y = dat[0:int(sample_size),0]
        _weight = dat[0:int(sample_size),1] * 1e5
    has_nan(_x)
    has_nan(_y)
    has_nan(_weight)
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

def create_model(optimizer='adam', layers=3, init_size = 100, remove=None):
    from keras.models import Sequential
    from keras.layers import Input, Dense, Dropout
    
    # Training with a simple FFNN
    model = Sequential()
    for lay in range(layers):
        size = int(init_size/2**lay)
        if size < 5: break
        if lay==0: 
            if remove == None:
                model.add(Dense(size, input_shape=(14,), activation='relu'))
            else:
                model.add(Dense(size, input_shape=(13,), activation='relu'))
        else:
            model.add(Dropout(0.5))
            model.add(Dense(size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))

    model.summary()
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
    return model

def tuning(sample_size = 0):
    print ("Tuning the model")
    x_train, y_train, weight_train = load_dataset(DATA_DIR+"/Undersampling_Dataset_850_800.h5",0,sample_size = sample_size)
    x_val, y_val, weight_val = load_dataset(DATA_DIR+"/Undersampling_Dataset_850_800.h5",1,sample_size=sample_size)

    #x_train = remove_outlier(x_train)
    #x_val = remove_outlier(x_val)
    
    from keras.utils.np_utils import to_categorical
    y_train = to_categorical(y_train,2)
    has_nan(y_train,"training categorical label")
    y_val = to_categorical(y_val,2)
    has_nan(y_val,"validation categorical label")
    
    print ("Scaling features...")
    has_nan(x_train, "unscaled training")
    if not np.isfinite(x_train).all():
        print ("Unscaled training probably contains inf")
    scale_fit(x_train, sample_size)
    x_train = scale_dataset(x_train, sample_size)
    x_val = scale_dataset(x_val, sample_size)
    has_nan(x_train, "scaled training")
    has_nan(y_val, "scaled validation")
    
    class_weight = get_class_weight(y_train)
    
    from sklearn.model_selection import GridSearchCV
    from keras.wrappers.scikit_learn import KerasClassifier 
    
    # create model
    model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=1000, verbose=0)
    # define the grid search parameters
    init_size = [ 1300, 1500, 1800, 2500, 5000]
    layers = [2]
    param_grid = dict(init_size=init_size, layers=layers)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(x_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))


def training(sample_size = 0, not_use_weight=False, label='Default', remove=None):
    print ("Loading data...")
    if not os.path.isdir('CheckPoint/FeatureRemoval'):
        os.makedirs('CheckPoint/FeatureRemoval')
    if not os.path.isdir('ScaledInput/FeatureRemoval'):
        os.makedirs('ScaledInput/FeatureRemoval')
    if not os.path.isdir('History/FeatureRemoval'):
        os.makedirs('History/FeatureRemoval')
    if not os.path.isdir('Result/FeatureRemoval'):
        os.makedirs('Result/FeatureRemoval')
    if not os.path.isdir('Scaler/FeatureRemoval'):
        os.makedirs('Scaler/FeatureRemoval')

    if remove == None:
        DataLocation = DATA_DIR+"/Undersampling_Dataset_850_800.h5"
        ScaleInputTrain = "ScaledInput/TrainingDataset{}.h5".format(sample_size)
        ScaleInputVal = "ScaledInput/ValidationDataset{}.h5".format(sample_size)
        CheckPoint = 'CheckPoint/CheckPoint{}_{}.h5'.format(sample_size,label)
        histfile = 'History/history{}_{}.sav'.format(sample_size, label)
        ValidationLocation = "Result/ValidationResult{size}_{label}.h5".format(size=sample_size, label=label)
        scaler_file = "Scaler/scaler_{}.pkl".format(sample_size)
    else:
        print("{} removed".format(remove))
        DataLocation = DATA_DIR+"/FeatureRemoval/Undersampling_Dataset_No_{}.h5".format(remove)
        ScaleInputTrain = "ScaledInput/FeatureRemoval/TrainingDataset{}_No_{}.h5".format(sample_size,remove)
        ScaleInputVal = "ScaledInput/FeatureRemoval/ValidationDataset{}_No_{}.h5".format(sample_size,remove)
        CheckPoint = 'CheckPoint/FeatureRemoval/CheckPoint{}_{}_No_{}.h5'.format(sample_size,label,remove)
        histfile = 'History/FeatureRemoval/history{}_{}_No_{}.sav'.format(sample_size, label, remove)
        ValidationLocation = "Result/FeatureRemoval/ValidationResult{size}_{label}_No_{remove}.h5".format(size=sample_size, label=label, remove=remove)
        scaler_file = "Scaler/FeatureRemoval/scaler_{}_No_{}.pkl".format(sample_size, remove)
    
    
    x_train, y_train, weight_train = load_dataset(DataLocation,0,sample_size = sample_size)
    x_val, y_val, weight_val = load_dataset(DataLocation,1,sample_size=sample_size)
        
    #x_train = remove_outlier(x_train)
    #x_val = remove_outlier(x_val)
    
    from keras.utils.np_utils import to_categorical
    y_train = to_categorical(y_train,2)
    has_nan(y_train,"training categorical label")
    y_val = to_categorical(y_val,2)
    has_nan(y_val,"validation categorical label")
    
    print ("Scaling features...")
    has_nan(x_train, "unscaled training")
    if not np.isfinite(x_train).all():
        print ("Unscaled training probably contains inf")
    scale_fit(x_train, sample_size, scaler_file)
    x_train = scale_dataset(x_train, sample_size, scaler_file)
    x_val = scale_dataset(x_val, sample_size, scaler_file)
    has_nan(x_train, "scaled training")
    has_nan(y_val, "scaled validation")

    # Save scaled training input to file

    train_ds = h5py.File(ScaleInputTrain,"w")

    train_ds['x'] = x_train
    train_ds['y'] = y_train
    train_ds['w'] = weight_train
    train_ds.close()
    print ("Write to {}".format(ScaleInputTrain))
    
    # Save scaled val input to file
    val_ds = h5py.File(ScaleInputVal,"w")
    val_ds['x'] = x_val
    val_ds['y'] = y_val
    val_ds['w'] = weight_val
    val_ds.close()
    print ("Write to {}".format(ScaleInputVal))
    
    class_weight = get_class_weight(y_train)

    model = create_model(remove=remove)

    if (not_use_weight):
        val_tuple = (x_val, y_val)
        sample_weight = None
    else:
        val_tuple = (x_val, y_val, weight_val)
        sample_weight = weight_train
    
    from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
    hist = model.fit(x_train, y_train,
            validation_data = val_tuple,
            epochs = 500,
            batch_size = 50,
            class_weight = class_weight,
            sample_weight = sample_weight,
            callbacks = [ModelCheckpoint(filepath=CheckPoint, verbose = 1, 
                save_best_only=True), 
                ReduceLROnPlateau(patience = 4, factor = 0.5, verbose = 1, min_lr=1e-7), 
                EarlyStopping(patience = 20)
                ],
            )

    pickle.dump(hist.history, open(histfile,'wb'))
    print ("Save history to {}".format(histfile))

    val_pred = model.predict(x_val)
    val_label = np.argmax(y_val, axis=1)

    val_result = h5py.File(ValidationLocation,'w')
    val_result.create_dataset("Prediction", data = val_pred)
    val_result.create_dataset("Truth", data = val_label)
    val_result.create_dataset("Weight", data = weight_val)
    print ("Save result to {}".format(ValidationLocation))
    val_result.close()

def testing(sample_size = 0, label='Default', remove=None):
    if remove == None:
        DataLocation = DATA_DIR+"/Undersampling_Dataset_850_800.h5"
        ScaleInputTrain = "ScaledInput/TrainingDataset{}.h5".format(sample_size)
        ScaleInputVal = "ScaledInput/ValidationDataset{}.h5".format(sample_size)
        CheckPoint = 'CheckPoint/CheckPoint{}_{}.h5'.format(sample_size,label)
        TestLocation = "Result/TestResult{size}_{label}.h5".format(size=sample_size, label=label)
        TrainLocation = "Result/TrainResult{size}_{label}.h5".format(size=sample_size, label=label)
        scaler_file = "Scaler/scaler_{}.pkl".format(sample_size)
    else:
        print("{} removed".format(remove))
        DataLocation = DATA_DIR+"/FeatureRemoval/Undersampling_Dataset_No_{}.h5".format(remove)
        CheckPoint = 'CheckPoint/FeatureRemoval/CheckPoint{}_{}_No_{}.h5'.format(sample_size,label,remove)
        ScaleInputTrain = "ScaledInput/FeatureRemoval/TrainingDataset{}_No_{}.h5".format(sample_size,remove)
        ScaleInputVal = "ScaledInput/FeatureRemoval/ValidationDataset{}_No_{}.h5".format(sample_size,remove)
        TestLocation = "Result/FeatureRemoval/TestResult{size}_{label}_{remove}.h5".format(size=sample_size, label=label, remove=remove)
        TrainLocation = "Result/FeatureRemoval/TrainResult{size}_{label}_{remove}.h5".format(size=sample_size, label=label, remove=remove)
        scaler_file = "Scaler/FeatureRemoval/scaler_{}_No_{}.pkl".format(sample_size, remove)
    
    x_train, y_train, weight_train = load_dataset(DataLocation,0,sample_size = sample_size)
    #x_train = remove_outlier(x_train)
    x_train = scale_dataset(x_train, sample_size, scaler_file)

    x_test, y_test, weight_test = load_dataset(DataLocation,2,sample_size = sample_size)
    #x_test = remove_outlier(x_test)
    x_test = scale_dataset(x_test, sample_size, scaler_file)

    from keras.models import load_model
    print ("Loading the model checkpoint: {}".format(CheckPoint))
    model = load_model(CheckPoint)
    test_pred = model.predict(x_test)
    train_pred = model.predict(x_train)

    if not os.path.isdir('Result/FeatureRemoval'):
        os.makedirs('Result/FeatureRemoval')
    test_result = h5py.File(TestLocation,"w")
    test_result.create_dataset("Prediction",data=test_pred)
    test_result.create_dataset("Truth",data=y_test)
    test_result.create_dataset("Data",data=x_test)
    test_result.create_dataset("Weight",data=weight_test)
    print("Save to {}".format(TestLocation))
    test_result.close()

    train_result = h5py.File(TrainLocation,"w")
    train_result.create_dataset("Prediction",data=train_pred)
    train_result.create_dataset("Truth",data=y_train)
    train_result.create_dataset("Data",data=x_train)
    train_result.create_dataset("Weight",data=weight_train)
    print("Save to {}".format(TrainLocation))
    train_result.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--create', action='store_true', help='Create dataset')
    parser.add_argument('-t','--test', action='store_true', help='Test on validation set')
    parser.add_argument('-u','--tune', action='store_true', help='Model tuning')
    parser.add_argument('-s','--sample', default=0, help='Use a small sample for training and validation')
    parser.add_argument('-d','--device', default="0", help='GPU device to use')
    parser.add_argument('-nw','--noweight', action='store_true', help='Not use sample weights')
    parser.add_argument('-l','--label', default='', help='Label for benchmark study')
    parser.add_argument('-r','--remove',help='Remove each feature')

    args = parser.parse_args()

    if args.noweight: print ("Not using sample weight")
    print ("Using GPU(s):",args.device)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    if args.remove and args.remove not in FEATURES:
        sys.exit("Request feature to be removed not found!")

    if args.create:
        sys.exit("Don't try to create the dataset here. Use the DataResampling notebook for the moment")
        #create_dataset()
    if args.test:
        testing(args.sample, label=args.label, remove=args.remove)
    elif args.tune:
        tuning(args.sample)
    else:
        training(args.sample, not_use_weight = args.noweight, label=args.label, remove=args.remove)
