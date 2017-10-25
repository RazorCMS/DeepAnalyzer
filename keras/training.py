import h5py
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.externals import joblib
import pickle 
import argparse
import os

DATA_DIR = '/bigdata/shared/analysis/'
#DATA_DIR = '/home/ubuntu/data/'
SCALER = 'scaler.pkl'

# Convert to regular numpy arrays
def to_regular_array(struct_array):
    # There is an integer column (nSelectedJets) in the structured array. Need to convert to float before converting to regular array
    dt = struct_array.dtype.descr
    dt[13] = (dt[13][0],np.float32)
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
    # In order: 'alphaT','dPhiMinJetMET','dPhiRazor','HT','jet1MT','leadingJetCISV','leadingJetPt','MET','MHT','MR','MT2','nSelectedJets','Rsq','subleadingJetPt'
    print ("Removing outlier")
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

def load_dataset(location, load_type = 0, train_size = 0):
    loadfile = h5py.File(location,"r")

    def decode(load_type):
        if load_type == 0: return "Training"
        elif load_type == 1: return "Validation"
        else: return "Test"

    dat = loadfile[decode(load_type)]
    if train_size==0:
        _x = dat[:,2:]
        _y = dat[:,0].astype(int)
        _weight = dat[:,1]*1e6
    else:
        _x = dat[0:train_size,2:]
        _y = dat[0:train_size,0]
        _weight = dat[0:train_size,1]*1e6
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

def scale_fit(x_train):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    joblib.dump(scaler, SCALER)
    print ("Saving scaler information to {}".format(SCALER))

def scale_dataset(x_train):
    scaler = joblib.load(SCALER)
    _x_train = scaler.transform(x_train)
    return _x_train

def create_model():
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout
    
    # Training with a simple FFNN
    i = Input(shape=(14,))
    layer = Dense(100, activation = 'relu')(i)
    #layer = Dropout(0.1)(layer)
    layer = Dense(30, activation = 'relu')(layer)
    #layer = Dropout(0.2)(layer)
    layer = Dense(10, activation = 'relu')(layer)
    #layer = Dropout(0.5)(layer)
    o = Dense(2, activation = 'softmax')(layer)
    #o = Dense(1, activation=None)(layer)

    model = Model(i,o)
    model.summary()
    return model

def training(train_size = 0, use_weight=False):
    print ("Loading data...")
    x_train, y_train, weight_train = load_dataset(DATA_DIR+"/CombinedDataset_Balanced.h5",0,train_size = train_size)
    x_val, y_val, weight_val = load_dataset(DATA_DIR+"/CombinedDataset_Balanced.h5",1,train_size=train_size)

    x_train = remove_outlier(x_train)
    x_val = remove_outlier(x_val)
    
    from keras.utils.np_utils import to_categorical
    y_train = to_categorical(y_train,2)
    has_nan(y_train,"training categorical label")
    y_val = to_categorical(y_val,2)
    has_nan(y_val,"validation categorical label")
    
    print ("Scaling features...")
    has_nan(x_train, "unscaled training")
    if not np.isfinite(x_train).all():
        print ("Unscaled training probably contains inf")
    scale_fit(x_train)
    x_train = scale_dataset(x_train)
    x_val = scale_dataset(x_val)
    has_nan(x_train, "scaled training")
    has_nan(y_val, "scaled validation")

    # Save scaled training input to file
    train_ds = h5py.File("TrainingDataset%d.h5"%(train_size),"w")
    train_ds['x'] = x_train
    train_ds['y'] = y_train
    train_ds['w'] = weight_train
    train_ds.close()
    print ("Write to TrainingDataset%d.h5"%(train_size))
    
    class_weight = get_class_weight(y_train)

    model = create_model()
    from keras import optimizers
    model.compile(optimizer = optimizers.Adam(lr=1e-3), loss = 'binary_crossentropy', metrics=['accuracy'])
   
    if (use_weight):
        val_tuple = (x_val, y_val, weight_val)
        sample_weight = weight_train
    else:
        val_tuple = (x_val, y_val)
        sample_weight = None
    
    from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
    hist = model.fit(x_train, y_train,
            validation_data = val_tuple,
            epochs = 100,
            batch_size = 128,
            class_weight = class_weight,
            sample_weight = sample_weight,
            callbacks = [ModelCheckpoint(filepath='CheckPoint%d.h5'%(train_size), verbose = 1, save_best_only=True), ReduceLROnPlateau(patience = 4, factor = 0.5, verbose = 1, min_lr=1e-7), EarlyStopping(patience = 10)],
            )

    histfile = 'history%d.sav'%(train_size)
    pickle.dump(hist.history, open(histfile,'wb'))
    print ("Save history to history%d.sav"%(train_size))

    val_pred = model.predict(x_val)
    val_label = np.argmax(y_val, axis=1)

    val_result = h5py.File("ValidationResult%d.h5"%(train_size),'w')
    val_result.create_dataset("Prediction", data = val_pred)
    val_result.create_dataset("Truth", data = val_label)
    val_result.create_dataset("Weight", data = weight_val)
    print ("Save result to ValidationResult%d.h5"%(train_size))
    val_result.close()

def testing(sample_size = 0, use_weight=False):
    print ("Loading the model checkpoint...")
    x_test, y_test, weight_test = load_dataset(DATA_DIR+"/CombinedDataset_Balanced.h5",2,train_size = sample_size)
    x_test = scale_dataset(x_test)

    x_train, y_train, weight_train = load_dataset(DATA_DIR+"/CombinedDataset_Balanced.h5",0,train_size = sample_size)
    x_train = scale_dataset(x_train)

    from keras.models import load_model
    model = load_model('CheckPoint%d.h5'%(sample_size))
    test_pred = model.predict(x_test)
    train_pred = model.predict(x_train)

    test_result = h5py.File("TestResult%d.h5"%(sample_size),"w")
    test_result.create_dataset("Prediction",data=test_pred)
    test_result.create_dataset("Truth",data=y_test)
    test_result.create_dataset("Weight",data=weight_test)
    print("Save to TestResult%d.h5"%(sample_size))
    test_result.close()

    train_result = h5py.File("TrainResult%d.h5"%(sample_size),"w")
    train_result.create_dataset("Prediction",data=train_pred)
    train_result.create_dataset("Truth",data=y_train)
    train_result.create_dataset("Weight",data=weight_train)
    print("Save to TrainResult%d.h5"%(sample_size))
    train_result.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--create', action='store_true', help='Create dataset')
    parser.add_argument('-t','--test', action='store_true', help='Test on validation set')
    parser.add_argument('-s','--sample', type=int, default=0, help='Use a small sample for training and validation')
    parser.add_argument('-d','--device', default="0", help='GPU device to use')
    parser.add_argument('-w','--weight', action='store_true', help='Use sample weights')

    # TODO: When using sample weight, the prediction probability should be weighted as well.
    # TODO: if using small sample, save output to a different file

    args = parser.parse_args()

    print ("Using GPU ",args.device)
    if args.weight: print ("Using sample weight")
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device


    if args.create:
        create_dataset()
    if args.test:
        testing(args.sample, use_weight = args.weight)
    else:
        training(args.sample, use_weight = args.weight)
        

