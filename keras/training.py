import h5py
import pandas as pd 
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle 

# Convert to regular numpy arrays
def to_regular_array(struct_array):
    return struct_array.view((np.float32, len(struct_array.dtype.names)))

def clean_dataset(arr):
    print "Before cleaning: {}".format(arr.shape)
    df = pd.DataFrame(data=arr)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    cleaned = pd.DataFrame.as_matrix(df[indices_to_keep].astype(np.float32))
    print "After cleaning: {}".format(cleaned.shape)
    return cleaned

def multiply_data(data, multiplicity):
    print "Dataset size before multiplicity: {}".format(data.shape[0])
    for i in range(multiplicity):
        if i==0: sum_data = np.copy(data)
        else: sum_data = np.hstack((sum_data, data))
    # Reduce the weight of the sum:
    sum_data['weight'] /= multiplicity
    print "Dataset size after multiplicity: {}".format(sum_data.shape[0])
    return sum_data

BACKGROUND = ['DYJets','Other','QCD','SingleTop','TTJets','WJets','ZInv']
SIGNAL = ['T2qq_900_850']
DATA_DIR = '/bigdata/shared/analysis/'

# Compute class weights
for i,bkg in enumerate(BACKGROUND):
    _file = h5py.File(DATA_DIR+'/'+bkg+'.h5','r')
    if i == 0: Background = np.copy(_file['Data'])
    else: Background = np.hstack((Background, _file['Data']))
Signal = h5py.File(DATA_DIR+SIGNAL[0]+'.h5','r')['Data']

Signal = multiply_data(Signal, 3000) # Increase the size of signal to match background


plt.ioff() # turn off interactive mode
n_bkg, bins, _ = plt.hist(x = Background['leadingJetPt'], range=(0,10000), bins=200, weights = Background['weight'])
n_sn, _, _ = plt.hist(x = Signal['leadingJetPt'], range=(0,10000), bins=200, weights = Signal['weight'])

bin_width = bins[1] - bins[0]
bkg_integral = bin_width * sum(n_bkg[:])
sn_integral = bin_width * sum(n_sn[:])
print "Background integral = {}, signal integral = {}".format(bkg_integral,sn_integral)

class_weight = { 0: sn_integral/bkg_integral, 1: 1.}

# Get shuffled unified dataset for training
Dataset = np.hstack((Background, Signal))
np.random.shuffle(Dataset)
del Background, Signal # free up memory (not sure it helps)

Dataset = to_regular_array(Dataset)
Dataset = clean_dataset(Dataset)

x = Dataset[:,2:]
y = Dataset[:,0]
sample_weight = Dataset[:,1]

# 60% training, 20% validation, 20% testing
data_size = x.shape[0]
training_index = int(0.6*data_size)
val_index = training_index + int(0.2*data_size)

x_train = x[:training_index]
y_train = y[:training_index]
sample_weight_train = sample_weight[:training_index]

x_val = x[training_index:val_index]
y_val = y[training_index:val_index]
sample_weight_val = sample_weight[training_index:val_index]

x_test = x[val_index:]
y_test = y[val_index:]
sample_weight_test = sample_weight[val_index:]

# Normalize dataset
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)

# Save scaler to file
scalerfile = 'scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))

# Training with a simple FFNN
i = Input(shape=(14,))
layer = Dense(100, activation = 'relu')(i)
layer = Dense(100, activation = 'relu')(layer)
layer = Dense(100, activation = 'relu')(layer)
layer = Dense(100, activation = 'relu')(layer)
layer = Dense(10, activation = 'relu')(layer)
o = Dense(1, activation = 'sigmoid')(layer)

model = Model(i,o)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x_train, y_train,
        validation_data = (x_val, y_val),
        nb_epoch = 10,
        batch_size = 128,
        shuffle = True,
        class_weight = class_weight,
        sample_weight = sample_weight_train,
        )

histfile = 'history.sav'
pickle.dump(hist.history, open(histfile,'wb'))
