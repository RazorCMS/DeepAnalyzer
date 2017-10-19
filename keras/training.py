import h5py
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Convert to regular numpy arrays
def to_regular_array(struct_array):
    return struct_array.view((struct_array.dtype[0], len(struct_array.dtype.names)))


BACKGROUND = ['DYJets','Other','QCD','SingleTop','TTJets','WJets','ZInv']
SIGNAL = ['T2qq_900_850']
DATA_DIR = '/bigdata/shared/analysis/'

# Compute class weights
for i,bkg in enumerate(BACKGROUND):
    _file = h5py.File(DATA_DIR+'/'+bkg+'.h5','r')
    if i == 0: Background = np.copy(_file['Data'])
    else: Background = np.hstack((Background, _file['Data']))
Signal = h5py.File(DATA_DIR+SIGNAL[0]+'.h5','r')['Data']

n_bkg, bins, _ = plt.hist(x = Background['leadingJetPt'], range=(0,10000), bins=200, weights = Background['weight'])
n_sn, _, _ = plt.hist(x = Signal['leadingJetPt'], range=(0,10000), bins=200, weights = Signal['weight'])

bin_width = bins[1] - bins[0]
bkg_integral = bin_width * sum(n_bkg[:])
sn_integral = bin_width * sum(n_sn[:])
print "Background integral = {}, signal integral = {}".format(bkg_integral,sn_integral)

class_weight = { 0: 1., 1: bkg_integral/sn_integral}

# Get shuffled unified dataset for training
Dataset = np.hstack((Background, Signal))
np.random.shuffle(Dataset)
del Background, Signal # free up memory (not sure it helps)

x = Dataset[['alphaT','dPhiMinJetMET','dPhiRazor','HT','jet1MT','leadingJetCISV','leadingJetPt','MET','MHT','MR','MT2','nSelectedJets','Rsq','subleadingJetPt']]
y = Dataset[['label']]
sample_weight = Dataset[['weight']]

x = to_regular_array(x)
y = to_regular_array(y)
sample_weight = to_regular_array(sample_weight)

print x.shape
print y.shape
print sample_weight.shape


data_size = x.shape[0]
training_index = int(0.6*data_size)
val_index = training_index + int(0.2*data_size)

# 60% training, 20% validation, 20% testing
x_train = x[:training_index]
y_train = y[:training_index]
sample_weight_train = sample_weight[:training_index]

x_val = x[training_index:val_index]
y_val = y[training_index:val_index]
sample_weight_val = sample_weight[training_index:val_index]

x_test = x[val_index:]
y_test = y[val_index:]
sample_weight_test = sample_weight[val_index:]

# Training with a simple FFNN
i = Input(shape=(14,))
layer = Dense(100, activation = 'relu')(i)
layer = Dense(100, activation = 'relu')(layer)
layer = Dense(100, activation = 'relu')(layer)
layer = Dense(100, activation = 'relu')(layer)
layer = Dense(10, activation = 'relu')(layer)
o = Dense(1, activation = 'sigmoid')(layer)

model = Model(i,o)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
model.summary()

hist = model.fit(x_train, y_train,
        validation_data = (x_val, y_val),
        nb_epoch = 10,
        batch_size = 128,
        shuffle = True,
        class_weight = class_weight,
        sample_weight = sample_weight_train,
        )

