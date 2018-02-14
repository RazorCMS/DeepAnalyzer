import h5py
from argparse import ArgumentParser
import glob
import sys
import numpy as np

parser = ArgumentParser()
parser.add_argument('sample',help='Sample to merge')

args = parser.parse_args()

samples = glob.glob(args.sample+'_*.h5')

if len(samples) == 0: sys.exit("No files to merge")
print(samples)
merge = None
for sample in sorted(samples):
    print ("Processing {}".format(sample))
    with h5py.File(sample,'r') as infile:
        data = infile['Data'][:]
        #print("Shape {}".format(data.shape))
        if merge == None:
            merge = np.copy(data)
        else:
            merge = np.hstack((merge, data))
            #print("Merged shape {}".format(merge.shape))
with h5py.File(args.sample+'.h5','w') as outfile:
    outfile['Data'] = merge
    print("Save to {}".format(args.sample))




        
