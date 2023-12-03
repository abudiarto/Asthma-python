import pandas as pd
import pyreadr
import numpy as np
import torch
import pickle
from os import listdir
from os.path import isfile, join
import sys
import time

path = '../SeqModel/SeqChunks/'
chunk_files = [join(path, f) for f in listdir(path) if (isfile(join(path, f)))]
chunk = 1
all_data = pickle.load(open('../SeqModel/SeqChunks/seq_data_1.sav', 'rb'))

for file in chunk_files:
    print(file)
    print(chunk)
    if (file != '../SeqModel/SeqChunks/seq_data_1.sav') & (chunk < 5):
        temp = pickle.load(open(file, 'rb'))
        all_data = pd.concat([all_data, temp])
        chunk+=1

pickle.dump(all_data, open('../SeqModel/all_data.sav', 'wb'))
    