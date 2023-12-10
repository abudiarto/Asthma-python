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
chunk_files = [join(path, f) for f in listdir(path) if (isfile(join(path, f))) & ('all_raw_data_indexed_' in f)]
chunk = 1
kept_columns = ['patid', 'read_code_seq_padded_idx', 'read_code_seq_padded_end_idx',
       'month_padded_idx', 'month_padded_idx_end', 'Country', 'age', 'system',
       '3months', '6months', '9months', '12months']
all_data = pickle.load(open('../SeqModel/SeqChunks/all_raw_data_indexed_1.sav', 'rb'))[kept_columns]


for file in chunk_files:
    print(file)
    print(chunk)
    if (file != '../SeqModel/SeqChunks/all_raw_data_indexed_1.sav') & (chunk < 25):
        temp = pickle.load(open(file, 'rb'))[kept_columns]
        all_data = pd.concat([all_data, temp])
        all_data.reset_index(drop=True, inplace=True)
        chunk+=1

print('data shape: ', all_data.shape)
# pickle.dump(all_data, open('../SeqModel/all_data.sav', 'wb'))
all_data.to_feather('../SeqModel/all_data_1year.feather')
    