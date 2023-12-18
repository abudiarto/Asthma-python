import pandas as pd
import pyreadr
import numpy as np
import torch
import pickle
from os import listdir
from os.path import isfile, join
import sys
import time

path = '../SeqModel/SeqChunks_therapy/'
chunk_files = [join(path, f) for f in listdir(path) if (isfile(join(path, f))) & ('vocab' in f)]
chunk = 1
all_vocab = pickle.load(open('../SeqModel/SeqChunks_therapy/vocab_1.sav', 'rb'))

for file in chunk_files:
    print(file)
    print(chunk)
    if (file != '../SeqModel/SeqChunks_therapy/vocab_1.sav') & (chunk < 100):
        temp = pickle.load(open(file, 'rb'))
        all_vocab = list(set(all_vocab)) + list(set(temp))
        all_vocab = list(set(all_vocab))
        chunk+=1
        

print('vocab size: ', str(len(all_vocab)))
pickle.dump(all_vocab, open('../SeqModel/all_vocab_therapy.sav', 'wb'))