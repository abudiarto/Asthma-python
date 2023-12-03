import pandas as pd
import pyreadr
import numpy as np
import torch
import pickle
from os import listdir
from os.path import isfile, join
import sys
import time

start_time = time.time()

print('################### Load patient and practice data #########################')
patient = pyreadr.read_r('../ServerData_13Oct2020/d_patient_overall.Rdata')
practice = pyreadr.read_r('../ServerData_13Oct2020/d_practice.Rdata')

patient = patient['d_patient_overall']
practice = practice['d_practice']

#Age in 2016-01-01
patient['age'] = patient.year_of_birth.apply(lambda x: 2016-x)

patient = patient[['patid', 'practice_id', 'age']].merge(practice[['practice_id', 'Country']], how='left', on='practice_id')

print('################### Generate outcomes: asthma attack in the next 3, 6, 9, and 12 months #########################')
# Outcomes data
outcomes = pd.read_csv("../FinalData/cleaned_outcomes_08112023.csv")


print('################### Merge data and vocab chunks #########################')
print('file_chunk_0')
data_all_big = pd.read_feather('../SeqModel/data_all_chunk_0.feather')
vocab_all_big = pickle.load(open('../SeqModel/vocab_all_big_chunk_0.sav', 'rb'))
for i in range(3,4):
    print('file_chunk_', i)
    temp_data = pd.read_feather('../SeqModel/data_all_chunk_'+str(i)+'.feather')
    temp_vocab = pickle.load(open('../SeqModel/vocab_all_big_chunk_'+str(i)+'.sav', 'rb'))
    data_all_big = pd.concat([data_all_big, temp_data])
    vocab_all_big = vocab_all_big + temp_vocab
    del([temp_data, temp_vocab])
    
print(data_all_big.shape)
print(len(vocab_all_big))    
    


print('################ Sequence indexing ###################')
#vocab and code2idx generation
vocab_all_big=list(dict.fromkeys(vocab_all_big)) #remove duplicate
idx_all_big = range(1, len(vocab_all_big)+1)
code2idx_all_big = dict(zip(vocab_all_big, idx_all_big))
idx2code_all_big = dict(zip(idx_all_big, vocab_all_big))
code2idx_all_big['PAD'] = 0
idx2code_all_big[0] = 'PAD'
VOCAB_SIZE_big = len(code2idx_all_big)
print('code2idx Size: {}'.format(len(code2idx_all_big)))
print('idx2code Size: {}'.format(len(idx2code_all_big)))

##month vocab
#month indexing
month_vocab_all_big = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
month_idx_all_big = range(1, len(month_vocab_all_big)+1)
month2idx_all_big = dict(zip(month_vocab_all_big, month_idx_all_big))
idx2month_all_big = dict(zip(month_idx_all_big, month_vocab_all_big))
month2idx_all_big['PAD'] = 0
idx2month_all_big[0] = 'PAD'
print('month2idx Size: {}'.format(len(month2idx_all_big)))

data_all_big['read_code_seq_padded_idx'] = data_all_big['read_code_seq_padded'].apply(lambda x: [code2idx_all_big.get(key) for key in x])
data_all_big['read_code_seq_padded_idx_end'] = data_all_big['read_code_seq_padded_end'].apply(lambda x: [code2idx_all_big.get(key) for key in x])
data_all_big['month_padded_idx'] = data_all_big['month_padded'].apply(lambda x: [month2idx_all_big.get(key) for key in x])
data_all_big['month_padded_idx_end'] = data_all_big['month_padded_end'].apply(lambda x: [month2idx_all_big.get(key) for key in x])
    
print('################### Merge with outcomes and patients data #########################')
data_all_big = data_all_big.merge(outcomes[['patid', 'new_3MonthsOutcome', 'new_6MonthsOutcome', 
                                   'new_9MonthsOutcome', 'new_12MonthsOutcome',]], how='inner', on='patid')
data_all_big = data_all_big.merge(patient[['patid', 'Country', 'age']], how='left', on='patid')
data_all_big = data_all_big.drop_duplicates(subset=['patid']).reset_index(drop=True)     

outcomes = []
patient = []
practice = []

print(data_all_big.shape)
print(data_all_big.patid.unique().shape)

pickle.dump(code2idx_all_big, open('../SeqModel/code2idx_all_big_08112023_75%.sav', 'wb'))
pickle.dump(idx2code_all_big, open('../SeqModel/idx2code_all_big_08112023_75%.sav', 'wb'))
pickle.dump(month2idx_all_big, open('../SeqModel/month2idx_all_big_08112023_75%.sav', 'wb'))
pickle.dump(idx2month_all_big, open('../SeqModel/idx2month_all_big_08112023_75%.sav', 'wb'))
print('-----------------------xxxxx-------------------------')
pickle.dump(data_all_big, open('../SeqModel/data_all_big_08112023_75%.sav', 'wb'))
# data_all_big.to_feather('../SeqModel/data_all_big_08112023.feather')



print("--- processing time: %s seconds ---" % (time.time() - start_time))