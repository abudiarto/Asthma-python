import pandas as pd
import cudf
import pyreadr
import numpy as np
import torch
import pickle
from os import listdir
from os.path import isfile, join
import sys
import time


#vocab and code2idx generation
vocab_all = pickle.load(open('../SeqModel/all_vocab_clinical.sav', 'rb'))
idx_all = range(1, len(vocab_all)+1)

code2idx_all = dict(zip(vocab_all, idx_all))
idx2code_all = dict(zip(idx_all, vocab_all))

code2idx_all['PAD'] = 0
# code2idx_all['NO_CODE'] = 1
# code2idx_all['start_visit'] = 2
# code2idx_all['end_visit'] = 3
idx2code_all[0] = 'PAD'
# idx2code_all[1] = 'NO_CODE'
# idx2code_all[2] = 'start_visit'
# idx2code_all[3] = 'end_visit'
VOCAB_SIZE = len(code2idx_all)
print('code2idx Size: {}'.format(len(code2idx_all)))
print('idx2code Size: {}'.format(len(idx2code_all)))

# month_vocab = pickle.load(open('../SeqModel/month_vocab.sav', 'rb'))
#month indexing
month_vocab = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
month_idx = range(1, len(month_vocab)+1)

month2idx_all = dict(zip(month_vocab, month_idx))
idx2month_all = dict(zip(month_idx, month_vocab))

month2idx_all['PAD'] = 0
# code2idx_all['start_visit'] = 1
# code2idx_all['end_visit'] = 2
idx2month_all[0] = 'PAD'
# idx2code_all[1] = 'start_visit'
# idx2code_all[2] = 'end_visit'
# VOCAB_SIZE = len(month2idx_all)
print('month2idx Size: {}'.format(len(month2idx_all)))
# print('idx2code Size: {}'.format(len(idx2code_all)))

def fix_system(x):
    if ('SystmOne' in x):
        return 'SystemOne'
    elif ('emis' in x) | ('Emis' in x) | ('EMIS' in x):
        return 'EMIS'
    elif ('Vision' in x):
        return 'Vision'
    elif ('iSoft' in x):
        return 'iSoft'
    elif ('Microtest' in x):
        return 'Microtest'
    else:
        return 'unknown'
patient = pyreadr.read_r('../ServerData_13Oct2020/d_patient_overall.Rdata')
practice = pyreadr.read_r('../ServerData_13Oct2020/d_practice.Rdata')
patient = patient['d_patient_overall']
practice = practice['d_practice']

practice['system'] = practice.apply(lambda x: fix_system(x.system), axis=1)

#Age in 2008-01-01
patient['age'] = patient.year_of_birth.apply(lambda x: 2016-x) #change the year based on the baseline year
patient = patient[['patid', 'practice_id', 'age']].merge(practice[['practice_id', 'Country', 'system']], how='left', on='practice_id')

# Outcomes data
outcomes = pd.read_csv("../FinalData/cleaned_outcomes_01122023.csv")
outcomes['3months'] = outcomes.apply(lambda x: x.outcome_3months, axis=1)
outcomes['6months'] = outcomes.apply(lambda x: (x.outcome_3months)|(x.outcome_6months), axis=1)
outcomes['9months'] = outcomes.apply(lambda x: (x.outcome_3months)|(x.outcome_6months)|(x.outcome_9months), axis=1)
outcomes['12months'] = outcomes.apply(lambda x: (x.outcome_3months)|(x.outcome_6months)|(x.outcome_9months)|(x.outcome_12months), axis=1)


###pocess each chunk
path = '../SeqModel/SeqChunks_clinical/'
files = [join(path, f) for f in listdir(path) if (isfile(join(path, f))) & ('seq_data' in f)]

chunk = 1
# files = zip(clinical_files[chunk-1:], therapy_files[chunk-1:])

for file in files:
    print(file)
    all_raw_data = pickle.load(open(file, 'rb'))
    print(all_raw_data.shape)
 
    all_raw_data['read_code_seq_padded_idx'] = all_raw_data['read_code_seq_padded'].apply(lambda x: [code2idx_all.get(key) for key in x])
    all_raw_data['read_code_seq_padded_end_idx'] = all_raw_data['read_code_seq_padded_end'].apply(lambda x: [code2idx_all.get(key) for key in x])

    all_raw_data['month_padded_idx'] = all_raw_data['month_padded'].apply(lambda x: [month2idx_all.get(key) for key in x])
    all_raw_data['month_padded_idx_end'] = all_raw_data['month_padded_end'].apply(lambda x: [month2idx_all.get(key) for key in x])

    all_raw_data = all_raw_data.merge(patient[['patid', 'Country', 'age', 'system']], how='left', on='patid')
    all_raw_data = all_raw_data.drop_duplicates(subset=['patid']).reset_index(drop=True)  

    all_raw_data = all_raw_data.merge(outcomes[['patid', '3months', '6months', 
                                       '9months', '12months',]], how='inner', on='patid')

    pickle.dump(all_raw_data, open('../SeqModel/SeqChunks_clinical/all_raw_data_indexed_' + str(chunk) +'.sav', 'wb'))
    chunk+=1
    # pickle.dump(vocab_all, open('../SeqModel/all_vocab.sav', 'wb'))
    # pickle.dump(month_vocab, open('../SeqModel/all_vocab_month.sav', 'wb'))