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


 
start_index_file = int(sys.argv[1])
end_index_file = int(sys.argv[2])



#function for sequence padding
def make_uniform_data(x):
    max_seq = 500
    if len(x) < max_seq:
        pads = ['PAD'] * (max_seq - len(x))
        return pads + x
    elif len(x) > max_seq:
        x = x[len(x)-max_seq:]
        return x
    else:
        return x

#padding at the end of the list
def make_uniform_data_end(x):
    max_seq = 500
    if len(x) < max_seq:
        pads = ['PAD'] * (max_seq - len(x))
        return x + pads
    elif len(x) > max_seq:
        x = x[len(x)-max_seq:]
        return x
    else:
        return x
#=========================================================================#

#function to generate the sequence of clinical and therapy data
def process_sequence(clinical_file, therapy_file):
    clinical = pyreadr.read_r(clinical_file)
    clinical = clinical['f_clinical_part']
    therapy_files = [join(path, f) for f in listdir(path) if (isfile(join(path, f))) & ('f_therapy_part' in f)]
    therapy = pyreadr.read_r(therapy_file)
    therapy = therapy['f_therapy_part']

    #data selection 2016&2017
    clinical = clinical.dropna(subset=['code_id'])
    clinical['event_date'] = pd.to_datetime(clinical['event_date'])
    clinical = clinical.loc[(clinical['event_date'] >= '2016-01-01') & (clinical['event_date'] < '2018-01-01')]

    therapy = therapy.dropna(subset=['code_id'])
    therapy['event_date'] = pd.to_datetime(therapy['event_date'])
    therapy = therapy.loc[(therapy['event_date'] >= '2016-01-01') & (therapy['event_date'] < '2018-01-01')]


    #concat 2 data
    temp_all_raw_data = pd.concat([clinical[['patid', 'event_date', 'code_id']],
               therapy[['patid', 'event_date', 'code_id']]])
    temp_all_raw_data.reset_index(drop=True, inplace=True)

    #create vocab list
    temp_vocab = temp_all_raw_data.code_id.unique().tolist()

    #extract year, month, day from event date (if needed)
    temp_all_raw_data['day'] = temp_all_raw_data.apply(lambda x: str(x['event_date'].day), axis=1)
    temp_all_raw_data['month'] = temp_all_raw_data.apply(lambda x: str(x['event_date'].month), axis=1)
    temp_all_raw_data['year'] = temp_all_raw_data.apply(lambda x: str(x['event_date'].year), axis=1)
    temp_event_data_seq_all = temp_all_raw_data.sort_values(['event_date'],  ascending=False).groupby('patid').agg({'day': lambda x: x.tolist(),
                                                          'month': lambda x: x.tolist(),
                                                          'year': lambda x: x.tolist()}).reset_index()

    #checkpoint 1
    print("checkpoint1")
    print(temp_all_raw_data.shape)
    print(temp_all_raw_data.patid.unique().shape)

    #concat all read code per patient (sorted by date, oldest first)
    temp_all_raw_data['read_code_seq'] = temp_all_raw_data.sort_values(['event_date'], ascending=False).groupby('patid')['code_id'].transform(lambda x: ', '.join(x))
    temp_all_raw_data = temp_all_raw_data.drop_duplicates(subset=['patid']).reset_index(drop=True)     
    temp_all_raw_data.reset_index(drop=True, inplace=True)

    #checkpoint 2
    print("checkpoint2")
    print(temp_all_raw_data.shape)
    print(temp_all_raw_data.patid.unique().shape)


    #merge read code sequence with month sequence
    temp_data_all = temp_all_raw_data[['patid', 'read_code_seq']].merge(temp_event_data_seq_all, how='left',  on='patid')

    #checkpoint 3
    print("checkpoint3")
    print(temp_data_all.shape)
    print(temp_data_all.patid.unique().shape)


    # get sequence length per patient and only keep patient with >10 read codes
    temp_data_all['read_code_seq'] = temp_data_all['read_code_seq'].apply(lambda x: x.strip('""').split(', '))
    temp_data_all['length_read_code_seq'] = temp_data_all['read_code_seq'].apply(lambda x: len(x))
    # temp_data_all = temp_data_all[temp_data_all.length_read_code_seq > 10]

    #checkpoint 4
    print("checkpoint4")
    print(temp_data_all.shape)
    print(temp_data_all.patid.unique().shape)

    #padded data
    temp_data_all['read_code_seq_padded'] = temp_data_all['read_code_seq'].apply(lambda x: make_uniform_data(x))
    temp_data_all['read_code_seq_padded_end'] = temp_data_all['read_code_seq'].apply(lambda x: make_uniform_data_end(x))
    temp_data_all['month_padded'] = temp_data_all['month'].apply(lambda x: make_uniform_data(x))
    temp_data_all['month_padded_end'] = temp_data_all['month'].apply(lambda x: make_uniform_data_end(x))
    
    #checkpoint 5
    print("checkpoint5")
    print(temp_data_all.shape)
    print(temp_data_all.patid.unique().shape)
    
    return temp_data_all, temp_vocab



print('############ Extract data and generate sequence #############')
path = '../ServerData_13Oct2020/'
clinical_files = [join(path, f) for f in listdir(path) if (isfile(join(path, f))) & ('f_clinical_part' in f)]
therapy_files = [join(path, f) for f in listdir(path) if (isfile(join(path, f))) & ('f_therapy_part' in f)]
vocab_all_big = []
cols = ['patid', 'read_code_seq', 'day', 'month', 'year', 'length_read_code_seq',
           'read_code_seq_padded']
data_all_big = pd.DataFrame(columns=cols)

if end_index_file > len(clinical_files):
    end_index_file = len(clinical_files)
    
print (start_index_file, end_index_file)
    
for clinical_file, therapy_file in zip(clinical_files[start_index_file:end_index_file], therapy_files[start_index_file:end_index_file]):
    print("===================================================")
    print(clinical_file, therapy_file)
    #combine data_all
    temp_data, temp_vocab = process_sequence(clinical_file, therapy_file)
    data_all_big = pd.concat([data_all_big, temp_data])
    data_all_big.reset_index(drop=True, inplace=True)
    vocab_all_big = vocab_all_big + temp_vocab
    temp_data = [] #release memory
    temp_vocab = []

    
print('################### Save to pickle and feather ########################')
pickle.dump(vocab_all_big, open('../SeqModel/vocab_all_big_chunk_' + str(start_index_file) + '_' + str(end_index_file) + '.sav', 'wb'))
data_all_big.to_feather('../SeqModel/data_all_chunk' + str(start_index_file) + '_' + str(end_index_file) + '.feather')

print("--- processing time: %s seconds ---" % (time.time() - start_time))

# pickle.dump(code2idx_all_big, open('../SeqModel/code2idx_all_big_06112023.sav', 'wb'))
# pickle.dump(idx2code_all_big, open('../SeqModel/idx2code_all_big_06112023.sav', 'wb'))
