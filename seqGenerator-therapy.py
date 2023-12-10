# Usage: to generate sequence of readcode per patient ordered by the event_date
#     - extract readcodes from raw data
#     - group them by event_date
#     - group them by patid
#     - join readcodes
#     - NOTES: change the max_seq and date range based on the needs

    

import pandas as pd
import cudf
import pyreadr
import numpy as np
import torch
import pickle
from os import listdir
from os.path import isfile, join


#padding at the beginning of the list

def make_uniform_data(x):
    max_seq = 129 #change this max_seq (max of sequence length) of date as needed
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
    max_seq = 129 #change this max_seq (max of sequence length) of date as needed
    if len(x) < max_seq: 
        pads = ['PAD'] * (max_seq - len(x))
        return x + pads
    elif len(x) > max_seq:
        x = x[len(x)-max_seq:]
        return x
    else:
        return x

path = '../ServerData_13Oct2020/'
therapy_files = [join(path, f) for f in listdir(path) if (isfile(join(path, f))) & ('f_therapy_part' in f)]

chunk = 1

for therapy_file in therapy_files:
    if (chunk >=25) & (chunk < 30):
        print(therapy_file)
        therapy = pyreadr.read_r(therapy_file)
        therapy = therapy['f_therapy_part']
        
        #data selection 
        therapy = therapy.dropna(subset=['code_id'])
        therapy['event_date'] = pd.to_datetime(therapy['event_date'])
        therapy = therapy.loc[(therapy['event_date'] >= '2016-01-01') & (therapy['event_date'] < '2017-01-01')] #change this range of date as needed
        therapy = therapy[['patid', 'event_date', 'code_id']]

        therapy['read_code_seq_perdate'] = therapy.sort_values(['event_date'], ascending=True).groupby(['patid', 'event_date'])['code_id'].transform(lambda x: ', '.join(x))

        all_raw_data = therapy
        therapy = []
        
        #extract year, month, day from event date
        all_raw_data['day'] = all_raw_data.apply(lambda x: str(x['event_date'].day), axis=1)
        all_raw_data['month'] = all_raw_data.apply(lambda x: str(x['event_date'].month), axis=1)
        all_raw_data['year'] = all_raw_data.apply(lambda x: str(x['event_date'].year), axis=1)
        event_data_seq_all = all_raw_data.sort_values(['event_date'],  ascending=True).groupby('patid').agg({'day': lambda x: x.tolist(),
                                                                  'month': lambda x: x.tolist(),
                                                                  'year': lambda x: x.tolist()}).reset_index()

        all_raw_data=all_raw_data.drop_duplicates(['patid', 'event_date'])
        all_raw_data.reset_index(drop=True, inplace=True)
        all_raw_data = all_raw_data[['patid', 'event_date', 'read_code_seq_perdate']]

        print(all_raw_data.shape)
        print(all_raw_data.patid.unique().shape)

        all_raw_data['read_code_seq'] = all_raw_data.sort_values(['event_date'], ascending=True).groupby(['patid'])['read_code_seq_perdate'].transform(lambda x: ', '.join(x))
        all_raw_data=all_raw_data.drop_duplicates(['patid'])
        all_raw_data.reset_index(drop=True, inplace=True)
        all_raw_data['read_code_seq'] = all_raw_data['read_code_seq'].apply(lambda x: x.strip('""').split(', '))
        all_raw_data['length_read_code_seq'] = all_raw_data['read_code_seq'].apply(lambda x: len(x))
        all_raw_data = all_raw_data.merge(event_data_seq_all, how='left',  on='patid')
        
        #put padding at the beginning
        all_raw_data['read_code_seq_padded'] = all_raw_data['read_code_seq'].apply(lambda x: make_uniform_data(x))
        #as alternative put padding at the end
        all_raw_data['read_code_seq_padded_end'] = all_raw_data['read_code_seq'].apply(lambda x: make_uniform_data_end(x))
        
        all_raw_data['month_padded'] = all_raw_data['month'].apply(lambda x: make_uniform_data(x))
        all_raw_data['month_padded_end'] = all_raw_data['month'].apply(lambda x: make_uniform_data_end(x))
        
        vocab_all = []
        for row in all_raw_data[['read_code_seq']].iterrows():
            vocab_all = vocab_all + row[1][0]
        vocab_all = list(set(vocab_all))

        pickle.dump(all_raw_data[['patid', 'length_read_code_seq',
                                 'read_code_seq_padded', 'read_code_seq_padded_end',
                                 'month_padded', 'month_padded_end']],
                    open('../SeqModel/SeqChunks_therapy/seq_data_'+str(chunk)+'.sav', 'wb'))
        pickle.dump(vocab_all,
                    open('../SeqModel/SeqChunks_therapy/vocab_'+str(chunk)+'.sav', 'wb'))
    chunk+=1