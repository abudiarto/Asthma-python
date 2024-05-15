import pickle
import _pickle as cPickle
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DistilBertTokenizerFast
import time

#function to translate Read into Desc
def read_to_desc_generator (x):
    result_desc = ""
    for code in x:
        # print(code)
        desc_array = read2desc[read2desc.CC == code].TERM60.values
        if len(desc_array) > 0:
            result_desc = result_desc + desc_array[0] + ". "
    return result_desc


#Load data
data = pickle.load(open('../Clean_data/clinical_data.sav', 'rb'))
read2desc = pd.read_csv('../FinalData/Read2Desc.csv')


#split into batches with 10000 data to handle the memory problem
split_size = 10000
n_split = math.ceil(data.shape[0]/10000)
for i in range(n_split):
    start_time = time.time()
    print('split ', str(i), 'from ', (i*split_size), 'to ', str(split_size*(i+1)))
    temp_data = data.iloc[i*split_size:split_size*(i+1),]
    temp_data = temp_data.reset_index(drop=True)
    temp_data['Desc'] = temp_data.apply(lambda x: read_to_desc_generator(x.read_code_seq_padded_noPAD), axis=1)
    pickle.dump(temp_data, open('../Clean_data/clinical_data_'+str(i)+'.sav', 'wb'))
    print(time.time() - start_time)

