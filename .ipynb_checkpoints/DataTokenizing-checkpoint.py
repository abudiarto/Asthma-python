import pickle
import _pickle as cPickle

import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, log_loss, confusion_matrix

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DistilBertTokenizerFast
from transformers import TrainerCallback
# from transformers import LlamaForCausalLM, LlamaTokenizer

import torch
from torch.nn import CrossEntropyLoss
from copy import deepcopy


class GetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

data = pickle.load(open('../Clean_data/Clinical_data_WithDesc.sav', 'rb'))
print('data loaded successfully')

target_outcomes = '12months'
ignore, use = train_test_split(data, stratify=data[target_outcomes], test_size=0.2, random_state=1234)
search_train, search_val = train_test_split(use, stratify=use[target_outcomes], test_size=0.2, random_state=1234)
search_train.reset_index(inplace=True, drop=True)
search_val.reset_index(inplace=True, drop=True)
print('trainset size: ', search_train.shape)
print('testset size: ', search_val.shape)

train_texts = search_train['TERM60'].values.tolist()
train_labels = search_train['12months'].values.tolist()
val_texts = search_val['TERM60'].values.tolist()
val_labels = search_val['12months'].values.tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=.2)

# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/medicine-LLM")
print('tokenizer loaded successfully')


# train_encodings = tokenizer(train_texts, truncation=True, padding=True)
# val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)



# train_dataset = GetDataset(train_encodings, train_labels)
# val_dataset = GetDataset(val_encodings, val_labels)
test_dataset = GetDataset(test_encodings, test_labels)

pickle.dump(test_dataset, open('../Clean_data/tokenizedData_test.sav', 'wb'))
print('tokenized data saved successfully')
