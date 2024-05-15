import pickle
import _pickle as cPickle

import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, log_loss, confusion_matrix

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
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


# train_dataset = pickle.load(open('../Clean_data/tokenizedData_train.sav', 'rb'))
# val_dataset = pickle.load(open('../Clean_data/tokenizedData_val.sav', 'rb'))
test_dataset = pickle.load(open('../Clean_data/tokenizedData_test.sav', 'rb'))
print('data loaded successfully')

lang = 'en'

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train@"+lang)
            return control_copy

def compute_metrics(pred):
    global num_labels
    num_labels = 2
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    loss_fct = CrossEntropyLoss()
    logits = torch.tensor(pred.predictions)
    labels = torch.tensor(labels)
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return {
        # 'accuracy@'+lang: acc,
        'f1@'+lang: f1,
        'precision@'+lang: precision,
        'recall@'+lang: recall,
        'loss@'+lang: loss,
    }

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

print('model loaded successfully')

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=test_dataset,         # training dataset
    eval_dataset=test_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,
)

trainer.add_callback(CustomCallback(trainer)) 

train_result = trainer.train()
# print('train_result: ', train_result)
# trainer.save_model("../Clean_data/MedLLM13052024.mdl")
# pickle.dump(train_result, open('../Clean_data/TrainResult_MedLLM13052024.sav', 'wb'))

# test_result = trainer.evaluate(metric_key_prefix='test_en',
#                 eval_dataset=test_dataset)
# pickle.dump(train_result, open('../Clean_data/TestResult_MedLLM13052024.sav', 'wb'))
# print('test_result: ', test_result)

# test_pred = trainer.predict(test_dataset)
# preds = []
# for x in test_pred.predictions:
#     preds.append(np.argmax(x))
    

# print('Confussion matrix')
# confusion_matrix(test_pred.label_ids, preds)
