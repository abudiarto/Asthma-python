import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, Input, concatenate, Reshape, Activation, Flatten, Add, BatchNormalization, Multiply, LeakyReLU
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.metrics import AUC, SensitivityAtSpecificity
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, Adamax, SGD, Adadelta
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import L1L2, L1, L2
from livelossplot import PlotLossesKeras


import keras_tuner

# from tensorflow.keras.backend.tensorflow_backend import set_session
from tensorflow.python.keras import backend as K

#internal validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score, matthews_corrcoef, auc, average_precision_score, roc_auc_score, balanced_accuracy_score, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
import pyreadr

from os import listdir
from os.path import isfile, join
import time
import datetime

# fix random seed for reproducibility
tf.random.set_seed(1234)

def lateFussion(Xt_train, Xs_train, Xm_train, y_train, Xt_val, Xs_val, Xm_val, y_val, vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length, hp):
    #Dense layer for tabular data
    inputs1 = Input(shape=(Xt_train.shape[1],))
    nn = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(inputs1)
    nn = Dropout(0.5)(nn)
    nn = Dense(32, activation='relu', kernel_initializer='glorot_uniform')(nn)
    nn = Dropout(0.3)(nn)
    # nn = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(nn)
    # #nn = Dropout(0.4)(nn)
    # nn = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(nn)
    #nn = Dropout(0.4)(nn)
    # nn = Dense(128, activation='relu')(nn)
    nn = Reshape((1, 64))(nn)
    
    #Embedding and LSTM for sequence data    
    inputs2 = Input(shape=(Xs_train.shape[1],))
    inputs3 = Input(shape=(Xm_train.shape[1],))
    embedding = Embedding(vocab_size, embedding_vector_length, input_length=max_codes)(inputs2)
    lstm = Bidirectional(LSTM(32, return_sequences=True, bias_regularizer=L1L2(l1=0.0, l2=0.01)))(embedding)
    lstm = Dropout(0.3)(lstm)
    lstm = Bidirectional(LSTM(16, return_sequences=True, bias_regularizer=L1L2(l1=0.0, l2=0.01)))(lstm)
    lstm = Dropout(0.3)(lstm)
    # lstm = Bidirectional(LSTM(16, return_sequences=True, bias_regularizer=L1L2(l1=0.0, l2=0.01)))(lstm)
    # lstm = Dropout(0.3)(lstm)
    
    embedding_month = Embedding(month_size, month_embedding_vector_length, input_length=max_codes)(inputs3)
    lstm_month = Bidirectional(LSTM(32, return_sequences=True, bias_regularizer=L1L2(l1=0.0, l2=0.01)))(embedding_month)
    lstm_month = Dropout(0.3)(lstm_month)
    lstm_month = Bidirectional(LSTM(16, return_sequences=True, bias_regularizer=L1L2(l1=0.0, l2=0.01)))(embedding_month)
    lstm_month = Dropout(0.3)(lstm_month)

    model_tot = concatenate([nn, lstm, lstm_month], axis=1)
    # model_tot = BatchNormalization()(model_tot)
    model_tot = Dense(32, activation='relu', kernel_initializer='glorot_uniform')(model_tot)
    # model_tot = Dropout(0.4)(model_tot)
    model_tot = Flatten()(model_tot)
    output = Dense(1, activation='sigmoid')(model_tot)
    
    opt = Adam(learning_rate=lr, clipvalue=clipvalue)
    metrics = [
        AUC(num_thresholds=3, name='auc', curve='ROC'),
        tf.keras.metrics.Precision(name='prec'),
        tf.keras.metrics.Recall(name='rec'),
        tf.keras.metrics.TrueNegatives(name='TN'),
        tf.keras.metrics.TruePositives(name='TP'),
        tf.keras.metrics.PrecisionAtRecall(0.8)
    ]
    
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=output)
    model.compile(
        loss='binary_crossentropy', 
        optimizer=opt, 
        metrics=metrics)
    return model

def earlyFussion(Xt_train, Xs_train, Xm_train, y_train, Xt_val, Xs_val, Xm_val, y_val, vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length, hp):
       
    inputs1 = Input(shape=(Xt_train.shape[1],))
    inputs2 = Input(shape=(Xs_train.shape[1],))
    inputs3 = Input(shape=(Xm_train.shape[1],))
    
    #LAYER 0 
    #demography
    neurons_layer0 = hp.Int("units_layer0", min_value=32, max_value=512, step=32)
    
    nn = Dense(units=neurons_layer0, 
               activation=hp.Choice("activation_layer0", ["relu", "elu", ]), 
               kernel_initializer=hp.Choice("kernel_initializer_layer0", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
               kernel_regularizer=L1L2(l1=hp.Float("l1_dense0", min_value=0.0, max_value=0.1, step=0.02), 
                                       l2=hp.Float("l2_dense0", min_value=0.0, max_value=0.1, step=0.02)
                                      )
              )(inputs1)
    
    nn = Reshape((1, neurons_layer0))(nn)
    
    #clinical embedding for lstm
    embedding = Embedding(vocab_size, 
                          output_dim = hp.Int("embedding_vector_length", min_value=50, max_value=100, step=10), 
                          input_length=max_codes
                         )(inputs2)
    lstmClinical = LSTM(units=neurons_layer0, return_sequences=True, 
                        kernel_regularizer=L1L2(l1=hp.Float("l1_lstmClinical", min_value=0.0, max_value=0.1, step=0.02), 
                                       l2=hp.Float("l2_lstmClinical", min_value=0.0, max_value=0.1, step=0.02)
                                      )
                       )(embedding)
    
    #month embedding for lstm
    embedding_month = Embedding(month_size, 
                                output_dim = hp.Int("embedding_month_length", min_value=3, max_value=7, step=1), 
                                input_length=max_codes
                               )(inputs3)
    lstmMonth = LSTM(units=neurons_layer0, 
                     return_sequences=True, 
                     kernel_regularizer=L1L2(l1=hp.Float("l1_lstmMonth", min_value=0.0, max_value=0.1, step=0.02), 
                                       l2=hp.Float("l2_lstmMonth", min_value=0.0, max_value=0.1, step=0.02)
                                      )
                    )(embedding_month)
   
    lstm = Add()([lstmClinical, lstmMonth]) #merge two Embedding
    lstm = LSTM(units=neurons_layer0, return_sequences=True, 
                kernel_regularizer=L1L2(l1=hp.Float("l1_lstm0", min_value=0.0, max_value=0.1, step=0.02), 
                                       l2=hp.Float("l2_lstm0", min_value=0.0, max_value=0.1, step=0.02)
                                      )
               )(lstm)
    
    #LAYER 1
    neurons_layer1 = hp.Int("units_layer1", min_value=32, max_value=512, step=32)
    
    add = concatenate([nn, lstm], axis=1) #merge LSTM and dense 
    nn = Dense(units=neurons_layer1, activation=hp.Choice("activation_layer1", ["relu", "elu"]), 
               kernel_initializer=hp.Choice("kernel_initializer_layer1", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
               kernel_regularizer=L1L2(l1=hp.Float("l1_dense1", min_value=0.0, max_value=0.1, step=0.02), 
                                       l2=hp.Float("l2_dense1", min_value=0.0, max_value=0.1, step=0.02)
                                      )
              )(add)
    lstm = LSTM(units=neurons_layer1, return_sequences=True, 
                kernel_regularizer=L1L2(l1=hp.Float("l1_lstm1", min_value=0.0, max_value=0.1, step=0.02), 
                                       l2=hp.Float("l2_lstm1", min_value=0.0, max_value=0.1, step=0.02)
                                      )
               )(lstm)
    
#     add = concatenate([nn, lstm], axis=1) #merge LSTM and dense 
#     nn = Dense(neurons_layer2, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=L1L2(l1=0.0, l2=0.01))(add)
#     lstm = LSTM(units=neurons_layer1, return_sequences=True, kernel_regularizer=L1L2(l1=0.0, l2=0.01))(lstm)
    
#     add = concatenate([nn, lstm], axis=1) #merge LSTM and dense 
#     nn = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=L1L2(l1=0.0, l2=0.01))(add)
#     lstm = LSTM(units=32, return_sequences=True, kernel_regularizer=L1L2(l1=0.0, l2=0.01))(lstm)
    #LAYER FINAL CONCAT
    neurons_layerFinal = hp.Int("units_layer_final", min_value=16, max_value=64, step=16)
    model_tot = concatenate([nn, lstm], axis=1)
    model_tot = BatchNormalization()(model_tot)
    model_tot = Dense(units=neurons_layerFinal, 
                      activation=hp.Choice("activation_layerFinal", ["relu", "elu"]),
                      kernel_regularizer=L1L2(l1=hp.Float("l1_final", min_value=0.0, max_value=0.1, step=0.02), 
                                       l2=hp.Float("l2_final", min_value=0.0, max_value=0.1, step=0.02)
                                      ), 
                      kernel_initializer=hp.Choice("kernel_initializer_final", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]),
                     )(model_tot)
    
    model_tot = Flatten()(model_tot)
    output = Dense(1, activation='sigmoid')(model_tot)
    
    opt = Adadelta(learning_rate=hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log"), 
                   clipvalue=hp.Float("clipvalue", min_value=0.3, max_value=0.7, step=0.2))
    
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='loss')
    
    metrics = [
        AUC(num_thresholds=3, name='auc', curve='ROC'),
        tf.keras.metrics.Precision(name='prec'),
        tf.keras.metrics.Recall(name='rec'),
        tf.keras.metrics.TrueNegatives(name='TN'),
        tf.keras.metrics.TruePositives(name='TP'),
        tf.keras.metrics.PrecisionAtRecall(0.8)
    ]
    
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=output)
    model.compile(
        loss=loss, 
        optimizer=opt, 
        metrics=metrics)
    return model

def load_data(target_outcome):
    print('----------------- Load Data ------------------------------')
    code2idx = pickle.load(open('../SeqModel/all_vocab.sav', 'rb'))
    month2idx = pickle.load(open('../SeqModel/all_vocab_month.sav', 'rb'))
    # idx2month = pickle.load(open('../SeqModel/idx2month_all_big_08112023_75%.sav', 'rb'))
    data = pickle.load(open('../SeqModel/all_raw_data_indexed.sav', 'rb'))
    print(data.shape)
    print('-----------------we passed the heaviest part------------------------------')
    tabularData = pd.read_csv('../FinalData/cleaned_features_2vs1_15112023.csv')
    tabularData = tabularData.drop_duplicates(subset=['patid'])

    extractVars = ['patid', 'sex', 'BMI', 'imd_decile', 'smokingStatus', 'month_1', 'month_2', 'month_3',
     'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
     'month_10', 'month_11', 'month_12', ]


    vocab_size = len(code2idx)
    month_size = len(month2idx)    
    print(month_size)
    print(data.shape)
    print(tabularData.shape)

    data = data.merge(tabularData[extractVars], how = 'left', on='patid')
    data = pd.get_dummies(data, columns=['imd_decile', 'smokingStatus', 'system']) #one hot encoding
    tabularData = []
    print(data.shape)
    
    print('########## Data split, train=England, eval=Scot+Wales ################')
    trainingData = data[(data.Country == 'England') & (data.age >= 18)]
    trainingData, valData = train_test_split(trainingData, test_size=0.3, stratify=trainingData[target_outcome], shuffle=True, random_state=1234)
    trainingData, evalData = train_test_split(trainingData, test_size=0.2, stratify=trainingData[target_outcome], shuffle=True, random_state=1234)
    testData = data[((data.Country == 'Wales') | (data.Country == 'Scotland')) & (data.age >= 18)]
    testDataWales = data[(data.Country == 'Wales') & (data.age >= 18)]
    testDataScotland = data[(data.Country == 'Scotland') & (data.age >= 18)]
    

    print('Train: ', trainingData.shape[0])
    print('Val: ', valData.shape[0])
    print('Eval (internal validation): ', evalData.shape[0])
    print('Test: ', testData.shape[0])
    print('Test - Wales: ', testDataWales.shape[0])
    print('Test - Scotland: ', testDataScotland.shape[0])

    print('############# make sure no data leak between sets #######################')
    print(list(set(trainingData.patid.values).intersection(set(valData.patid.values))))
    print(list(set(trainingData.patid.values).intersection(set(evalData.patid.values))))
    print(list(set(valData.patid.values).intersection(set(evalData.patid.values))))
    print(list(set(valData.patid.values).intersection(set(testData.patid.values))))
    print(list(set(trainingData.patid.values).intersection(set(testData.patid.values))))
    print(len(list(set(testData.patid.values).intersection(set(testDataScotland.patid.values))))) # here data leak is expected)

    print('############# Positive and negative groups ratio #######################')
    print(trainingData[target_outcome].value_counts(normalize=True))
    print(valData[target_outcome].value_counts(normalize=True))
    print(evalData[target_outcome].value_counts(normalize=True))
    print(testData[target_outcome].value_counts(normalize=True))
    print(testDataWales[target_outcome].value_counts(normalize=True))
    print(testDataScotland[target_outcome].value_counts(normalize=True))
    
    return trainingData, valData, evalData, testData, testDataWales, testDataScotland, vocab_size, month_size

def generate_X_y(trainingData, valData, evalData, testData, testDataWales, testDataScotland, target_outcome):

    print('############## Generate X (Xt: tabular | Xs: read code sequence | Xm: month visit sequence) and y ####################')
    tabularVars = ['age', 'sex', 'BMI', 'smokingStatus_Active Smoker', 
                   'smokingStatus_Former Smoker', 'smokingStatus_Non Smoker',
                   'imd_decile_0', 'imd_decile_1', 'imd_decile_2', 'imd_decile_3',
                   'imd_decile_4', 'imd_decile_5', 'imd_decile_6', 'imd_decile_7',
                   'imd_decile_8', 'imd_decile_9', 'imd_decile_10', 
                   'system_EMIS', 'system_SystemOne', 'system_Vision',
                   # 'system_iSoft', 'system_Microtest', 'system_unknown',
                   # 'month_1', 'month_2', 'month_3','month_4', 'month_5', 
                   # 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 
                   # 'month_11', 'month_12',
                  ]
    Xt_train = np.array(trainingData[tabularVars].values)
    Xt_val = np.array(valData[tabularVars].values)
    Xt_eval = np.array(evalData[tabularVars].values)
    Xt_test = np.array(testData[tabularVars].values)
    Xt_testWales = np.array(testDataWales[tabularVars].values)
    Xt_testScotland= np.array(testDataScotland[tabularVars].values)

    Xs_train = np.array(trainingData.read_code_seq_padded_end_idx.values)
    Xs_train = np.array([x for x in Xs_train])
    Xs_val = np.array(valData.read_code_seq_padded_end_idx.values)
    Xs_val = np.array([x for x in Xs_val])
    Xs_eval = np.array(evalData.read_code_seq_padded_end_idx.values)
    Xs_eval = np.array([x for x in Xs_eval])
    Xs_test = np.array(testData.read_code_seq_padded_end_idx.values)
    Xs_test = np.array([x for x in Xs_test])
    Xs_testWales = np.array(testDataWales.read_code_seq_padded_end_idx.values)
    Xs_testWales = np.array([x for x in Xs_testWales])
    Xs_testScotland = np.array(testDataScotland.read_code_seq_padded_end_idx.values)
    Xs_testScotland = np.array([x for x in Xs_testScotland])

    Xm_train = np.array(trainingData.read_code_seq_padded_end_idx.values)
    Xm_train = np.array([x for x in Xm_train])
    Xm_val = np.array(valData.read_code_seq_padded_end_idx.values)
    Xm_val = np.array([x for x in Xm_val])
    Xm_eval = np.array(evalData.read_code_seq_padded_end_idx.values)
    Xm_eval = np.array([x for x in Xm_eval])
    Xm_test = np.array(testData.read_code_seq_padded_end_idx.values)
    Xm_test = np.array([x for x in Xm_test])
    Xm_testWales = np.array(testDataWales.read_code_seq_padded_end_idx.values)
    Xm_testWales = np.array([x for x in Xm_testWales])
    Xm_testScotland = np.array(testDataScotland.read_code_seq_padded_end_idx.values)
    Xm_testScotland = np.array([x for x in Xm_testScotland])

    y_train = trainingData[target_outcome].values
    y_val = valData[target_outcome].values
    y_eval = evalData[target_outcome].values
    y_test = testData[target_outcome].values
    y_testWales = testDataWales[target_outcome].values
    y_testScotland = testDataScotland[target_outcome].values

    #scalling tabular data
    scaler = StandardScaler().fit(Xt_train)
    Xt_train = scaler.transform(Xt_train)
    Xt_val = scaler.transform(Xt_val)
    Xt_eval = scaler.transform(Xt_eval)
    Xt_test = scaler.transform(Xt_test)
    Xt_testWales = scaler.transform(Xt_testWales)
    Xt_testScotland = scaler.transform(Xt_testScotland)
    
    return Xt_train, Xt_val, Xt_eval, Xt_test, Xt_testWales, Xt_testScotland, Xs_train, Xs_val, Xs_eval, Xs_test, Xs_testWales, Xs_testScotland, Xm_train, Xm_val, Xm_eval, Xm_test, Xm_testWales, Xm_testScotland, y_train, y_val, y_eval, y_test, y_testWales, y_testScotland


def set_parameters(trainingData, target_outcome):
    print('################# Set model parameters')
    pos_weight = trainingData[target_outcome].value_counts()[0]/trainingData[target_outcome].value_counts()[1]
    neg_weight = trainingData[target_outcome].value_counts()[1]/trainingData[target_outcome].value_counts()[0]
    class_weight = {0:1, 1:pos_weight}
    print('class_weight: ', class_weight)

    output_bias = np.array([np.log(neg_weight)])
    output_bias = Constant(output_bias)
    print('output bias: ', output_bias)

    lr = 3e-3
    print('lr: ', lr)
    clipvalue = 0.6
    print('clipvalue: ', clipvalue)
    epoch = 1000
    print('epoch: ', epoch)
    batch_size = 256
    print('batch_size: ', batch_size)
    embedding_vector_length = 50
    print('embedding vector length :', embedding_vector_length)
    month_embedding_vector_length = 5
    print('month embedding vector length :', month_embedding_vector_length)
    return class_weight, output_bias, lr, clipvalue, epoch, batch_size, embedding_vector_length, month_embedding_vector_length

def choose_model(Xt_train, Xs_train, Xm_train, y_train, Xt_val, Xs_val, Xm_val, y_val, architecture, vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length, hp):
    if architecture == 'early':
        model = earlyFussion(Xt_train, Xs_train, Xm_train, y_train, Xt_val, Xs_val, Xm_val, y_val, vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length, hp)
    else:
        model = lateFussion(Xt_train, Xs_train, Xm_train, y_train, Xt_val, Xs_val, Xm_val, y_val, vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length, hp)
    return model

class MyHyperModel_basic(keras_tuner.HyperModel):
        
    def build(self, hp):
        
        target_outcome = 'new_12MonthsOutcome'
        max_codes = 150
        code2idx = pickle.load(open('../SeqModel/code2idx_all_big_08112023_75%.sav', 'rb'))
        month2idx = pickle.load(open('../SeqModel/month2idx_all_big_08112023_75%.sav', 'rb'))
        vocab_size = len(code2idx)
        month_size = len(month2idx)
        
        
        sets = pickle.load(open('../SeqModel/sets_search.sav', 'rb'))
        Xt_train, Xt_val, Xs_train, Xs_val, Xm_train, Xm_val, y_train, y_val = sets
        
        
        inputs1 = Input(shape=(Xt_train.shape[1],)) #tabular input
        #Dense layer for tabular data
        #dense0
        nn = Dense(units=hp.Int("units_dense0", min_value=64, max_value=256, step=64), 
                   activation=hp.Choice("activation_dense0", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_dense0", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense0", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense0", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(inputs1)
        nn = Dropout(hp.Float("dropout_dense0", min_value=0, max_value=0.5, step=0.1))(nn)
        
        #dense1
        nn = Dense(units=hp.Int("units_dense1", min_value=64, max_value=256, step=64), 
                   activation=hp.Choice("activation_dense1", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_dense1", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense1", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense1", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(nn)
        nn = Dropout(hp.Float("dropout_dense1", min_value=0, max_value=0.5, step=0.1))(nn)
        
        #dense2
        nn = Dense(units=hp.Int("units_dense2", min_value=64, max_value=256, step=64), 
                   activation=hp.Choice("activation_dense2", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_dense2", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense2", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense2", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(nn)
        nn = Dropout(hp.Float("dropout_dense2", min_value=0, max_value=0.5, step=0.1))(nn)
        
        #dense3
        nn = Dense(units=hp.Int("units_dense3", min_value=64, max_value=256, step=64), 
                   activation=hp.Choice("activation_dense3", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_dense3", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense3", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense3", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(nn)
        nn = Dropout(hp.Float("dropout_dense3", min_value=0, max_value=0.5, step=0.1))(nn)
        
        #dense4
        final_units = hp.Int("final_units", min_value=32, max_value=64, step=16)
        nn = Dense(units=final_units, 
                   activation=hp.Choice("activation_dense4", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_dense4", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense4", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense4", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(nn)
        nn = Dropout(hp.Float("dropout_dense4", min_value=0, max_value=0.5, step=0.1))(nn)
        
        nn = Reshape((1, final_units))(nn) #for concatenation

        #Embedding and LSTM for sequence data - clinical + therapy
        inputs2 = Input(shape=(Xs_train.shape[1],))

        embedding = Embedding(vocab_size, 
                              output_dim = hp.Int("embedding_vector_length", min_value=50, max_value=100, step=10), 
                              input_length=max_codes
                             )(inputs2)
        #lstm clinical 0 
        lstmClinical = LSTM(units=hp.Int("lstm_clinical_units0", min_value=16, max_value=64, step=16), 
                            return_sequences=True, 
                            kernel_regularizer=L1L2(l1=hp.Float("l1_lstmClinical0", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmClinical0", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                           )(embedding)
        lstmClinical = Dropout(hp.Float("dropout_lstmclinical0", min_value=0, max_value=0.5, step=0.1))(lstmClinical)
        
        #lstm clinical 1 
        lstmClinical = LSTM(units=hp.Int("lstm_clinical_units1", min_value=16, max_value=64, step=16), 
                            return_sequences=True, 
                            kernel_regularizer=L1L2(l1=hp.Float("l1_lstmClinical1", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmClinical1", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                           )(lstmClinical)
        lstmClinical = Dropout(hp.Float("dropout_lstmclinical1", min_value=0, max_value=0.5, step=0.1))(lstmClinical)
        
        #lstm clinical 2 
        lstmClinical = LSTM(units=hp.Int("lstm_clinical_units2", min_value=16, max_value=64, step=16), 
                            return_sequences=True, 
                            kernel_regularizer=L1L2(l1=hp.Float("l1_lstmClinical2", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmClinical2", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                           )(lstmClinical)
        lstmClinical = Dropout(hp.Float("dropout_lstmclinical2", min_value=0, max_value=0.5, step=0.1))(lstmClinical)
        
        #lstm clinical 3 
        lstmClinical = LSTM(units=final_units, 
                            return_sequences=True, 
                            kernel_regularizer=L1L2(l1=hp.Float("l1_lstmClinical3", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmClinical3", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                           )(lstmClinical)
        # lstmClinical = Dropout(hp.Float("dropout_lstmclinical3", min_value=0, max_value=0.5, step=0.1))(lstmClinical)
        
        #Embedding and LSTM for sequence data - month
        inputs3 = Input(shape=(Xm_train.shape[1],))
        embedding_month = Embedding(month_size, 
                                    output_dim = hp.Int("embedding_month_length", min_value=3, max_value=7, step=1), 
                                    input_length=max_codes
                                   )(inputs3)
        
        #lstm month 0
        lstmMonth = LSTM(units=hp.Int("lstm_month_units0", min_value=16, max_value=64, step=16), 
                         return_sequences=True, 
                         kernel_regularizer=L1L2(l1=hp.Float("l1_lstmMonth0", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmMonth0", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                        )(embedding_month)
        lstmMonth = Dropout(hp.Float("dropout_lstmmonth0", min_value=0, max_value=0.5, step=0.1))(lstmMonth)
        
        #lstm month 1
        lstmMonth = LSTM(units=hp.Int("lstm_month_units1", min_value=16, max_value=64, step=16), 
                         return_sequences=True, 
                         kernel_regularizer=L1L2(l1=hp.Float("l1_lstmMonth1", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmMonth1", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                        )(lstmMonth)
        lstmMonth = Dropout(hp.Float("dropout_lstmmonth1", min_value=0, max_value=0.5, step=0.1))(lstmMonth)
        
        #lstm month 2
        lstmMonth = LSTM(units=hp.Int("lstm_month_units2", min_value=16, max_value=64, step=16), 
                         return_sequences=True, 
                         kernel_regularizer=L1L2(l1=hp.Float("l1_lstmMonth2", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmMonth2", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                        )(lstmMonth)
        lstmMonth = Dropout(hp.Float("dropout_lstmmonth2", min_value=0, max_value=0.5, step=0.1))(lstmMonth)
        
        #lstm month 3
        lstmMonth = LSTM(units=final_units, 
                         return_sequences=True, 
                         kernel_regularizer=L1L2(l1=hp.Float("l1_lstmMonth3", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmMonth3", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                        )(lstmMonth)
        # lstmMonth = Dropout(hp.Float("dropout_lstmmonth3", min_value=0, max_value=0.5, step=0.1))(lstmMonth)
        
        #multiply lstm clinical and month
        lstm = Multiply()([lstmClinical, lstmMonth]) #merge two Embedding
        model_tot = concatenate([nn, lstm], axis=1)
        # model_tot = BatchNormalization()(model_tot)
        neurons_layerFinal = hp.Int("units_layer_final", min_value=32, max_value=64, step=32)
        model_tot = Dense(units=neurons_layerFinal, 
                   activation=hp.Choice("activation_dense4", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_dense4", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense4", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense4", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(model_tot)
        model_tot = Flatten()(model_tot)
        output = Dense(1, activation='sigmoid')(model_tot)

        opt = Adamax(learning_rate=hp.Choice("lr", [1e-3, 5e-4, 1e-4] ), 
                       clipvalue=hp.Float("clipvalue", min_value=0.3, max_value=0.7, step=0.2))

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='loss')

        metrics = [
            AUC(num_thresholds=3, name='auc', curve='ROC'),
            tf.keras.metrics.Precision(name='prec'),
            tf.keras.metrics.Recall(name='rec'),
            tf.keras.metrics.TrueNegatives(name='TN'),
            tf.keras.metrics.TruePositives(name='TP'),
            tf.keras.metrics.PrecisionAtRecall(0.8)
        ]

        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=output)
        print(model.summary())
        model.compile(
            loss=loss, 
            optimizer=opt, 
            metrics=metrics)
        return model
      
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )

class MyHyperModel_late(keras_tuner.HyperModel):
        
    def build(self, hp):
        
        target_outcome = '12months'
        max_codes = 150
        code2idx = pickle.load(open('../SeqModel/code2idx_therapy_long22112023.sav', 'rb'))
        month2idx = pickle.load(open('../SeqModel/month2idx_therapy_long22112023.sav', 'rb'))
        vocab_size = len(code2idx)
        month_size = len(month2idx)
        
        
        sets = pickle.load(open('../SeqModel/sets_search_therapy.sav', 'rb'))
        Xt_train, Xt_val, Xs_train, Xs_val, Xm_train, Xm_val, y_train, y_val = sets
        
        
        inputs1 = Input(shape=(Xt_train.shape[1],)) #tabular input
        #Dense layer for tabular data
        #dense0
        nn = Dense(units=hp.Int("units_dense0", min_value=64, max_value=256, step=64), 
                   activation=hp.Choice("activation_dense0", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_dense0", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense0", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense0", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(inputs1)
        nn = Dropout(hp.Float("dropout_dense0", min_value=0, max_value=0.5, step=0.1))(nn)
        
        #dense1
        nn = Dense(units=hp.Int("units_dense1", min_value=64, max_value=256, step=64), 
                   activation=hp.Choice("activation_dense1", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_dense1", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense1", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense1", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(nn)
        nn = Dropout(hp.Float("dropout_dense1", min_value=0, max_value=0.5, step=0.1))(nn)
        
        #dense2
        nn = Dense(units=hp.Int("units_dense2", min_value=64, max_value=256, step=64), 
                   activation=hp.Choice("activation_dense2", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_dense2", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense2", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense2", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(nn)
        nn = Dropout(hp.Float("dropout_dense2", min_value=0, max_value=0.5, step=0.1))(nn)
        
        #dense3
        nn = Dense(units=hp.Int("units_dense3", min_value=64, max_value=256, step=64), 
                   activation=hp.Choice("activation_dense3", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_dense3", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense3", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense3", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(nn)
        nn = Dropout(hp.Float("dropout_dense3", min_value=0, max_value=0.5, step=0.1))(nn)
        
        #dense4
        final_units = hp.Int("final_units", min_value=32, max_value=64, step=16)
        nn = Dense(units=final_units, 
                   activation=hp.Choice("activation_dense4", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_dense4", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense4", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense4", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(nn)
        nn = Dropout(hp.Float("dropout_dense4", min_value=0, max_value=0.5, step=0.1))(nn)
        
        nn = Reshape((1, final_units))(nn) #for concatenation

        #Embedding and LSTM for sequence data - clinical + therapy
        inputs2 = Input(shape=(Xs_train.shape[1],))

        embedding = Embedding(vocab_size, 
                              output_dim = hp.Int("embedding_vector_length", min_value=50, max_value=100, step=10), 
                              input_length=max_codes
                             )(inputs2)
        #lstm clinical 0 
        lstmClinical = LSTM(units=hp.Int("lstm_clinical_units0", min_value=16, max_value=64, step=16), 
                            return_sequences=True, 
                            kernel_regularizer=L1L2(l1=hp.Float("l1_lstmClinical0", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmClinical0", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                           )(embedding)
        lstmClinical = Dropout(hp.Float("dropout_lstmclinical0", min_value=0, max_value=0.5, step=0.1))(lstmClinical)
        
        #lstm clinical 1 
        lstmClinical = LSTM(units=hp.Int("lstm_clinical_units1", min_value=16, max_value=64, step=16), 
                            return_sequences=True, 
                            kernel_regularizer=L1L2(l1=hp.Float("l1_lstmClinical1", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmClinical1", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                           )(lstmClinical)
        lstmClinical = Dropout(hp.Float("dropout_lstmclinical1", min_value=0, max_value=0.5, step=0.1))(lstmClinical)
        
        #lstm clinical 2 
        lstmClinical = LSTM(units=hp.Int("lstm_clinical_units2", min_value=16, max_value=64, step=16), 
                            return_sequences=True, 
                            kernel_regularizer=L1L2(l1=hp.Float("l1_lstmClinical2", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmClinical2", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                           )(lstmClinical)
        lstmClinical = Dropout(hp.Float("dropout_lstmclinical2", min_value=0, max_value=0.5, step=0.1))(lstmClinical)
        
        #lstm clinical 3 
        lstmClinical = LSTM(units=final_units, 
                            return_sequences=True, 
                            kernel_regularizer=L1L2(l1=hp.Float("l1_lstmClinical3", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmClinical3", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                           )(lstmClinical)
        # lstmClinical = Dropout(hp.Float("dropout_lstmclinical3", min_value=0, max_value=0.5, step=0.1))(lstmClinical)
        
        #Embedding and LSTM for sequence data - month
        inputs3 = Input(shape=(Xm_train.shape[1],))
        embedding_month = Embedding(month_size, 
                                    output_dim = hp.Int("embedding_month_length", min_value=3, max_value=7, step=1), 
                                    input_length=max_codes
                                   )(inputs3)
        
        #lstm month 0
        lstmMonth = LSTM(units=hp.Int("lstm_month_units0", min_value=16, max_value=64, step=16), 
                         return_sequences=True, 
                         kernel_regularizer=L1L2(l1=hp.Float("l1_lstmMonth0", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmMonth0", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                        )(embedding_month)
        lstmMonth = Dropout(hp.Float("dropout_lstmmonth0", min_value=0, max_value=0.5, step=0.1))(lstmMonth)
        
        #lstm month 1
        lstmMonth = LSTM(units=hp.Int("lstm_month_units1", min_value=16, max_value=64, step=16), 
                         return_sequences=True, 
                         kernel_regularizer=L1L2(l1=hp.Float("l1_lstmMonth1", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmMonth1", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                        )(lstmMonth)
        lstmMonth = Dropout(hp.Float("dropout_lstmmonth1", min_value=0, max_value=0.5, step=0.1))(lstmMonth)
        
        #lstm month 2
        lstmMonth = LSTM(units=hp.Int("lstm_month_units2", min_value=16, max_value=64, step=16), 
                         return_sequences=True, 
                         kernel_regularizer=L1L2(l1=hp.Float("l1_lstmMonth2", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmMonth2", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                        )(lstmMonth)
        lstmMonth = Dropout(hp.Float("dropout_lstmmonth2", min_value=0, max_value=0.5, step=0.1))(lstmMonth)
        
        #lstm month 3
        lstmMonth = LSTM(units=final_units, 
                         return_sequences=True, 
                         kernel_regularizer=L1L2(l1=hp.Float("l1_lstmMonth3", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmMonth3", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                        )(lstmMonth)
        # lstmMonth = Dropout(hp.Float("dropout_lstmmonth3", min_value=0, max_value=0.5, step=0.1))(lstmMonth)
        
        #multiply lstm clinical and month
        lstm = Multiply()([lstmClinical, lstmMonth]) #merge two Embedding
        model_tot = concatenate([nn, lstm], axis=1)
        # model_tot = BatchNormalization()(model_tot)
        neurons_layerFinal = hp.Int("units_layer_final", min_value=32, max_value=64, step=32)
        model_tot = Dense(units=neurons_layerFinal, 
                   activation=hp.Choice("activation_dense4", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_dense4", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense4", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense4", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(model_tot)
        model_tot = Flatten()(model_tot)
        output = Dense(1, activation='sigmoid')(model_tot)

        opt = Adamax(learning_rate=hp.Choice("lr", [1e-3, 5e-4, 1e-4] ), 
                       clipvalue=hp.Float("clipvalue", min_value=0.3, max_value=0.7, step=0.2))

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='loss')

        metrics = [
            AUC(num_thresholds=3, name='auc', curve='ROC'),
            tf.keras.metrics.Precision(name='prec'),
            tf.keras.metrics.Recall(name='rec'),
            tf.keras.metrics.TrueNegatives(name='TN'),
            tf.keras.metrics.TruePositives(name='TP'),
            tf.keras.metrics.PrecisionAtRecall(0.8)
        ]

        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=output)
        print(model.summary())
        model.compile(
            loss=loss, 
            optimizer=opt, 
            metrics=metrics)
        return model
      
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )

class MyHyperModel_early(keras_tuner.HyperModel):
        
    def build(self, hp):
        
        max_codes = 500
        code2idx = pickle.load(open('../SeqModel/all_vocab.sav', 'rb'))
        month2idx = pickle.load(open('../SeqModel/all_vocab_month.sav', 'rb'))
        vocab_size = len(code2idx)
        month_size = len(month2idx)
        
        
        sets = pickle.load(open('../SeqModel/sets_search_long.sav', 'rb'))
        Xt_train, Xt_val, Xs_train, Xs_val, Xm_train, Xm_val, y_train, y_val = sets
        
        inputs1 = Input(shape=(Xt_train.shape[1],))
        print(inputs1.shape)
        inputs2 = Input(shape=(Xs_train.shape[1],))
        print(inputs2.shape)
        inputs3 = Input(shape=(Xm_train.shape[1],))
        print(inputs3.shape)
        
               

        #LAYER 0 
        #demography
        neurons_layer0 = hp.Int("units_layer0", min_value=32, max_value=128, step=64)

        nn = Dense(units=neurons_layer0, 
                   activation=hp.Choice("activation_layer0", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_layer0", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense0", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense0", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(inputs1)
        nn = Dropout(hp.Float("dropout_dense0", min_value=0, max_value=0.5, step=0.1))(nn)

        nn = Reshape((1, neurons_layer0))(nn)

        #clinical embedding for lstm
        lstm_units0 = hp.Int("lstm_clinical_units", min_value=16, max_value=64, step=16)
        embedding = Embedding(vocab_size, 
                              output_dim = hp.Int("embedding_vector_length", min_value=50, max_value=100, step=10), 
                              input_length=max_codes
                             )(inputs2)
        lstmClinical = LSTM(units=lstm_units0, 
                            return_sequences=True, 
                            kernel_regularizer=L1L2(l1=hp.Float("l1_lstmClinical", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmClinical", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                           )(embedding)

        #month embedding for lstm
        embedding_month = Embedding(month_size, 
                                    output_dim = hp.Int("embedding_month_length", min_value=3, max_value=7, step=1), 
                                    input_length=max_codes
                                   )(inputs3)
        lstmMonth = LSTM(units=lstm_units0, 
                         return_sequences=True, 
                         kernel_regularizer=L1L2(l1=hp.Float("l1_lstmMonth", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmMonth", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                        )(embedding_month)

        lstm = Multiply()([lstmClinical, lstmMonth]) #merge two Embedding
        lstm = LSTM(units=neurons_layer0, return_sequences=True, 
                    kernel_regularizer=L1L2(l1=hp.Float("l1_lstm0", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstm0", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                   )(lstm)
        lstm = Dropout(hp.Float("dropout_lstm0", min_value=0, max_value=0.5, step=0.1))(lstm)

        #LAYER 1
        neurons_layer1 = hp.Int("units_layer1", min_value=64, max_value=128, step=32)

        add = concatenate([nn, lstm], axis=1) #concat LSTM and dense 
        nn = Dense(units=neurons_layer1, 
                   activation=hp.Choice("activation_layer1", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_layer1", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense1", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense1", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(add)
        nn = Dropout(hp.Float("dropout_dense1", min_value=0, max_value=0.5, step=0.1))(nn)
        lstm = LSTM(units=neurons_layer1, return_sequences=True, 
                    kernel_regularizer=L1L2(l1=hp.Float("l1_lstm1", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstm1", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                   )(lstm)
        lstm = Dropout(hp.Float("dropout_lstm1", min_value=0, max_value=0.5, step=0.1))(lstm)
        
        #LAYER 2
        neurons_layer2 = hp.Int("units_layer2", min_value=64, max_value=128, step=32)

        add = concatenate([nn, lstm], axis=1) #concat LSTM and dense 
        nn = Dense(units=neurons_layer2, 
                   activation=hp.Choice("activation_layer2", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_layer2", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense2", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense2", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(add)
        nn = Dropout(hp.Float("dropout_dense2", min_value=0, max_value=0.5, step=0.1))(nn)
        lstm = LSTM(units=neurons_layer2, return_sequences=True, 
                    kernel_regularizer=L1L2(l1=hp.Float("l1_lstm2", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstm2", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                   )(lstm)
        lstm = Dropout(hp.Float("dropout_lstm2", min_value=0, max_value=0.5, step=0.1))(lstm)
        
        #LAYER 3
        neurons_layer3 = hp.Int("units_layer3", min_value=64, max_value=128, step=32)

        add = concatenate([nn, lstm], axis=1) #concat LSTM and dense 
        nn = Dense(units=neurons_layer3, 
                   activation=hp.Choice("activation_layer3", ["relu", "elu"]), 
                   kernel_initializer=hp.Choice("kernel_initializer_layer3", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense3", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense3", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(add)
        nn = Dropout(hp.Float("dropout_dense3", min_value=0, max_value=0.5, step=0.1))(nn)
        lstm = LSTM(units=neurons_layer3, return_sequences=True, 
                    kernel_regularizer=L1L2(l1=hp.Float("l1_lstm3", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstm3", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                   )(lstm)
        lstm = Dropout(hp.Float("dropout_lstm3", min_value=0, max_value=0.5, step=0.1))(lstm)

        #LAYER FINAL CONCAT
        neurons_layerFinal = hp.Int("units_layer_final", min_value=16, max_value=64, step=16)
        model_tot = concatenate([nn, lstm], axis=1)
        # model_tot = BatchNormalization()(model_tot)
        model_tot = Dense(units=neurons_layerFinal, 
                          activation=hp.Choice("activation_layerFinal", 
                                               ["relu", "elu"]),
                          kernel_regularizer=L1L2(l1=hp.Float("l1_final", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_final", min_value=0.0, max_value=0.1, step=0.02)
                                          ), 
                          kernel_initializer=hp.Choice("kernel_initializer_final", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]),
                         )(model_tot)

        model_tot = Flatten()(model_tot)
        output = Dense(1, activation='sigmoid')(model_tot)

        opt = Adadelta(learning_rate=hp.Choice("lr", [1e-3, 3e-4, 1e-4] ), 
                       clipvalue=hp.Float("clipvalue", min_value=0.3, max_value=0.7, step=0.2))

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='loss')

        metrics = [
            AUC(num_thresholds=3, name='auc', curve='ROC'),
            tf.keras.metrics.Precision(name='prec'),
            tf.keras.metrics.Recall(name='rec'),
            tf.keras.metrics.TrueNegatives(name='TN'),
            tf.keras.metrics.TruePositives(name='TP'),
            tf.keras.metrics.PrecisionAtRecall(0.8)
        ]

        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=output)
        print(model.summary())
        model.compile(
            loss=loss, 
            optimizer=opt, 
            metrics=metrics)
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )
    
class MyHyperModel_early_leakyReLU(keras_tuner.HyperModel):
        
    def build(self, hp):
        
        target_outcome = 'new_12MonthsOutcome'
        max_codes = 150
        code2idx = pickle.load(open('../SeqModel/code2idx_all_big_08112023_75%.sav', 'rb'))
        month2idx = pickle.load(open('../SeqModel/month2idx_all_big_08112023_75%.sav', 'rb'))
        vocab_size = len(code2idx)
        month_size = len(month2idx)
        
        
        sets = pickle.load(open('../SeqModel/sets_search.sav', 'rb'))
        Xt_train, Xt_val, Xs_train, Xs_val, Xm_train, Xm_val, y_train, y_val = sets
        
        inputs1 = Input(shape=(Xt_train.shape[1],))
        inputs2 = Input(shape=(Xs_train.shape[1],))
        inputs3 = Input(shape=(Xm_train.shape[1],))

        #LAYER 0 
        #demography
        neurons_layer0 = hp.Int("units_layer0", min_value=32, max_value=128, step=64)

        nn = Dense(units=neurons_layer0, 
                   activation= LeakyReLU(hp.Float("alpha", min_value=0.1, max_value=0.5, step=0.2)), 
                   kernel_initializer=hp.Choice("kernel_initializer_layer0", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense0", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense0", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(inputs1)
        nn = Dropout(hp.Float("dropout_dense0", min_value=0, max_value=0.5, step=0.1))(nn)

        nn = Reshape((1, neurons_layer0))(nn)

        #clinical embedding for lstm
        lstm_units0 = hp.Int("lstm_clinical_units", min_value=16, max_value=64, step=16)
        embedding = Embedding(vocab_size, 
                              output_dim = hp.Int("embedding_vector_length", min_value=50, max_value=100, step=10), 
                              input_length=max_codes
                             )(inputs2)
        lstmClinical = LSTM(units=lstm_units0, 
                            return_sequences=True, 
                            kernel_regularizer=L1L2(l1=hp.Float("l1_lstmClinical", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmClinical", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                           )(embedding)

        #month embedding for lstm
        embedding_month = Embedding(month_size, 
                                    output_dim = hp.Int("embedding_month_length", min_value=3, max_value=7, step=1), 
                                    input_length=max_codes
                                   )(inputs3)
        lstmMonth = LSTM(units=lstm_units0, 
                         return_sequences=True, 
                         kernel_regularizer=L1L2(l1=hp.Float("l1_lstmMonth", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstmMonth", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                        )(embedding_month)

        lstm = Add()([lstmClinical, lstmMonth]) #merge two Embedding
        lstm = LSTM(units=neurons_layer0, return_sequences=True, 
                    kernel_regularizer=L1L2(l1=hp.Float("l1_lstm0", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstm0", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                   )(lstm)
        lstm = Dropout(hp.Float("dropout_lstm0", min_value=0, max_value=0.5, step=0.1))(lstm)

        #LAYER 1
        neurons_layer1 = hp.Int("units_layer1", min_value=64, max_value=128, step=32)

        add = concatenate([nn, lstm], axis=1) #concat LSTM and dense 
        nn = Dense(units=neurons_layer1, 
                   activation= LeakyReLU(hp.Float("alpha", min_value=0.1, max_value=0.5, step=0.2)), 
                   kernel_initializer=hp.Choice("kernel_initializer_layer1", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense1", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense1", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(add)
        nn = Dropout(hp.Float("dropout_dense1", min_value=0, max_value=0.5, step=0.1))(nn)
        lstm = LSTM(units=neurons_layer1, return_sequences=True, 
                    kernel_regularizer=L1L2(l1=hp.Float("l1_lstm1", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstm1", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                   )(lstm)
        lstm = Dropout(hp.Float("dropout_lstm1", min_value=0, max_value=0.5, step=0.1))(lstm)
        
        #LAYER 2
        neurons_layer2 = hp.Int("units_layer2", min_value=64, max_value=128, step=32)

        add = concatenate([nn, lstm], axis=1) #concat LSTM and dense 
        nn = Dense(units=neurons_layer2, 
                   activation= LeakyReLU(hp.Float("alpha", min_value=0.1, max_value=0.5, step=0.2)), 
                   kernel_initializer=hp.Choice("kernel_initializer_layer2", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense2", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense2", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(add)
        nn = Dropout(hp.Float("dropout_dense2", min_value=0, max_value=0.5, step=0.1))(nn)
        lstm = LSTM(units=neurons_layer2, return_sequences=True, 
                    kernel_regularizer=L1L2(l1=hp.Float("l1_lstm2", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstm2", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                   )(lstm)
        lstm = Dropout(hp.Float("dropout_lstm2", min_value=0, max_value=0.5, step=0.1))(lstm)
        
        #LAYER 3
        neurons_layer3 = hp.Int("units_layer3", min_value=64, max_value=128, step=32)

        add = concatenate([nn, lstm], axis=1) #concat LSTM and dense 
        nn = Dense(units=neurons_layer3, 
                   activation= LeakyReLU(hp.Float("alpha", min_value=0.1, max_value=0.5, step=0.2)), 
                   kernel_initializer=hp.Choice("kernel_initializer_layer3", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                   kernel_regularizer=L1L2(l1=hp.Float("l1_dense3", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_dense3", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                  )(add)
        nn = Dropout(hp.Float("dropout_dense3", min_value=0, max_value=0.5, step=0.1))(nn)
        lstm = LSTM(units=neurons_layer3, return_sequences=True, 
                    kernel_regularizer=L1L2(l1=hp.Float("l1_lstm3", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_lstm3", min_value=0.0, max_value=0.1, step=0.02)
                                          )
                   )(lstm)
        lstm = Dropout(hp.Float("dropout_lstm3", min_value=0, max_value=0.5, step=0.1))(lstm)

        #LAYER FINAL CONCAT
        neurons_layerFinal = hp.Int("units_layer_final", min_value=16, max_value=64, step=16)
        model_tot = concatenate([nn, lstm], axis=1)
        model_tot = BatchNormalization()(model_tot)
        model_tot = Dense(units=neurons_layerFinal, 
                          activation= LeakyReLU(hp.Float("alpha", min_value=0.1, max_value=0.5, step=0.2)), 
                          kernel_regularizer=L1L2(l1=hp.Float("l1_final", min_value=0.0, max_value=0.1, step=0.02), 
                                           l2=hp.Float("l2_final", min_value=0.0, max_value=0.1, step=0.02)
                                          ), 
                          kernel_initializer=hp.Choice("kernel_initializer_final", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]),
                         )(model_tot)

        model_tot = Flatten()(model_tot)
        output = Dense(1, activation='sigmoid')(model_tot)

        opt = Adadelta(learning_rate=hp.Choice("lr", [1e-3, 3e-4, 1e-4] ), 
                       clipvalue=hp.Float("clipvalue", min_value=0.3, max_value=0.7, step=0.2))

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='loss')

        metrics = [
            AUC(num_thresholds=3, name='auc', curve='ROC'),
            tf.keras.metrics.Precision(name='prec'),
            tf.keras.metrics.Recall(name='rec'),
            tf.keras.metrics.TrueNegatives(name='TN'),
            tf.keras.metrics.TruePositives(name='TP'),
            tf.keras.metrics.PrecisionAtRecall(0.8)
        ]

        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=output)
        print(model.summary())
        model.compile(
            loss=loss, 
            optimizer=opt, 
            metrics=metrics)
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )

def search_model_parameters(method):
    print('################# Search parameters #########################')
    start_time = time.time()
    hp = keras_tuner.HyperParameters()
    if method == 'early':
        model = MyHyperModel_early()
    elif method == 'leaky_relu':
        model = MyHyperModel_early_leakyReLU()
    elif method == 'basic':
        model = MyHyperModel_basic()
    else:
        model = MyHyperModel_late()
    with tf.device('/GPU:0'):
        tuner = keras_tuner.Hyperband(
            hypermodel= model,
            objective=keras_tuner.Objective("val_auc", direction="max"),
            max_epochs=25,
            hyperband_iterations=1,
            overwrite=True,
            directory='../SeqModel/tuner/',
            project_name="LSTM_Asthma",
        )
        
        tuner.search_space_summary()
        
        sets = pickle.load(open('../SeqModel/sets_search_long.sav', 'rb'))
        Xt_train, Xt_val, Xs_train, Xs_val, Xm_train, Xm_val, y_train, y_val = sets
        
        pos_weight = sum(x == 0 for x in y_train)/sum(x == 1 for x in y_train)
        class_weight = {0:1, 1:pos_weight}
        earlyStopping = EarlyStopping(monitor='val_auc', patience=3, verbose=0, mode='max', restore_best_weights=True)
        
        tuner.search([Xt_train, Xs_train, Xm_train], y_train, 
                     validation_data=([Xt_val, Xs_val, Xm_val], y_val), 
                     epochs=10, 
                     batch_size = 128,
                     class_weight = class_weight,
                     callbacks = [earlyStopping])
    
    print("--- Training time: %s seconds ---" % (time.time() - start_time))
    return tuner

def evaluate_model(model, sets):
    for Xt, Xs, Xm, y in sets:
        with tf.device('/GPU:0'):
            model.evaluate([Xt, Xs, Xm], y, batch_size=300)