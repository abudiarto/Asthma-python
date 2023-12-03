import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, Input, concatenate, Reshape, Activation, Flatten, Add, BatchNormalization, Multiply
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.metrics import AUC, SensitivityAtSpecificity
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, Adamax, SGD, Adadelta
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import L1L2, L1, L2
from livelossplot import PlotLossesKeras

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

def lateFussion(Xt_train, Xs_train, Xm_train, y_train, Xt_val, Xs_val, Xm_val, y_val, vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length):
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

def earlyFussion(Xt_train, Xs_train, Xm_train, y_train, Xt_val, Xs_val, Xm_val, y_val, vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length):
       
    inputs1 = Input(shape=(Xt_train.shape[1],))
    inputs2 = Input(shape=(Xs_train.shape[1],))
    inputs3 = Input(shape=(Xm_train.shape[1],))
    
    #demography
    nn = Dense(64, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=L1L2(l1=0.0, l2=0.01))(inputs1)
    nn = Reshape((1, 64))(nn)
    
    #clinical embedding for lstm
    embedding = Embedding(vocab_size, embedding_vector_length, input_length=max_codes)(inputs2)
    lstmClinical = LSTM(units=64, return_sequences=True, kernel_regularizer=L1L2(l1=0.0, l2=0.01))(embedding)
    
    #month embedding for lstm
    embedding_month = Embedding(month_size, month_embedding_vector_length, input_length=max_codes)(inputs3)
    lstmMonth = LSTM(units=64, return_sequences=True, kernel_regularizer=L1L2(l1=0.0, l2=0.01))(embedding_month)
   
    lstm = Add()([lstmClinical, lstmMonth]) #merge two Embedding
    lstm = LSTM(units=64, return_sequences=True, kernel_regularizer=L1L2(l1=0.0, l2=0.01))(lstm)
    
    
    add = concatenate([nn, lstm], axis=1) #merge LSTM and dense 
    nn = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=L1L2(l1=0.0, l2=0.01))(add)
    lstm = LSTM(units=32, return_sequences=True, kernel_regularizer=L1L2(l1=0.0, l2=0.01))(lstm)
    
    add = concatenate([nn, lstm], axis=1) #merge LSTM and dense 
    nn = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=L1L2(l1=0.0, l2=0.01))(add)
    lstm = LSTM(units=32, return_sequences=True, kernel_regularizer=L1L2(l1=0.0, l2=0.01))(lstm)
    
    add = concatenate([nn, lstm], axis=1) #merge LSTM and dense 
    nn = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=L1L2(l1=0.0, l2=0.01))(add)
    lstm = LSTM(units=32, return_sequences=True, kernel_regularizer=L1L2(l1=0.0, l2=0.01))(lstm)
    

    model_tot = concatenate([nn, lstm], axis=1)
    model_tot = BatchNormalization()(model_tot)

    model_tot = Dense(units=32, activation='relu', kernel_regularizer=L2(0.01), 
                      kernel_initializer='glorot_uniform')(model_tot)
    
    model_tot = Flatten()(model_tot)
    output = Dense(1, activation='sigmoid')(model_tot)
    
    opt = Adadelta(learning_rate=lr, clipvalue=clipvalue)
    
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
    code2idx = pickle.load(open('../SeqModel/code2idx_all_big_08112023_75%.sav', 'rb'))
    idx2code = pickle.load(open('../SeqModel/idx2code_all_big_08112023_75%.sav', 'rb'))
    month2idx = pickle.load(open('../SeqModel/month2idx_all_big_08112023_75%.sav', 'rb'))
    idx2month = pickle.load(open('../SeqModel/idx2month_all_big_08112023_75%.sav', 'rb'))
    data = pickle.load(open('../SeqModel/data_all_big_08112023_75%.sav', 'rb'))
    print('-----------------we passed the heaviest part------------------------------')
    tabularData = pd.read_csv('../FinalData/cleaned_features_08112023.csv')
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
    data = pd.get_dummies(data, columns=['imd_decile', 'smokingStatus']) #one hot encoding
    tabularData = []
    
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

    Xs_train = np.array(trainingData.read_code_seq_padded_idx.values)
    Xs_train = np.array([x for x in Xs_train])
    Xs_val = np.array(valData.read_code_seq_padded_idx.values)
    Xs_val = np.array([x for x in Xs_val])
    Xs_eval = np.array(evalData.read_code_seq_padded_idx.values)
    Xs_eval = np.array([x for x in Xs_eval])
    Xs_test = np.array(testData.read_code_seq_padded_idx.values)
    Xs_test = np.array([x for x in Xs_test])
    Xs_testWales = np.array(testDataWales.read_code_seq_padded_idx.values)
    Xs_testWales = np.array([x for x in Xs_testWales])
    Xs_testScotland = np.array(testDataScotland.read_code_seq_padded_idx.values)
    Xs_testScotland = np.array([x for x in Xs_testScotland])

    Xm_train = np.array(trainingData.month_padded_idx.values)
    Xm_train = np.array([x for x in Xm_train])
    Xm_val = np.array(valData.month_padded_idx.values)
    Xm_val = np.array([x for x in Xm_val])
    Xm_eval = np.array(evalData.month_padded_idx.values)
    Xm_eval = np.array([x for x in Xm_eval])
    Xm_test = np.array(testData.month_padded_idx.values)
    Xm_test = np.array([x for x in Xm_test])
    Xm_testWales = np.array(testDataWales.month_padded_idx.values)
    Xm_testWales = np.array([x for x in Xm_testWales])
    Xm_testScotland = np.array(testDataScotland.month_padded_idx.values)
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

def choose_model(Xt_train, Xs_train, Xm_train, y_train, Xt_val, Xs_val, Xm_val, y_val, architecture, vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length):
    if architecture == 'early':
        model = earlyFussion(Xt_train, Xs_train, Xm_train, y_train, Xt_val, Xs_val, Xm_val, y_val, vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length)
    else:
        model = lateFussion(Xt_train, Xs_train, Xm_train, y_train, Xt_val, Xs_val, Xm_val, y_val, vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length)
    return model

def train_model(Xt_train, Xs_train, Xm_train, y_train, Xt_val, Xs_val, Xm_val, y_val, architecture, epoch, batch_size, class_weight, vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length):
    print('################# Define model #########################')
    model = choose_model(Xt_train, Xs_train, Xm_train, y_train, Xt_val, Xs_val, Xm_val, y_val, architecture, vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length)
    print(model.summary())
    print('################# Train model #########################')
    start_time = time.time()
    with tf.device('/GPU:0'):
        earlyStopping = EarlyStopping(monitor='val_auc', patience=8, verbose=0, mode='max', restore_best_weights=True)
        mcp_save = ModelCheckpoint('../SeqModel/seqModel_all_tabSeq.mdl_wts.hdf5', save_best_only=True, monitor='val_auc', mode='min')
        log_dir = "../SeqModel/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit([Xt_train, Xs_train, Xm_train], y_train, validation_data=([Xt_val, Xs_val, Xm_val], y_val), 
                                epochs=epoch, batch_size=batch_size, 
                            class_weight = class_weight, 
                            callbacks = [earlyStopping, mcp_save, tensorboard_callback])
    print("--- Training time: %s seconds ---" % (time.time() - start_time))
    model.save('../SeqModel/model_all_tabSeq.h5')
    pickle.dump(history, open('../SeqModel/historya_ll_tabSeq.sav', 'wb'))
    return model, history

def evaluate_model(model, sets):
    for Xt, Xs, Xm, y in sets:
        with tf.device('/GPU:0'):
            model.evaluate([Xt, Xs, Xm], y, batch_size=300)