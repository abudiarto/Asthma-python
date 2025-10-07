import pandas as pd
import pyreadr
import numpy as np
# import torch
import pickle
from os import listdir
from os.path import isfile, join


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


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

random_state = 42
tf.random.set_seed(random_state)


target_outcomes = [
    'outcome_12months',
    'outcome_3months', 'outcome_6months', 'outcome_9months',   
] 

for target_outcome in target_outcomes:
    print(target_outcome)
    gridSearchData, crossValData, internalEvaluationData, externalEvaluationData = pickle.load(open('../../Clean_data/dataset_scaled_ordinal.sav', 'rb'))
    gridSearch_seq, crossVal_seq, internalEvaluation_seq, externalEvaluation_seq = pickle.load(open('../../Clean_data/seasonal_long_dataset_28102024.sav', 'rb'))
    
    gridSearch_seq = gridSearchData[['patid', target_outcome]].merge(gridSearch_seq, on = 'patid', how='inner').fillna(0).drop_duplicates('patid').reset_index(drop=True)
    crossVal_seq = crossValData[['patid', target_outcome]].merge(crossVal_seq, on = 'patid', how='inner').fillna(0).drop_duplicates('patid').reset_index(drop=True)
    internalEvaluation_seq = internalEvaluationData[['patid', target_outcome]].merge(internalEvaluation_seq, on = 'patid', how='inner').fillna(0).drop_duplicates('patid').reset_index(drop=True)
    externalEvaluation_seq = externalEvaluationData[['patid', target_outcome]].merge(externalEvaluation_seq, on = 'patid', how='inner').fillna(0).drop_duplicates('patid').reset_index(drop=True)

    #tabular vectors
    tabular_vars = ['sex', 'rhinitis', 'cardiovascular', 'heartfailure', 'psoriasis', 'anaphylaxis', 'diabetes', 'ihd', 'anxiety', 'eczema', 
                    'nasalpolyps', 'ethnic_group_Asian', 'ethnic_group_Black', 'ethnic_group_Mixed',
                    'ethnic_group_Other', 'ethnic_group_White', 'ethnic_group_not recorded',
                    'smokingStatus_current', 'smokingStatus_former', 'smokingStatus_never',
                    'DeviceType_BAI', 'DeviceType_DPI', 'DeviceType_NEB', 'DeviceType_not recorded', 'DeviceType_pMDI', 
                    'PriorEducation_No', 'PriorEducation_Yes',  'BMI_cat',  'imd_decile', 'CharlsonScore', 'PEFStatus', 
                    'EosinophilLevel', 'BTS_step','age', 
                    'numAsthmaManagement', 'numAsthmaReview', 'ICS_medication_possesion_ratio',
                    ]
    
    # train_tab = trainingData[tabular_vars]
    # val_tab = validationData[tabular_vars]
    # test_tab = evaluationData[tabular_vars]
    gridSearch_tab = gridSearchData.merge(gridSearch_seq[['patid']], on='patid', how='inner').reset_index(drop=True)[tabular_vars]
    crossVal_tab = crossValData.merge(crossVal_seq[['patid']], on='patid', how='inner').reset_index(drop=True)[tabular_vars]
    internalEvaluation_tab = internalEvaluationData.merge(internalEvaluation_seq[['patid']], on='patid', how='inner').reset_index(drop=True)[tabular_vars]
    externalEvaluation_tab = externalEvaluationData.merge(externalEvaluation_seq[['patid']], on='patid', how='inner').reset_index(drop=True)[tabular_vars]
    
    gridSearch_tab = np.asarray(gridSearch_tab).astype('float32')
    crossVal_tab = np.asarray(crossVal_tab).astype('float32')
    internalEvaluation_tab = np.asarray(internalEvaluation_tab).astype('float32')
    externalEvaluation_tab = np.asarray(externalEvaluation_tab).astype('float32')

    # training_seq.index = training_seq.patid
    gridSearch_X = gridSearch_seq.iloc[:,2:]
    gridSearch_y = gridSearch_seq[target_outcome]
    
    # val_seq.index = val_seq.patid
    crossVal_X = crossVal_seq.iloc[:,2:]
    crossVal_y = crossVal_seq[target_outcome]
    
    # internalEvaluation_seq.index = internalEvaluation_seq.patid
    internalEvaluation_X = internalEvaluation_seq.iloc[:,2:]
    internalEvaluation_y = internalEvaluation_seq[target_outcome]
    
    # internalEvaluation_seq.index = internalEvaluation_seq.patid
    externalEvaluation_X = externalEvaluation_seq.iloc[:,2:]
    externalEvaluation_y = externalEvaluation_seq[target_outcome]

    print(gridSearch_y.value_counts(normalize=True))
    print(crossVal_y.value_counts(normalize=True))
    print(internalEvaluation_y.value_counts(normalize=True))
    print(externalEvaluation_y.value_counts(normalize=True))


    #sequence vectors
    # train_X = train_X.values
    # val_X = val_X.values
    # test_X = test_X.values
    
    gridSearch_X = gridSearch_X.to_numpy(dtype='float16')
    crossVal_X = crossVal_X.to_numpy(dtype='float16')
    internalEvaluation_X = internalEvaluation_X.to_numpy(dtype='float16')
    externalEvaluation_X = externalEvaluation_X.to_numpy(dtype='float16')
    
    
    
    #outcome vectors
    gridSearch_y = gridSearch_y.values
    crossVal_y = crossVal_y.values
    internalEvaluation_y = internalEvaluation_y.values
    externalEvaluation_y = externalEvaluation_y.values
    
    #reshape into 3D vectors
    num_of_months = 41
    num_of_features = 11
    gridSearch_X = gridSearch_X.reshape((gridSearch_X.shape[0], num_of_months, num_of_features))
    crossVal_X = crossVal_X.reshape((crossVal_X.shape[0], num_of_months, num_of_features))
    internalEvaluation_X = internalEvaluation_X.reshape((internalEvaluation_X.shape[0], num_of_months, num_of_features))
    externalEvaluation_X = externalEvaluation_X.reshape((externalEvaluation_X.shape[0], num_of_months, num_of_features))
    
    gridSearch_X = np.asarray(gridSearch_X).astype('float32')
    crossVal_X = np.asarray(crossVal_X).astype('float32')
    internalEvaluation_X = np.asarray(internalEvaluation_X).astype('float32')
    externalEvaluation_X = np.asarray(externalEvaluation_X).astype('float32')
    
    print('sequence input: ', gridSearch_X.shape, crossVal_X.shape, internalEvaluation_X.shape, externalEvaluation_X.shape)
    print('tabular input: ', gridSearch_tab.shape, crossVal_tab.shape, internalEvaluation_tab.shape, externalEvaluation_tab.shape)
    print('outcome: ', gridSearch_y.shape, crossVal_y.shape, internalEvaluation_y.shape, externalEvaluation_X.shape)

    datasets = [gridSearch_X, crossVal_X, internalEvaluation_X, externalEvaluation_X,
                gridSearch_tab, crossVal_tab, internalEvaluation_tab, externalEvaluation_tab,
                gridSearch_y, crossVal_y, internalEvaluation_y, externalEvaluation_y]
    pickle.dump(datasets, open('../../Clean_data/seasonal_dataset_ordinal_'+target_outcome+'.sav', 'wb'))



    
    
    
    
    
    
    
    
    
    
    
