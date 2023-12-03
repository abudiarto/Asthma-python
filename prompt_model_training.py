import model_helper
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

target_outcome = 'new_12MonthsOutcome'
max_codes = 300
params = {'lr': 1e-3,
          'clipvalue': 0.6,
          'epoch': 1000,
          'batch_size' : 256,
          'embedding_vector_length' : 50,
          'month_embedding_vector_length' : 5
         }

#split data
trainingData, valData, evalData, testData, testDataWales, testDataScotland, vocab_size, month_size = model_helper.load_data(target_outcome)

#generate X and y
Xt_train, Xt_val, Xt_eval, Xt_test, Xt_testWales, Xt_testScotland, Xs_train, Xs_val, Xs_eval, Xs_test, Xs_testWales, Xs_testScotland, Xm_train, Xm_val, Xm_eval, Xm_test, Xm_testWales, Xm_testScotland, y_train, y_val, y_eval, y_test, y_testWales, y_testScotland = model_helper.generate_X_y(trainingData, valData, evalData, testData, testDataWales, testDataScotland, target_outcome)

#set parameter
class_weight, output_bias, lr, clipvalue, epoch, batch_size, embedding_vector_length, month_embedding_vector_length = model_helper.set_parameters(trainingData, target_outcome)


#define model
# model = model_helper.choose_model('early', vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length)

#train model
model, history = model_helper.train_model(Xt_train, Xs_train, Xm_train, y_train, Xt_val, Xs_val, Xm_val, y_val, epoch, batch_size, class_weight, vocab_size, month_size, max_codes, lr, clipvalue, embedding_vector_length, month_embedding_vector_length)

#model evaluation
model_helper.evaluate_model(model, sets)