import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, Input, concatenate, Reshape, Activation, Flatten, Add, BatchNormalization, Multiply, LeakyReLU, Conv1D
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.metrics import AUC, SensitivityAtSpecificity
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, Adamax, SGD, Adadelta
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import L1L2, L1, L2
from livelossplot import PlotLossesKeras

import keras_tuner

#internal validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score, matthews_corrcoef, auc, average_precision_score, roc_auc_score, balanced_accuracy_score, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
import time
# fix random seed for reproducibility
tf.random.set_seed(1234)

# Data loader
trainingData, validationData, internalEvaluationData, evaluationData, evaluationDataWales, evaluationDataScotland = pickle.load(open('../data/dataset_scaled_2vs1_09122023.sav', 'rb'))

trainingData = trainingData[(trainingData.age >=8) & (trainingData.age <=80)]
validationData = validationData[(validationData.age >=8) & (validationData.age <=80)]
internalEvaluationData = internalEvaluationData[(internalEvaluationData.age >=8) & (internalEvaluationData.age <=80)]
evaluationData = evaluationData[(evaluationData.age >=8) & (evaluationData.age <=80)]
evaluationDataWales = evaluationDataWales[(evaluationDataWales.age >=8) & (evaluationDataWales.age <=80)]
evaluationDataScotland = evaluationDataScotland[(evaluationDataScotland.age >=8) & (evaluationDataScotland.age <=80)]

trainingData = trainingData.rename({'3MonthsOutcome': '3months', '6MonthsOutcome': '6months','9MonthsOutcome': '9months','12MonthsOutcome': '12months',}, axis=1)
validationData = validationData.rename({'3MonthsOutcome': '3months', '6MonthsOutcome': '6months','9MonthsOutcome': '9months','12MonthsOutcome': '12months',}, axis=1)
internalEvaluationData = internalEvaluationData.rename({'3MonthsOutcome': '3months', '6MonthsOutcome': '6months','9MonthsOutcome': '9months','12MonthsOutcome': '12months',}, axis=1)
evaluationData = evaluationData.rename({'3MonthsOutcome': '3months', '6MonthsOutcome': '6months','9MonthsOutcome': '9months','12MonthsOutcome': '12months',}, axis=1)
evaluationDataWales = evaluationDataWales.rename({'3MonthsOutcome': '3months', '6MonthsOutcome': '6months','9MonthsOutcome': '9months','12MonthsOutcome': '12months',}, axis=1)
evaluationDataScotland = evaluationDataScotland.rename({'3MonthsOutcome': '3months', '6MonthsOutcome': '6months','9MonthsOutcome': '9months','12MonthsOutcome': '12months',}, axis=1)

X = trainingData[features_columns]
X_val = validationData[features_columns]

X_internaleval = internalEvaluationData[features_columns]
X_eval = evaluationData[features_columns]
X_eval_Wales = evaluationDataWales[features_columns]
X_eval_Scotland = evaluationDataScotland[features_columns]

print(X.shape)
print(X_val.shape)
print(X_eval.shape)
print(X_eval_Wales.shape)
print(X_eval_Scotland.shape)

target_outcomes = ['12months'] 

%%time

# create the model
y = trainingData[target_outcomes[0]].values
y_val = validationData[target_outcomes[0]].values
earlyStopping = EarlyStopping(monitor='val_auc', patience=10, verbose=0, mode='max', restore_best_weights=True)
mcp_save = ModelCheckpoint('../SeqModel/seqModel_FCN.mdl_wts.hdf5', save_best_only=True, monitor='val_auc', mode='min')
pos_weight = sum(x == 0 for x in y)/sum(x == 1 for x in y)
class_weight = {0:1, 1:pos_weight}
# print(class_weight)
set
with tf.device('/GPU:0'):
    model = Sequential()
    model.add(Dense(units=48, 
                    kernel_initializer="lecun_uniform", 
                    kernel_regularizer=L1L2(l1=0,l2=0.02),
                    bias_regularizer=L1L2(l1=0.04,l2=0.1),
                    activity_regularizer=L1L2(l1=0.04,l2=0.08),
                    input_shape = (X.shape[1],)
                   )
            )
    model.add(BatchNormalization())
    model.add(Activation("silu"))
    # model.add(Dropout(rate=0.2))

    for i in range(1):
        model.add(Dense(units=48, 
                    kernel_initializer="lecun_uniform", 
                    kernel_regularizer=L1L2(l1=0,l2=0.02),
                    bias_regularizer=L1L2(l1=0.04,l2=0.1),
                    activity_regularizer=L1L2(l1=0.04,l2=0.08),
                   )
            )
        model.add(BatchNormalization())
        model.add(Activation("silu"))
        model.add(Dropout(rate=0.2))

    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(learning_rate=5e-4, clipvalue=.1)
    metrics = [
        AUC(num_thresholds=1000, name='auc'),
        AUC(num_thresholds=1000, name='auprc', curve='PR'),
    ]
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics, )
    print(model.summary())
    history = model.fit(X.values, y, validation_data=(X_val.values, y_val), epochs=200, batch_size=32, class_weight=class_weight,  callbacks = [earlyStopping])
    
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('model AUC')
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
# plt.ylim(0.3, 1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['auprc'])
plt.plot(history.history['val_auprc'])
plt.title('model auprc')
# plt.ylim(0.3, 1)
plt.ylabel('auprc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

y_internaleval = internalEvaluationData[target_outcomes[0]]
y_eval = evaluationData[target_outcomes[0]]
y_eval_Wales = evaluationDataWales[target_outcomes[0]]
y_eval_Scotland = evaluationDataScotland[target_outcomes[0]]
print(model.evaluate(X_internaleval.values, y_internaleval))
print(model.evaluate(X_eval.values, y_eval))
print(model.evaluate(X_eval_Wales.values, y_eval_Wales))
print(model.evaluate(X_eval_Scotland.values, y_eval_Scotland))