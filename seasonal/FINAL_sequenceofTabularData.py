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
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

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
# tf.set_random_seed(random_state)
# Initialize the variables
# init = tf.global_variables_initializer()



tf.keras.backend.clear_session()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


#LOAD DATA
gridSearch_X, crossVal_X, internalEvaluation_X, externalEvaluation_X, gridSearch_tab, crossVal_tab, internalEvaluation_tab, externalEvaluation_tab, gridSearch_y, crossVal_y, internalEvaluation_y, externalEvaluation_y = pickle.load(open('../Clean_data/sequence_dataset_full_27102024.sav', 'rb'))

def modelCombined():
    
###################################################################################################################################################            
        #tabular data - demography   
        input1 = Input(shape=(crossVal_tab.shape[1],))
        nn = Dense(units=64, 
                    input_shape = (crossVal_tab.shape[1],)
                   )(input1)
        nn = BatchNormalization()(nn)
        nn = Activation("relu")(nn)
        nn = Dropout(.3)(nn)
        


###################################################################################################################################################                
    

        # LSTM - layer 1
        
        input2 = Input(shape=(crossVal_X.shape[1], crossVal_X.shape[2]))
        lstm = Bidirectional(LSTM(units=100, 
                            return_sequences=True,
                                 )
                            )(input2)
        lstm = Dropout(.3)(lstm)
        
        lstm = Bidirectional(LSTM(units=50, 
                            return_sequences=True,
                                 )
                            )(lstm)
        lstm = Dropout(.3)(lstm)
        

        # LSTM - layer 2
        lstm = Bidirectional(LSTM(units=32, 
                            # return_sequences=True,
                                 )
                            )(lstm)
        lstm = Dropout(.3)(lstm)
        
        
###################################################################################################################################################        

        ##layer 4 - FCN before classification layer
        #merge tabular and sequence layers
        # nn = Reshape((1, 64))(nn) #reshape the nn product to be concatenated with LSTM product
        add = concatenate([nn, lstm], axis=1)

        final = Dense(units=32)(add)
        final = BatchNormalization()(final)
        final = Activation("relu")(final)
        final = Dropout(0.3)(final)     
        
                                                    
        ###layer 5 - classification layer
        # final = Flatten()(add)
        output = Dense(1, activation='sigmoid')(final)
        

        opt = Adamax(learning_rate=1e-4)
        # opt = RMSprop(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, weight_decay=weight_decay)

        metrics = [
            AUC(num_thresholds=1000, name='auc', curve='ROC'),
            AUC(num_thresholds=1000, name='auprc', curve='PR'),
        ]


        model = Model(inputs=[input1, input2], outputs=output)
        model.compile(
            loss='binary_crossentropy', 
            optimizer=opt, 
            metrics=metrics)
        # print(model.summary())
        return model



# with tf.Session() as sess:
#     sess.run(init)
model = modelCombined()
model.summary()

earlyStopping = EarlyStopping(monitor='val_auc', patience=10, verbose=0, mode='max', restore_best_weights=True)
# mcp_save = ModelCheckpoint('../SeqModel/lstm_CV_18+_'+set_vocab+str(fold_var)+'.h5', save_best_only=True, monitor='val_auc', mode='max')
# class_weight = trainingData['outcome_12months'].value_counts()[0]/trainingData['outcome_12months'].value_counts()[1]
class_weight = np.unique(crossVal_y, return_counts=True)[1][0]/np.unique(crossVal_y, return_counts=True)[1][1]
class_weight = {0:1, 1:class_weight}

# set
# with tf.device('/GPU:0'):
history = model.fit(
                    # [crossVal_tab, crossVal_X], crossVal_y,
                    [gridSearch_tab, gridSearch_X], gridSearch_y,
                    validation_split=.2,
                    # validation_data=([val_tab, val_X], val_y),
                    epochs=100, batch_size=32, class_weight=class_weight, callbacks = [earlyStopping])


# plot history
pyplot.plot(history.history['loss'], label='loss train')
pyplot.plot(history.history['val_loss'], label='loss val')
pyplot.legend()
pyplot.show()
pyplot.plot(history.history['auc'], label='auc train')
pyplot.plot(history.history['val_auc'], label='auc val')
pyplot.legend()
pyplot.show()


model.evaluate([externalEvaluation_tab,externalEvaluation_X], externalEvaluation_y)



predsraw = model.predict([externalEvaluation_tab,externalEvaluation_X])
preds = [0 if x <= 0.55 else 1 for x in predsraw]
print(confusion_matrix(externalEvaluation_y, preds))
print(roc_auc_score(externalEvaluation_y, preds))
print(average_precision_score(externalEvaluation_y, preds))

tf.keras.backend.clear_session()