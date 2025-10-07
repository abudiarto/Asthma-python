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
# from livelossplot import PlotLossesKeras
#internal validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score, matthews_corrcoef, auc, average_precision_score, roc_auc_score, balanced_accuracy_score, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import keras_tuner

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle

import time
# fix random seed for reproducibility
random_state = 42
tf.random.set_seed(random_state)
batch_size = 64
#prevent out of memmory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


gridSearch_X, crossVal_X, internalEvaluation_X, externalEvaluation_X, gridSearch_tab, crossVal_tab, internalEvaluation_tab, externalEvaluation_tab, gridSearch_y, crossVal_y, internalEvaluation_y, externalEvaluation_y = pickle.load(open('../Clean_data/seasonal_dataset_full_05112024.sav', 'rb'))

print(gridSearch_X.shape)
print(gridSearch_tab.shape)
print(gridSearch_y.shape)

unique, counts = np.unique(gridSearch_y, return_counts=True)
event_rate = counts[1]/len(gridSearch_y)

print(f'event rate: {event_rate}')

# print(search_train[target_outcomes].value_counts()[0]/search_train[target_outcomes].value_counts()[1])
# print(search_val[target_outcomes].value_counts()[0]/search_val[target_outcomes].value_counts()[1])


hp = keras_tuner.HyperParameters()
class MyHyperModel_allParams(keras_tuner.HyperModel):     
    def build(self, hp):
        #tabular dara - demography   

        input1 = Input(shape=(gridSearch_tab.shape[1],))
        neurons_layer0 = hp.Int('neuron_units', min_value=32, max_value=128, step=32)
        nn = Dense(units=neurons_layer0, 
                    kernel_initializer=hp.Choice("kernel_initializer_layer0", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                    kernel_regularizer=L1L2(l1=hp.Float("kernel_l1_dense0", min_value=0.0, max_value=.1, step=0.02), 
                                            l2=hp.Float("kernel_l2_dense0", min_value=0.0, max_value=.1, step=0.02)
                                           ),
                    bias_regularizer=L1L2(l1=hp.Float("bias_l1_dense0", min_value=0.0, max_value=.1, step=0.02), 
                                            l2=hp.Float("bias_l2_dense0", min_value=0.0, max_value=.1, step=0.02)
                                           ),
                    activity_regularizer=L1L2(l1=hp.Float("act_l1_dense0", min_value=0.0, max_value=.1, step=0.02), 
                                            l2=hp.Float("act_l2_dense0", min_value=0.0, max_value=.1, step=0.02)
                                           ),
                    input_shape = (gridSearch_tab.shape[1],)
                   )(input1)
        nn = BatchNormalization()(nn)
        nn = Activation(hp.Choice("activation0", ["relu", "elu", "gelu", "silu", "selu"]))(nn)
        nn = Dropout(rate=hp.Float("rate0", min_value=0.1, max_value=0.5, step=0.1))(nn)

        
            
#######################################################################################################################################################################

        ###layer 2 - LSTM to the final product
        input2 = Input(shape=(gridSearch_X.shape[1], gridSearch_X.shape[2]))
        neuron_lstm_units = hp.Int('neuron_lstm_units', min_value=32, max_value=128, step=32)
        
        lstm = Bidirectional(LSTM(units=neuron_lstm_units, 
                            return_sequences=True, 
                            kernel_regularizer=L1L2(l1=hp.Float("kernel_l1_lstm1", min_value=0.0, max_value=0.1), 
                                           l2=hp.Float("kernel_l2_lstmTherapy", min_value=0.0, max_value=0.1)
                                          ),
                            bias_regularizer=L1L2(l1=hp.Float("bias_l1_lstm1", min_value=0.0, max_value=0.1), 
                                           l2=hp.Float("bias_l2_lstmTherapy", min_value=0.0, max_value=0.1)
                                          ),
                            activity_regularizer=L1L2(l1=hp.Float("act_l1_lstm1", min_value=0.0, max_value=0.1), 
                                           l2=hp.Float("act_l2_lstmTherapy", min_value=0.0, max_value=0.1)
                                          ),
                            recurrent_regularizer=L1L2(l1=hp.Float("rec_l1_lstm1", min_value=0.0, max_value=0.1), 
                                           l2=hp.Float("rec_l2_lstmTherapy", min_value=0.0, max_value=0.1)
                                          ),
                                 )
                            )(input2)
        lstm = Dropout(rate=hp.Float("rate_lstm1", min_value=0.1, max_value=0.5))(lstm)
        
            
        lstm = Bidirectional(LSTM(units=int(neurons_layer0/2), 
                            # return_sequences=True, 
                            kernel_regularizer=L1L2(l1=hp.Float("kernel_l1_lstm2", min_value=0.0, max_value=0.1), 
                                           l2=hp.Float("kernel_l2_lstmTherapy", min_value=0.0, max_value=0.1)
                                          ),
                            bias_regularizer=L1L2(l1=hp.Float("bias_l1_lstm2", min_value=0.0, max_value=0.1), 
                                           l2=hp.Float("bias_l2_lstmTherapy", min_value=0.0, max_value=0.1)
                                          ),
                            activity_regularizer=L1L2(l1=hp.Float("act_l1_lstm2", min_value=0.0, max_value=0.1), 
                                           l2=hp.Float("act_l2_lstmTherapy", min_value=0.0, max_value=0.1)
                                          ),
                            recurrent_regularizer=L1L2(l1=hp.Float("rec_l1_lstm2", min_value=0.0, max_value=0.1), 
                                           l2=hp.Float("rec_l2_lstmTherapy", min_value=0.0, max_value=0.1)
                                          ),
                                         )
                                    )(lstm)
        lstm = Dropout(rate=hp.Float("rate_lstm2", min_value=0.1, max_value=0.5))(lstm)
        
        
###################################################################################################################################################        
        
        #merge tabular and sequence layers
        # nn = Reshape((1, neurons_layer1))(nn)
        add = concatenate([nn, lstm], axis=1)

        ##layer 4 - FCN before classification layer
        neurons_final_layer = hp.Int("units_layer1", min_value=32, max_value=128, step=32)
        final = Dense(units=neurons_final_layer, 
                        kernel_initializer=hp.Choice("kernel_initializer_layer1", ["glorot_uniform", "glorot_normal", "lecun_uniform", "lecun_normal"]), 
                        kernel_regularizer=L1L2(l1=hp.Float("kernel_l1_dense1", min_value=0.0, max_value=.1, step=0.02), 
                                                l2=hp.Float("kernel_l2_dense1", min_value=0.0, max_value=.1, step=0.02)
                                               ),
                        bias_regularizer=L1L2(l1=hp.Float("bias_l1_dense1", min_value=0.0, max_value=.1, step=0.02), 
                                                l2=hp.Float("bias_l2_dense1", min_value=0.0, max_value=.1, step=0.02)
                                               ),
                        activity_regularizer=L1L2(l1=hp.Float("act_l1_dense1", min_value=0.0, max_value=.1, step=0.02), 
                                                l2=hp.Float("act_l2_dense1", min_value=0.0, max_value=.1, step=0.02)
                                               ),
                     )(add)
        final = BatchNormalization()(final)
        final = Activation(hp.Choice("activation", ["relu", "elu", "gelu", "silu", "selu"]))(final)
        final = Dropout(0.4)(final)     
        
                                                      


        ###layer 5 - classification layer
        # final = Flatten()(add)
        output = Dense(1, activation='sigmoid')(final)
        
        lr=hp.Choice("lr", [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]) 
        clipvalue=hp.Float("clipvalue", min_value=0.1, max_value=0.7, step=0.2)
        beta_1=hp.Float("beta_1", min_value=0.8, max_value=0.95, step=0.05)
        beta_2=hp.Float("beta_2", min_value=0.900, max_value=0.999, step=0.011)
        epsilon=hp.Choice("epsilon", [1e-6, 1e-7, 1e-8]) 
        weight_decay=hp.Float("weight_decay", min_value=0, max_value=0.1) 

        opt = Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, weight_decay=weight_decay)
        # opt = RMSprop(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, weight_decay=weight_decay)

        metrics = [
            AUC(num_thresholds=1000, name='auc', curve='ROC'),
            AUC(num_thresholds=1000, name='auprc', curve='PR'),
            # tf.keras.metrics.Precision(name='prec'),
            # tf.keras.metrics.Recall(name='rec'),
            # tf.keras.metrics.TrueNegatives(name='TN'),
            # tf.keras.metrics.TruePositives(name='TP'),
            # tf.keras.metrics.PrecisionAtRecall(0.8)
        ]

        loss = tf.keras.losses.BinaryCrossentropy()

        model = Model(inputs=[input1, input2], outputs=output)
        model.compile(
            loss='binary_crossentropy', 
            optimizer=opt, 
            metrics=metrics)
        print(model.summary())
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )
    
start_time = time.time()

#run tuner


#free up memory
sequence_data = []

with tf.device('/GPU:0'):
    model = MyHyperModel_allParams()
    tuner = keras_tuner.BayesianOptimization(
        hypermodel= model,
        objective=keras_tuner.Objective("val_auc", direction="max"),
        max_trials=100,
        overwrite=True,
        seed=random_state,
        directory='../MODELS/LSTM/',
        project_name="lstm_tabseq_gridsearch_v2.0",
    )
    
    tuner.search_space_summary()
    earlyStopping = EarlyStopping(monitor='val_auc', patience=3, verbose=0, mode='max', restore_best_weights=True)
    pos_weight = counts[0]/counts[1]
    class_weight = {0:1, 1:pos_weight}
    
    tuner.search([gridSearch_tab, gridSearch_X], gridSearch_y, 
                 validation_split=.2,
                 epochs=20, 
                 batch_size = batch_size,
                 class_weight = class_weight,
                 callbacks = [earlyStopping])

    
print("#########################FINISH##############################")

print("--- Training time: %s seconds ---" % (time.time() - start_time))


#=========================================================================RETRAIN==============================================

# hp = keras_tuner.HyperParameters()

# #LOAD TUNER
# model = MyHyperModel_allParams()
# tuner = keras_tuner.BayesianOptimization(
#     hypermodel= model,
#     objective=keras_tuner.Objective("val_auc", direction="max"),
#     max_trials=50,
#     overwrite=False,
#     seed = random_state,
#     directory='../MODELS/LSTM/',
#     project_name="lstm_tabseq_gridsearch",
# )
# best_models = tuner.get_best_models(num_models=1)[0]

# print(best_models.summary())
# earlyStopping = EarlyStopping(monitor='val_auc', patience=20, verbose=1, mode='max', restore_best_weights=True)
# # mcp_save = ModelCheckpoint('../MODELS/lstm_tabseq.weights.hdf5', save_best_only=True, monitor='val_auc', mode='max')
# history = best_models.fit([gridSearch_tab, gridSearch_X], gridSearch_y, 
#                  validation_split=.2,
#                           epochs=1000, batch_size=batch_size, class_weight=class_weight, callbacks = [earlyStopping])

# pickle.dump(history, open('../MODELS/history_lstm_tabseq.sav', 'wb'))
# # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['auc'])
# plt.plot(history.history['val_auc'])
# plt.title('model AUC')
# plt.ylabel('AUC')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# # plt.show()
# plt.savefig('../SeqModel/val_auc_LSTM_bayesian-5%.png')

# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# # plt.ylim(0.3, 1)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# # plt.show()
# plt.savefig('../SeqModel/val_loss_LSTM_bayesian-5%.png')

# plt.plot(history.history['auprc'])
# plt.plot(history.history['val_auprc'])
# plt.title('model auprc')
# # plt.ylim(0.3, 1)
# plt.ylabel('auprc')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# # plt.show()
# plt.savefig('../SeqModel/val_auprc_LSTM_bayesian-5%.png')

# with tf.device('/GPU:0'):
#     print(best_models.evaluate([internalEvaluation_tab, internalEvaluation_X], internalEvaluation_y))
#     print(best_models.evaluate([externalEvaluation_tab, externalEvaluation_X], externalEvaluation_y))


