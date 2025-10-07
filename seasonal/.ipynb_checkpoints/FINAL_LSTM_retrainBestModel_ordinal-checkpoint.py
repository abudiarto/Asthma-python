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
from tensorflow.keras.utils import plot_model
# from livelossplot import PlotLossesKeras
#internal validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score, matthews_corrcoef, auc, average_precision_score, roc_auc_score, balanced_accuracy_score, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#internal validation
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold, cross_val_score, GridSearchCV, PredefinedSplit, train_test_split

#performance metrices
from sklearn.metrics import make_scorer, confusion_matrix, classification_report, f1_score, balanced_accuracy_score, matthews_corrcoef, auc, average_precision_score, roc_auc_score, balanced_accuracy_score, roc_curve, accuracy_score

import keras_tuner

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle

import time
# fix random seed for reproducibility
random_state = 42
tf.random.set_seed(random_state)

#prevent out of memmory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


gridSearch_X, crossVal_X, internalEvaluation_X, externalEvaluation_X, gridSearch_tab, crossVal_tab, internalEvaluation_tab, externalEvaluation_tab, gridSearch_y, crossVal_y, internalEvaluation_y, externalEvaluation_y = pickle.load(open('../Clean_data/seasonal_dataset_ordinal.sav', 'rb'))

print(gridSearch_X.shape)
print(gridSearch_tab.shape)
print(gridSearch_y.shape)

unique, counts = np.unique(crossVal_y, return_counts=True)
event_rate = counts[1]/len(crossVal_y)

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
        
        for i in range (2):
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


#=========================================================================RETRAIN==============================================

def summariseResult (testTab, testX, testY, model):
    preds = model.predict([testTab, testX])
    preds = [0 if x < 0.5 else 1 for x in preds]
    # tn, fp, fn, tp = confusion_matrix(testY, preds).ravel()
    # specificity = tn / (tn+fp)
    # sensitivity = tp / (tp+fn)
    # ppv = 100*tp/(tp+fp)
    # npv = 100*tn/(fn+tn)
    # acc = accuracy_score(testY, preds)
    # f1score = f1_score(testY, preds, average = 'binary')
    # balanceacc = balanced_accuracy_score(testY, preds)
    fpr, tpr, thresholds = roc_curve(testY, preds, pos_label=1)
    # aucscore = auc(fpr, tpr)
    aucscore = roc_auc_score(testY, preds)
    auprc = average_precision_score(testY, preds)
    # plot_confusion_matrix(model, testX, testY, cmap='viridis')  
    return np.round(aucscore,4), np.round(auprc,4)


hp = keras_tuner.HyperParameters()

#LOAD TUNER
summary_result_train = []
cols = ['model_name', 'fold', 'outcome', 'class_ratio', 'auc', 'auprc']
n_splits = 2
batch_size = 128
patience = 2
epochs = 3
target_outcomes = ['outcome_12months']

#start CV
for target_outcome in target_outcomes:
    print(target_outcome)
    # y = crossValData[target_outcome]
    # y_internalVal = internalEvaluationData[target_outcome]
    # y_externalVal = externalEvaluationData[target_outcome]
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf.get_n_splits(crossVal_X)
    fold = 0
    
    for train_index, test_index in kf.split(crossVal_X, crossVal_y):
        #split data
        fold+=1
        print(f'fold: {fold}')
        tab_train, tab_test = crossVal_tab[train_index], crossVal_tab[test_index]
        X_train, X_test = crossVal_X[train_index], crossVal_X[test_index]
        y_train, y_test = crossVal_y[train_index], crossVal_y[test_index]

        #retrain model
        model = MyHyperModel_allParams()
        tuner = keras_tuner.BayesianOptimization(
            hypermodel= model,
            objective=keras_tuner.Objective("val_auc", direction="max"),
            max_trials=50,
            overwrite=False,
            seed = random_state,
            directory='../MODELS/LSTM/',
            project_name="lstm_tabseq_gridsearch",
        )
        best_models = tuner.get_best_models(num_models=1)[0]
        print(best_models.summary())
        plot_model(best_models, to_file='../MODELS/LSTM_simple_plot.png', show_shapes=True, show_layer_names=True)

        pos_weight = counts[0]/counts[1]
        class_weight = {0:1, 1:pos_weight}
        earlyStopping = EarlyStopping(monitor='val_auc', patience=patience, verbose=1, mode='max', restore_best_weights=True)
        mcp_save = ModelCheckpoint('../MODELS/lstm_tabseq.weights_CV_'+str(fold)+'.keras', save_best_only=True, monitor='val_auc', mode='max') #not needed in CV
        history = best_models.fit([tab_train, X_train], y_train, 
                         validation_split=.2,
                                  epochs=epochs, batch_size=batch_size, class_weight=class_weight, callbacks = [earlyStopping])
        
        pickle.dump(history, open('../MODELS/history_lstm_tabseq_CV'+str(fold)+'.sav', 'wb'))
        
        modelname = 'LSTM'
        summary_result_train.append((modelname, fold, target_outcome, event_rate, ) + summariseResult (tab_test, X_test, y_test, best_models))
        # with tf.device('/GPU:0'):
        #     print(best_models.evaluate([internalEvaluation_tab, internalEvaluation_X], internalEvaluation_y))
        #     print(best_models.evaluate([externalEvaluation_tab, externalEvaluation_X], externalEvaluation_y))

summary_result_train = pd.DataFrame(summary_result_train, columns=cols)
summary_result_train['model_num'] = summary_result_train.index
summary_result_train.to_csv('../MODELS/LSTM.csv', index_label=False, index=False)


print("#########################FINISH##############################")

print("--- Training time: %s seconds ---" % (time.time() - start_time))


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




