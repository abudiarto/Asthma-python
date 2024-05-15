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
tf.random.set_seed(1234)


trainingData, validationData, internalEvaluationData, evaluationData, evaluationDataWales, evaluationDataScotland = pickle.load(open('../FinalData/dataset_scaled_2vs1_09122023.sav', 'rb'))

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


#Define feature candidates
features_columns = trainingData.columns.to_list()
exclude_columns = ['patid', 'practice_id', #identifier
                   'BMI', #use the categorical instead
                   'ethnicity', #use ethnic_group instead
                   'Spacer',  #all zero
                   
                   'outcome_3months', 'outcome_6months', 'outcome_9months', 'outcome_12months', 'outcome_15months', 'outcome_18months', 
                   'outcome_21months', 'outcome_24months', 'outcome_combined_6months', 'outcome_combined_9months', 'outcome_combined_12months', 
                   'outcome_combined_15months', 'outcome_combined_18months', 'outcome_combined_24months', '3months', '6months', '9months', '12months', '24months', #outcomes variable
                   
                   'postcode_district', 'County', 'LocalAuthority', 'OutputAreaClassification', #location related variables, use IMD decile only
                   
                   'cat_age', 'cat_average_daily_dose_ICS', 'cat_prescribed_daily_dose_ICS', 'cat_ICS_medication_possesion_ratio', 'cat_numOCS', 'cat_numOCSEvents', 
                   'cat_numOCSwithLRTI', 'cat_numAcuteRespEvents', 'cat_numAntibioticsEvents', 'cat_numAntibioticswithLRTI', 'cat_numAsthmaAttacks', 'cat_numHospEvents', 
                   'cat_numPCS', 'cat_numPCSAsthma', #use continous vars instead
                   
                   'count_rhinitis', 'count_cardiovascular', 'count_heartfailure',
                   'count_psoriasis', 'count_anaphylaxis', 'count_diabetes', 'count_ihd',
                   'count_anxiety', 'count_eczema', 'count_nasalpolyps',
                   'count_paracetamol', 'count_nsaids', 'count_betablocker', #use binary ones
                   
                   'paracetamol', 'nsaids', 'betablocker', #no data in evaluation
                   
                   'numOCSEvents', #duplicate with numOCS
                   
                   'month_12', 'month_4', 'month_5', 'month_10', 'month_1', 'month_6', 'month_3', 
                   'month_11', 'month_8', 'month_9', 'month_7', 'month_2', #month of attacks
                   
                   # 'system_EMIS', 'system_SystemOne', 'system_Vision', #primary care system used
                  ]
exclude_columns = exclude_columns + [x for x in features_columns if '_count' in x] #filter out commorbid count variables
features_columns = [x for x in features_columns if x not in exclude_columns]
print('Features size: ', len(features_columns))
print(features_columns)


#load sequence
clinical = pd.read_feather('../SeqModel/all_data_clinical_specific.feather')
therapy = pd.read_feather('../SeqModel/all_data_therapy_specific.feather')
seqCols = ['patid',
       'read_code_seq_padded_end_idx_clin',
       'month_padded_idx_end_clin',
       'read_code_seq_padded_end_idx_ther',
       'month_padded_idx_end_ther']
sequence_data = clinical.merge(therapy[['patid', 'read_code_seq_padded_idx', 'read_code_seq_padded_end_idx',
       'month_padded_idx', 'month_padded_idx_end']], on='patid', suffixes=['_clin', '_ther'], how='inner')

trainingData = trainingData.merge(sequence_data[seqCols], on='patid', how='inner')
validationData = validationData.merge(sequence_data[seqCols], on='patid', how='inner')
internalEvaluationData = internalEvaluationData.merge(sequence_data[seqCols], on='patid', how='inner')
evaluationData = evaluationData.merge(sequence_data[seqCols], on='patid', how='inner')
evaluationDataWales = evaluationDataWales.merge(sequence_data[seqCols], on='patid', how='inner')
evaluationDataScotland = evaluationDataScotland.merge(sequence_data[seqCols], on='patid', how='inner')

#vocab
code2idx_clin = pickle.load(open('../SeqModel/all_vocab_clinical_specific.sav', 'rb'))
code2idx_ther = pickle.load(open('../SeqModel/all_vocab_therapy_specific.sav', 'rb'))
month2idx = pickle.load(open('../SeqModel/all_vocab_month.sav', 'rb'))
vocab_size_clinical = len(code2idx_clin)+1
vocab_size_therapy = len(code2idx_ther)+1
month_size = len(month2idx)+1
print(vocab_size_clinical)
print(vocab_size_therapy)
print(month_size)

Xt_train = np.array(trainingData[features_columns].values)
Xt_val = np.array(validationData[features_columns].values)
Xt_internaleval = np.array(internalEvaluationData[features_columns].values)
Xt_eval = np.array(evaluationData[features_columns].values)
Xt_eval_Wales = np.array(evaluationDataWales[features_columns].values)
Xt_eval_Scotland = np.array(evaluationDataScotland[features_columns].values)

#scalling tabular data
scaler = StandardScaler().fit(Xt_train)
Xt_train = scaler.transform(Xt_train)
Xt_val = scaler.transform(Xt_val)
Xt_internaleval = scaler.transform(Xt_internaleval)
Xt_eval = scaler.transform(Xt_eval)
Xt_eval_Wales = scaler.transform(Xt_eval_Wales)
Xt_eval_Scotland = scaler.transform(Xt_eval_Scotland)

Xclin_train = np.array(trainingData['read_code_seq_padded_end_idx_clin'].values)
Xclin_val = np.array(validationData['read_code_seq_padded_end_idx_clin'].values)
Xclin_internaleval = np.array(internalEvaluationData['read_code_seq_padded_end_idx_clin'].values)
Xclin_eval = np.array(evaluationData['read_code_seq_padded_end_idx_clin'].values)
Xclin_eval_Wales = np.array(evaluationDataWales['read_code_seq_padded_end_idx_clin'].values)
Xclin_eval_Scotland = np.array(evaluationDataScotland['read_code_seq_padded_end_idx_clin'].values)
Xclin_train = np.array([x for x in Xclin_train])
Xclin_val = np.array([x for x in Xclin_val])
Xclin_internaleval = np.array([x for x in Xclin_internaleval])
Xclin_eval = np.array([x for x in Xclin_eval])
Xclin_eval_Wales = np.array([x for x in Xclin_eval_Wales])
Xclin_eval_Scotland = np.array([x for x in Xclin_eval_Scotland])

Xther_train = np.array(trainingData['read_code_seq_padded_end_idx_ther'].values)
Xther_val = np.array(validationData['read_code_seq_padded_end_idx_ther'].values)
Xther_internaleval = np.array(internalEvaluationData['read_code_seq_padded_end_idx_ther'].values)
Xther_eval = np.array(evaluationData['read_code_seq_padded_end_idx_ther'].values)
Xther_eval_Wales = np.array(evaluationDataWales['read_code_seq_padded_end_idx_ther'].values)
Xther_eval_Scotland = np.array(evaluationDataScotland['read_code_seq_padded_end_idx_ther'].values)
Xther_train = np.array([x for x in Xther_train])
Xther_val = np.array([x for x in Xther_val])
Xther_internaleval = np.array([x for x in Xther_internaleval])
Xther_eval = np.array([x for x in Xther_eval])
Xther_eval_Wales = np.array([x for x in Xther_eval_Wales])
Xther_eval_Scotland = np.array([x for x in Xther_eval_Scotland])


print(Xt_train.shape)
print(Xt_internaleval.shape)
print(Xt_val.shape)
print(Xt_eval.shape)
print(Xt_eval_Wales.shape)
print(Xt_eval_Scotland.shape)


target_outcome = '12months'
y_train = trainingData[target_outcome].values
y_val = validationData[target_outcome].values
y_internaleval = internalEvaluationData[target_outcome].values
y_eval = evaluationData[target_outcome].values
y_eval_Wales = evaluationDataWales[target_outcome].values
y_eval_Scotland = evaluationDataScotland[target_outcome].values


max_codes_clin = Xclin_train.shape[1]
max_codes_ther = Xther_train.shape[1]
max_codes = 100
tab_feature_size = Xt_train.shape[1]
top_vocabs_portion = .1

print(max_codes_clin)
print(max_codes_ther)

hp = keras_tuner.HyperParameters()
class MyHyperModel_allParams(keras_tuner.HyperModel):
    def build(self, hp):
        #tabular dara - demography   
        inputs1 = Input(shape=tab_feature_size)
        neurons_layer0 = hp.Int('neuron_units', min_value=32, max_value=128, step=32)
        neurons_layer1 = hp.Int('neuron_units', min_value=32, max_value=128, step=32)
        nn = Dense(units=neurons_layer1, 
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
                    input_shape = (tab_feature_size,)
                   )(inputs1)
        nn = BatchNormalization()(nn)
        nn = Activation(hp.Choice("activation0", ["relu", "elu", "gelu", "silu", "selu"]))(nn)
        nn = Dropout(rate=hp.Float("rate0", min_value=0.1, max_value=0.5, step=0.1))(nn)

        
        #==================================================================================================================================================#

        #clinical embedding for lstm
        inputs2 = Input(shape=max_codes)
        embedding_size = hp.Int("embedding_size", min_value=int(np.cbrt(vocab_size_clinical)), max_value=int(np.sqrt(vocab_size_therapy)))
        embedding_clin = Embedding(int(top_vocabs_portion*vocab_size_clinical), 
                                   output_dim = embedding_size, 
                                   input_length=max_codes,
                                   mask_zero=True,
                                  )(inputs2)
        


        #therapy embedding for lstm
        inputs3 = Input(shape=max_codes)
        embedding_ther = Embedding(int(top_vocabs_portion*vocab_size_therapy), 
                                   output_dim = embedding_size, 
                                   input_length=max_codes,
                                   mask_zero=True,
                             )(inputs3)
        

        ###Layer 1 - merge add (clin+ther) and lstm ther
        allEmbedding = Add()([embedding_clin, embedding_ther])
        # allEmbedding = concatenate([embedding_clin, embedding_ther], axis=1)
#######################################################################################################################################################################

        ###layer 2 - LSTM to the final product
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
                                )(allEmbedding)
            lstm = Dropout(rate=hp.Float("rate_lstm1", min_value=0.1, max_value=0.5))(lstm)
        
            
        lstm = Bidirectional(LSTM(units=int(neurons_layer1/2), 
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

        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=output)
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
earlyStopping = EarlyStopping(monitor='val_auc', patience=5, verbose=0, mode='max', restore_best_weights=True)
pos_weight = trainingData[target_outcome].value_counts()[0]/trainingData[target_outcome].value_counts()[1]
class_weight = {0:1, 1:pos_weight}

#free up memory
sequence_data = []

with tf.device('/GPU:0'):
    model = MyHyperModel_allParams()
    tuner = keras_tuner.BayesianOptimization(
        hypermodel= model,
        objective=keras_tuner.Objective("val_auc", direction="max"),
        max_trials=15,
        overwrite=False,
        seed=1234,
        directory='../SeqModel/tuner/',
        project_name="lstmv2.0AllParams1-bayesian-deeper",
    )
    
best_models = tuner.get_best_models(num_models=1)[0]
print(best_models.summary())
earlyStopping = EarlyStopping(monitor='val_auc', patience=20, verbose=0, mode='max', restore_best_weights=True)
mcp_save = ModelCheckpoint('../SeqModel/lstmv2.0AllParams1-bayesian-deeper-specific.weights.hdf5', save_best_only=True, monitor='val_auc', mode='max')
history = best_models.fit([Xt_train, Xclin_train[:,:max_codes], Xther_train[:,:max_codes]], y_train, 
                 validation_data=([Xt_val, Xclin_val[:,:max_codes], Xther_val[:,:max_codes]], y_val),
                          epochs=200, batch_size=128, class_weight=class_weight, callbacks = [earlyStopping])

pickle.dump(history, open('../SeqModel/history_lstmv2.0AllParams1-bayesian-deeper.sav', 'wb'))
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('model AUC')
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('../SeqModel/val_auc_LSTM_bayesian-deeper.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
# plt.ylim(0.3, 1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
# plt.show()
plt.savefig('../SeqModel/val_loss_LSTM_bayesian-deeper.png')

plt.plot(history.history['auprc'])
plt.plot(history.history['val_auprc'])
plt.title('model auprc')
# plt.ylim(0.3, 1)
plt.ylabel('auprc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
# plt.show()
plt.savefig('../SeqModel/val_auprc_LSTM_bayesian-deeper.png')

with tf.device('/GPU:0'):
    print(best_models.evaluate([Xt_internaleval, Xclin_internaleval[:,:max_codes], Xther_internaleval[:,:max_codes]], y_internaleval))
    print(best_models.evaluate([Xt_eval, Xclin_eval[:,:max_codes], Xther_eval[:,:max_codes]], y_eval))
    print(best_models.evaluate([Xt_eval_Wales, Xclin_eval_Wales[:,:max_codes], Xther_eval_Wales[:,:max_codes]], y_eval_Wales))
    print(best_models.evaluate([Xt_eval_Scotland, Xclin_eval_Scotland[:,:max_codes], Xther_eval_Scotland[:,:max_codes]], y_eval_Scotland))