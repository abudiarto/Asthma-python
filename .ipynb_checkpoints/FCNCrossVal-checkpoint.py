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

from sklearn.model_selection import KFold, StratifiedKFold

import keras_tuner

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle

import time
# fix random seed for reproducibility
tf.random.set_seed(1234)


trainingData, _, _, _, _, _ = pickle.load(open('../FinalData/dataset_scaled_2vs1_09122023.sav', 'rb'))

trainingData = trainingData[(trainingData.age >=18) & (trainingData.age <=80)]




trainingData = trainingData.rename({'3MonthsOutcome': '3months', '6MonthsOutcome': '6months','9MonthsOutcome': '9months','12MonthsOutcome': '12months',}, axis=1)




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
# change this path whether to use sequence with full readcodes (~80,000 vocabs) or only asthma-related readcodes (~1500 vocabs)
clinical = pd.read_feather('../SeqModel/all_data_clinical_specific.feather')
therapy = pd.read_feather('../SeqModel/all_data_therapy_specific.feather')
seqCols = ['patid',
       'read_code_seq_padded_end_idx_clin',
       'month_padded_idx_end_clin',
       'read_code_seq_padded_end_idx_ther',
       'month_padded_idx_end_ther']

sequence_data = clinical.merge(therapy[['patid', 'read_code_seq_padded_idx', 'read_code_seq_padded_end_idx',
       'month_padded_idx', 'month_padded_idx_end']], on='patid', suffixes=['_clin', '_ther'], how='inner')

#free up memory
clinical = []
therapy = []

trainingData = trainingData.merge(sequence_data[seqCols], on='patid', how='inner')


print(trainingData.shape)


#vocab - change this too, full readcodes (~80,000 vocabs) or only asthma-related readcodes (~1500 vocabs)
code2idx_clin = pickle.load(open('../SeqModel/all_vocab_clinical_specific.sav', 'rb'))
code2idx_ther = pickle.load(open('../SeqModel/all_vocab_therapy_specific.sav', 'rb'))
month2idx = pickle.load(open('../SeqModel/all_vocab_month.sav', 'rb'))
vocab_size_clinical = len(code2idx_clin)+1
vocab_size_therapy = len(code2idx_ther)+1
month_size = len(month2idx)+1
print(vocab_size_clinical)
print(vocab_size_therapy)
print(month_size)




# use only 20% of training data for parameter search
target_outcomes = '12months'
ignore, cvData = train_test_split(trainingData, stratify=trainingData[target_outcomes], test_size=0.10, random_state=1234)


Xt = np.array(cvData[features_columns].values)
scaler = StandardScaler().fit(Xt)
Xt = scaler.transform(Xt)
Xclin = np.array(cvData['read_code_seq_padded_end_idx_clin'].values)
Xther = np.array(cvData['read_code_seq_padded_end_idx_ther'].values)

Xclin = np.array([x for x in Xclin])
Xther = np.array([x for x in Xther])

y = cvData[target_outcomes].values

print(Xt.shape)
print(y.shape)



#for all data max_codes=100, for specific asthma code max_codes=25
max_codes = 25
tab_feature_size = Xt.shape[1]
set_vocab = 'FCN'
top_vocabs_portion = 1

hp = keras_tuner.HyperParameters()
# %%time

# create the model
# embedding_vector_length = 50
hp = keras_tuner.HyperParameters()

# print(class_weight)

class MyHyperModel_allParameters(keras_tuner.HyperModel):
        
    def build(self, hp):
        model = Sequential()
        
        neurons_layer1 = hp.Int("units_layer1", min_value=32, max_value=128, step=16)
        model.add(Dense(units=neurons_layer1, 
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
                        input_shape = (Xt.shape[1],)
                       )
                 )
        model.add(BatchNormalization())
        model.add(Activation(hp.Choice("activation", ["relu", "elu", "gelu", "silu", "selu"])))
        model.add(Dropout(rate=hp.Float("rate", min_value=0.1, max_value=0.5, step=0.1)))
        
        layers = hp.Int("num_layers", min_value=0, max_value=10, step=1)
        if layers > 0:
            for i in range(layers):
                neurons_layer1 = hp.Int("units_layer"+str(i), min_value=32, max_value=128, step=16)
                model.add(Dense(units=neurons_layer1, 
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
                               )
                         )
                model.add(BatchNormalization())
                model.add(Activation(hp.Choice("activation", ["relu", "elu", "gelu", "silu", "selu"])))
                model.add(Dropout(rate=hp.Float("rate", min_value=0.1, max_value=0.5, step=0.1)))

        model.add(Dense(1, activation='sigmoid'))
        lr=hp.Choice("lr", [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]) 
        clipvalue=hp.Float("clipvalue", min_value=0.1, max_value=0.7, step=0.2)
        opt = Adam(learning_rate=lr, clipvalue=clipvalue)
        metrics = [
            AUC(num_thresholds=1000, name='auc'),
            AUC(num_thresholds=1000, name='auprc', curve='PR'),
        ]
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics, )
        print(model.summary())
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )

    
    
def get_model_name(k):
    return str(k)+'.h5'

#extract best modedl from tuner
hp = keras_tuner.HyperParameters()


model = MyHyperModel_allParameters()
tuner = keras_tuner.Hyperband(
        hypermodel= model,
        objective=keras_tuner.Objective("val_auc", direction="max"),
        max_epochs=25,
        hyperband_iterations=1,
        overwrite=False,
        directory='../SeqModel/tuner/',
        project_name="allParameters",
    )


#initialise cross val split
n_splits = 10
skf = StratifiedKFold(n_splits = n_splits, random_state = 1234, shuffle = True) 

VALIDATION_ACCURACY = []
VALIDATION_LOSS = []

fold_var = 1
n = Xt.shape[0]

for train_index, val_index in skf.split(np.zeros(n),y):
    print('Fold: ', fold_var)
    Xt_train = np.array([Xt[i] for i in train_index])
    Xt_val = np.array([Xt[i] for i in val_index])
    print(len(Xt_train))
    print(len(Xt_val))
    
    
    y_train = np.array([y[i] for i in train_index])
    y_val = np.array([y[i] for i in val_index])
    print(len(y_train))
    print(len(y_val))
    
    #call best model from tuner
    best_models = tuner.get_best_models(num_models=1)[0]
    
    print(best_models.summary())
    earlyStopping = EarlyStopping(monitor='val_auc', patience=5, verbose=0, mode='max', restore_best_weights=True)
    mcp_save = ModelCheckpoint('../SeqModel/lstm_CV_18+_'+set_vocab+str(fold_var)+'.h5', save_best_only=True, monitor='val_auc', mode='max')
    pos_weight = trainingData[target_outcomes].value_counts()[0]/trainingData[target_outcomes].value_counts()[1]
    class_weight = {0:1, 1:pos_weight}
    
    
    history = best_models.fit(Xt_train, y_train,
                              validation_split = 0.2,
                              epochs=10, batch_size=128, class_weight=class_weight, callbacks = [earlyStopping, mcp_save])
    
    
    # LOAD BEST MODEL to evaluate the performance of the model
    best_models.load_weights('../SeqModel/lstm_CV_18+_'+set_vocab+str(fold_var)+".h5")
    results = best_models.evaluate(Xt_val, y_val)
    results = dict(zip(best_models.metrics_names,results))
    VALIDATION_ACCURACY.append(results['auc'])
    VALIDATION_LOSS.append(results['loss'])
    tf.keras.backend.clear_session()
    fold_var += 1
    
pickle.dump([VALIDATION_ACCURACY, VALIDATION_LOSS], open('../SeqModel/CV_result_lstm_18+_'+ set_vocab + '.sav', 'wb'))


