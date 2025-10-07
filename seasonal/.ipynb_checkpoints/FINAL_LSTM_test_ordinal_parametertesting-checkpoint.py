import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, Input, concatenate, Reshape, Activation, Flatten, Add, BatchNormalization, Multiply, LeakyReLU
from tensorflow.keras.layers import MultiHeadAttention, Attention, Dot, Multiply, Lambda
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
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




# print(search_train[target_outcomes].value_counts()[0]/search_train[target_outcomes].value_counts()[1])
# print(search_val[target_outcomes].value_counts()[0]/search_val[target_outcomes].value_counts()[1])

#for outer product
def outer_product(inputs): #function to merge
    return tf.einsum('bi,bj->bij', inputs[0], inputs[1])
    
def modelCombined(X, tab):
    
###################################################################################################################################################            
        #tabular data - demography 
        neurons_layer1=64
        input1 = Input(shape=(tab.shape[1],))
        nn = Dense(units=neurons_layer1, 
                    input_shape = (tab.shape[1],)
                   )(input1)
        nn = BatchNormalization()(nn)
        nn = Activation("relu")(nn)
        nn = Dropout(.3)(nn)
        


###################################################################################################################################################                

        # LSTM - layer 1
        
        input2 = Input(shape=(X.shape[1], X.shape[2]))
        lstm = Bidirectional(LSTM(units=32, 
                            return_sequences=True,
                                 )
                            )(input2)
        lstm = Dropout(.3)(lstm)
        

        # LSTM - layer 2
        lstm = Bidirectional(LSTM(units=32, 
                            return_sequences=True,
                                 )
                            )(lstm)
        lstm = Dropout(.3)(lstm)

     # # LSTM - layer 3
     #    lstm = Bidirectional(LSTM(units=32, 
     #                        return_sequences=True,
     #                             )
     #                        )(lstm)
     #    lstm = Dropout(.3)(lstm)

     # LSTM - layer 4
        lstm = Bidirectional(LSTM(units=16, 
                            # return_sequences=True,
                                 )
                            )(lstm)
        lstm = Dropout(.3)(lstm)

        #dense layer before merging
        lstm = Dense(units=64, 
                   )(lstm)
        lstm = BatchNormalization()(lstm)
        lstm = Activation("relu")(lstm)
        
        
###################################################################################################################################################        

        
        #merge tabular and sequence layers
        # nn = Reshape((1, 64))(nn) #reshape the nn product to be concatenated with LSTM product
        add = concatenate([nn, lstm], axis=1)

        # nn = Reshape((1, neurons_layer1))(nn)
        # add = concatenate([nn, lstm], axis=1)
        
        #Dot product
        # add = Dot(axes=-1, normalize=True)([nn, lstm])
        
        #Addition
        add = Add()([nn, lstm]) #.8062
        
        # # Multiply
        # add = Multiply()([nn, lstm])


        # # Reshape outputs to be compatible with MultiHeadAttention
        # nn = Reshape((1, -1))(nn)
        # lstm = Reshape((1, -1))(lstm)
        # # Apply cross-attention
        # add = MultiHeadAttention(num_heads=2, key_dim=16)(nn, lstm)
        # add = Flatten()(add)
        
        # #outer merge
        # add = Lambda(outer_product)([nn, lstm])
        # add = Flatten()(add)
        
        # # ###Gatted mechanism
        # Compute gates
        # gate_nn = Dense(32, activation='sigmoid')(nn)
        # gate_lstm = Dense(32, activation='sigmoid')(lstm)
        # # Apply gates
        # gated_tabular = Multiply()([nn, gate_nn])
        # gated_time_series = Multiply()([lstm, gate_lstm])
        # # Merge
        # add = Add()([gate_nn, gate_lstm])

####################################################################################################################################
    ## FCN before classification layer
        final = Dense(units=32)(add)
        final = BatchNormalization()(final)
        final = Activation("relu")(final)
        final = Dropout(0.3)(final)     
        
                                                    
        ###layer 5 - classification layer
        # final = Flatten()(add)
        output = Dense(1, activation='sigmoid')(final)
        

        opt = Adam(learning_rate=1e-3)
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
    
start_time = time.time()

#run tuner

#free up memory
sequence_data = []


#=========================================================================RETRAIN==============================================

def summariseResult (testTab, testX, testY, model):
    preds = model.predict([testTab, testX])
    # preds = [0 if x < 0.5 else 1 for x in preds]
    preds = [x for x in preds]
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



# Define a custom learning rate schedule function
def lr_schedule(epoch):
    initial_lr = 0.01
    if epoch < 10:
        return initial_lr
    elif epoch < 20:
        return initial_lr * 0.1
    else:
        return initial_lr * 0.01



#parameters setting
summary_result_internalVal = []
summary_result_externalVal = []
cols = ['model_name', 'fold', 'outcome', 'class_ratio', 'auc', 'auprc']
n_splits = 5
batch_size = 256
patience = 5
epochs = 10
target_outcomes = [
    'outcome_12months',
    'outcome_3months', 'outcome_6months', 'outcome_9months', 
    
] 

#start CV
for target_outcome in target_outcomes:
    print(target_outcome)
    gridSearch_X, crossVal_X, internalEvaluation_X, externalEvaluation_X, gridSearch_tab, crossVal_tab, internalEvaluation_tab, externalEvaluation_tab, gridSearch_y, crossVal_y, internalEvaluation_y, externalEvaluation_y = pickle.load(open('../../Clean_data/seasonal_dataset_ordinal_'+target_outcome+'.sav', 'rb'))

    X = np.concatenate((gridSearch_X, crossVal_X))
    tab = np.concatenate((gridSearch_tab, crossVal_tab))
    y = np.concatenate((gridSearch_y, crossVal_y))

    X = X[:, :9, :] #select only data from the last 2 years
    internalEvaluation_X = internalEvaluation_X[:, :9, :]
    externalEvaluation_X = externalEvaluation_X[:, :9, :]

    print(tab.shape)
    print(X.shape)
    print(y.shape)
    
    unique, counts = np.unique(y, return_counts=True)
    event_rate = counts[1]/len(y)
    
    print(f'event rate: {event_rate}')

    #split data
    fold='99'
    print(f'fold: {fold}')

    
    # with tf.device('/GPU:0'):
    model = modelCombined(X, tab)
    model.summary()
    # plot_model(model, to_file='../../MODELS/TestResultOrdinal/LSTM_simple_plot.png', show_shapes=True, show_layer_names=True)

    earlyStopping = EarlyStopping(monitor='val_auc', patience=patience, verbose=1, mode='max', restore_best_weights=True)
    mcp_save = ModelCheckpoint('../../MODELS/TestResultOrdinal/lstm_2years_new'+ target_outcome +'.keras', save_best_only=True, monitor='val_auc', mode='max') #not needed in CV
    pos_weight = counts[0]/counts[1]
    class_weight = {0:1, 1:pos_weight}
    # Instantiate the scheduler
    lr_scheduler = LearningRateScheduler(lr_schedule)




    history = model.fit([tab, X], y,
                        validation_split=.2,
                        # validation_data=([val_tab, val_X], val_y),
                        epochs=epochs, batch_size=batch_size, class_weight=class_weight, callbacks = [lr_scheduler, earlyStopping, mcp_save])

    pickle.dump(history, open('../../MODELS/TestResultOrdinal/history_2years_new'+target_outcome+'.sav', 'wb'))
        
    modelname = 'LSTM'
    summary_result_internalVal.append((modelname, fold, target_outcome, event_rate, ) + summariseResult (internalEvaluation_tab, internalEvaluation_X, internalEvaluation_y, model))
    summary_result_externalVal.append((modelname, fold, target_outcome, event_rate, ) + summariseResult (externalEvaluation_tab, externalEvaluation_X, externalEvaluation_y, model))
        # with tf.device('/GPU:0'):
        #     print(best_models.evaluate([internalEvaluation_tab, internalEvaluation_X], internalEvaluation_y))
        #     print(best_models.evaluate([externalEvaluation_tab, externalEvaluation_X], externalEvaluation_y))



summary_result_internalVal = pd.DataFrame(summary_result_internalVal, columns=cols)
summary_result_internalVal['model_num'] = summary_result_internalVal.index
summary_result_internalVal.to_csv('../../MODELS/internalValResultLSTMOrdinal_2years_new.csv', index_label=False, index=False)

summary_result_externalVal = pd.DataFrame(summary_result_externalVal, columns=cols)
summary_result_externalVal['model_num'] = summary_result_externalVal.index
summary_result_externalVal.to_csv('../../MODELS/externalValResultLSTMOrdinal_2years_new.csv', index_label=False, index=False)

print("#########################FINISH##############################")

print("--- Training time: %s seconds ---" % (time.time() - start_time))







