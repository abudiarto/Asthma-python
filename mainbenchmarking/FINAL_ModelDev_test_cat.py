import pandas as pd
import numpy as np
import sklearn
import pyreadr

#statistics
from scipy.stats import chi2_contingency, ttest_ind

# import cudf #gpu-powered DataFrame (Pandas alternative)

#imbalance handling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, RepeatedEditedNearestNeighbours
from imblearn.pipeline import Pipeline

#preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler


#internal validation
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold, cross_val_score, GridSearchCV, PredefinedSplit, train_test_split

#performance metrices
from sklearn.metrics import make_scorer, confusion_matrix, classification_report, f1_score, balanced_accuracy_score, matthews_corrcoef, auc, average_precision_score, roc_auc_score, balanced_accuracy_score, roc_curve, accuracy_score

#Models selection
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# from cuml.svm import SVC #gpu-powered SVM

#Tree pruning
from sklearn.tree._tree import TREE_LEAF


#save and load trained model
import pickle

#visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

from collections import Counter
import os
import torch
import time
random_state = 42


# Data loader
# features = pd.read_csv("../FinalData/cleaned_features_11072023.csv")
gridSearchData, crossValData, internalEvaluationData, externalEvaluationData = pickle.load(open('../Clean_data/dataset_cat_2vs1_25102024.sav', 'rb'))
# outcomes = pd.read_csv("../Clean_data/cleaned_outcomes_24102024.csv")

#Define feature candidates

features_columns = gridSearchData.columns.to_list()
exclude_columns = ['patid', 'practice_id', 'set', #identifier
                   'BMI', #use the categorical instead
                   'ethnicity', #use ethnic_group instead
                   'Spacer',  #all zero
                   
                   'outcome_3months', 'outcome_6months', 'outcome_9months', 'outcome_12months', 'outcome_15months', 'outcome_18months', 
                   'outcome_21months', 'outcome_24months', 'outcome_combined_6months', 'outcome_combined_9months', 'outcome_combined_12months', 
                   'outcome_combined_15months', 'outcome_combined_18months', 'outcome_combined_24months', '3months', '6months', '12months', '24months', #outcomes variable
                   
                   'postcode_district', 'County', 'LocalAuthority', 'OutputAreaClassification', #location related variables, use IMD decile only
                   
                   'age', 'average_daily_dose_ICS', 'prescribed_daily_dose_ICS',
                   'ICS_medication_possesion_ratio', 'numPCS', 'numPCSAsthma',
                   'numAntibioticsEvents', 'numAntibioticswithLRTI', 'numOCSEvents',
                   'numOCSwithLRTI', 'numAsthmaAttacks', 'numAcuteRespEvents',
                   'numHospEvents', 'numAsthmaManagement', 'numAsthmaReview',
                   'numAsthmaMedReview', 'numAsthmaReviewRCP', #use categorical vars instead
                   
                   'count_rhinitis', 'count_cardiovascular', 'count_heartfailure',
                   'count_psoriasis', 'count_anaphylaxis', 'count_diabetes', 'count_ihd',
                   'count_anxiety', 'count_eczema', 'count_nasalpolyps',
                   'count_paracetamol', 'count_nsaids', 'count_betablocker', #use binary ones
                   
                   'paracetamol', 'nsaids', 'betablocker', #no data in evaluation
                                      
                  ]
# exclude_columns = exclude_columns + [x for x in features_columns if '_count' in x] #filter out commorbid count variables
features_columns = [x for x in features_columns if x not in exclude_columns]
print('Features size: ', len(features_columns))
print(features_columns)


#Model evaluation function

def summariseResult (testX, testY, model):
    preds = model.predict_proba(testX)
    preds = [x[1] for x in preds]
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
    # return np.round(acc,4), np.round(specificity,4), np.round(sensitivity,4), np.round(aucscore,4), np.round(auprc,4), np.round(balanceacc,4), np.round(f1score,4), np.round(ppv,4), np.round(npv,4)

#Fix model name for visualisation

def modelNameFixer(x):
    if 'liblinear' in x:
        return 'Lasso'
    elif 'GaussianNB' in x:
        return 'GNB'
    elif 'SVC' in x:
        return 'SVC'
    elif 'RandomForest' in x:
        return 'RF'
    elif 'XGB' in x:
        return 'XGBoost'
    elif 'DecisionTree' in x:
        return 'DT'
    else:
        return 'LR'
    
    
# instantiate the model (using the default parameters)
def build_models (X_train, y_train, target_outcome, params_dict, model_folder, fold):
    models = [] #list to store all the models
    print("Building models . . . .")

    #LR
    model = 'LR'
    params = params_dict[(params_dict['outcome']==target_outcome)&(params_dict['model']==model)]['params'].tolist()[0]
    # params = eval(params)
    print(params)
    lr_model = LogisticRegression(class_weight='balanced', C = params['C'], max_iter=params['max_iter'], solver=params['solver'], random_state=random_state)
    lr_model.fit(X_train,y_train)
    pickle.dump(lr_model, open(model_folder+ target_outcome + '_'+ model + str(fold) + '.sav', 'wb'))
    models.append([model + str(fold), target_outcome, y_train.value_counts()[1]/y_train.value_counts()[0]]) 
    print("LR done")

    # #Lasso
    # model = 'Lasso'
    # params = params_dict[(params_dict['outcome']==target_outcome)&(params_dict['model']==model)]['params'].tolist()[0]
    # # params = eval(params)
    # print(params)
    # lasso_model = LogisticRegression(class_weight='balanced',  C = params['C'], max_iter=params['max_iter'], penalty='l1', solver=params['solver'], random_state=random_state) #only the LIBLINEAR and SAGA (added in v0.19) solvers handle the L1 penalty
    # lasso_model.fit(X_train, y_train)
    # pickle.dump(lasso_model, open(model_folder+ target_outcome + '_'+ model + str(fold) + '.sav', 'wb'))
    # models.append([model + str(fold), target_outcome, y_train.value_counts()[1]/y_train.value_counts()[0]])
    # print("Lasso done")
    
    #Elastics
    model = 'ElasticNet'
    params = params_dict[(params_dict['outcome']==target_outcome)&(params_dict['model']==model)]['params'].tolist()[0]
    # params = eval(params)
    print(params)
    elastics_model = LogisticRegression(class_weight='balanced', solver='saga', l1_ratio=params['l1_ratio'], max_iter=params['max_iter'],  penalty = 'elasticnet', random_state=random_state)
    elastics_model.fit(X_train, y_train)
    pickle.dump(elastics_model, open(model_folder+ target_outcome + '_'+ model + str(fold) + '.sav', 'wb'))
    models.append([model + str(fold), target_outcome, y_train.value_counts()[1]/y_train.value_counts()[0]])
    print("Elastics done")


    #DT
    model = 'DT'
    params = params_dict[(params_dict['outcome']==target_outcome)&(params_dict['model']==model)]['params'].tolist()[0]
    # params = eval(params)
    print(params)
    dt_model = DecisionTreeClassifier(class_weight='balanced', max_depth=params['max_depth'], criterion=params['criterion'], splitter=params['splitter'], random_state=random_state)
    dt_model.fit(X_train, y_train)
    pickle.dump(dt_model, open(model_folder+ target_outcome + '_'+ model + str(fold) + '.sav', 'wb'))    
    models.append([model + str(fold), target_outcome, y_train.value_counts()[1]/y_train.value_counts()[0]])
    print("DT done")

    #RF
    model = 'RF'
    params = params_dict[(params_dict['outcome']==target_outcome)&(params_dict['model']==model)]['params'].tolist()[0]
    # params = eval(params)
    print(params)
    rf_model = RandomForestClassifier(class_weight='balanced', max_depth=params['max_depth'], 
                                      criterion=params['criterion'], n_estimators=params['n_estimators'], 
                                      min_samples_split=params['min_samples_split'],
                                      min_samples_leaf=params['min_samples_leaf'],
                                      max_features=params['max_features'],
                                      bootstrap=params['bootstrap'], 
                                      random_state=random_state)
    rf_model.fit(X_train, y_train)
    pickle.dump(rf_model, open(model_folder+ target_outcome + '_'+ model + str(fold) + '.sav', 'wb'))     
    models.append([model + str(fold), target_outcome, y_train.value_counts()[1]/y_train.value_counts()[0]])
    print("RF done")

    #XGB
    model = 'XGB'
    params = params_dict[(params_dict['outcome']==target_outcome)&(params_dict['model']==model)]['params'].tolist()[0]
    # params = eval(params)
    print(params)
    scale_pos_ratio = y_train.value_counts()[0]/y_train.value_counts()[1]
    xgb_model = xgb.XGBClassifier(objective ='binary:logistic', tree_method = "hist", 
                                  n_estimators=params['n_estimators'],
                                  max_depth=params['max_depth'],
                                  learning_rate=params['learning_rate'],
                                  reg_alpha=params['reg_alpha'],
                                  reg_lambda=params['reg_lambda'],
                                  subsample=params['subsample'],
                                  colsample_bytree=params['colsample_bytree'],
                                  scale_pos_weight=params['scale_pos_weight'],
                                  device = "cuda", 
                                  verbosity = 3,
                                  importance_type = 'gain', random_state=random_state)
    # xgb_model = xgb.XGBClassifier(objective ='binary:logistic', learning_rate = 0.001, tree_method='gpu_hist', gpu_id=0,  verbosity = 0, random_state = 1234)
    xgb_model.fit(X_train,y_train)
    pickle.dump(xgb_model, open(model_folder+ target_outcome + '_'+ model + str(fold) + '.sav', 'wb')) 
    models.append([model + str(fold),  target_outcome, y_train.value_counts()[1]/y_train.value_counts()[0]])
    print("XGB done")
    
    return models
    # return [xgb_model]

    



X = pd.concat([crossValData[features_columns], gridSearchData[features_columns]]) #use all gridsearch and crossval sets
X_internalVal = internalEvaluationData[features_columns]
X_externalVal = externalEvaluationData[features_columns]


print(X.shape)
print(X_internalVal.shape)
print(X_externalVal.shape)


target_outcomes = [
    'outcome_12months',
    'outcome_3months', 'outcome_6months', 'outcome_9months', 
    
] 


#params
params_dict = pd.read_csv('../MODELS/BS_result_new.csv')
def process_params(param_items, best_param):
    a = eval(param_items)
    b = eval(best_param)
    c = {}
    for key, value in zip(a,b):
        c[key] = value
    return c

params_dict['params'] = params_dict.apply(lambda x: dict(eval(x.best_param[11:])), axis=1)




#EXECUTE model training

summary_result_internalVal = []
summary_result_externalVal = []

cols = ['model_name', 'fold', 'outcome', 'class_ratio', 'auc', 'auprc']
model_folder = '../MODELS/TestResultCat/'

start_time = time.time()
for target_outcome in target_outcomes:
    models = pd.DataFrame(columns=['modelname', 'target_outcome', 'class_ratio'])
    print(target_outcome)
    y = pd.concat([crossValData[target_outcome] , gridSearchData[target_outcome]])
    y_internalVal = internalEvaluationData[target_outcome]
    y_externalVal = externalEvaluationData[target_outcome]
    fold = '' #no fold for the FINAL MODEL - trained on crossval and gridsearch sets
    
    
    #Build models -> it can be commented if the models have been trained
    models_temp = pd.DataFrame(build_models(X, y, target_outcome, params_dict, model_folder, fold), columns=['modelname', 'target_outcome', 'class_ratio'])
    models = pd.concat([models,models_temp]).reset_index(drop=True)


    #evaluate model
    for modelname, target_outcome, classratio in models.values:
        # print('======================================================================')
        print(modelname)
        model = pickle.load(open(model_folder + target_outcome + '_'+ modelname + '.sav', 'rb'))       
        summary_result_internalVal.append((modelname, fold, target_outcome, classratio, ) + summariseResult (X_internalVal, y_internalVal, model) )       
        summary_result_externalVal.append((modelname, fold, target_outcome, classratio, ) + summariseResult (X_externalVal, y_externalVal, model) )       
        torch.cuda.empty_cache()


summary_result_internalVal = pd.DataFrame(summary_result_internalVal, columns=cols)
summary_result_internalVal['model_num'] = summary_result_internalVal.index
summary_result_internalVal.to_csv('../MODELS/internalValResultCat.csv', index_label=False, index=False)

summary_result_externalVal = pd.DataFrame(summary_result_externalVal, columns=cols)
summary_result_externalVal['model_num'] = summary_result_externalVal.index
summary_result_externalVal.to_csv('../MODELS/externalValResultCat.csv', index_label=False, index=False)

# Calculate and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
