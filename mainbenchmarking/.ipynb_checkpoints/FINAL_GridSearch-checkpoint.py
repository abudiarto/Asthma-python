import pandas as pd
import numpy as np
import sklearn

#statistics
from scipy.stats import chi2_contingency, ttest_ind

#preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler

#hyperparameter search
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

#internal validation
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold, cross_val_score, GridSearchCV, PredefinedSplit, RandomizedSearchCV


#performance metrices
from sklearn.metrics import make_scorer, confusion_matrix, classification_report, f1_score, balanced_accuracy_score, r2_score, auc, average_precision_score, roc_auc_score, recall_score, roc_curve, accuracy_score

#Models selection
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# from cuml.svm import SVC #gpu-powered SVM

import cupy as cp

#save and load trained model
import pickle

#visualisation
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

import os
import time
random_state = 42



# Data loader
# features = pd.read_csv("../FinalData/cleaned_features_11072023.csv")
gridSearchData, crossValData, internalEvaluationData, externalEvaluationData = pickle.load(open('../Clean_data/dataset_scaled_2vs1_25102024.sav', 'rb'))
outcomes = pd.read_csv("../../Clean_data/cleaned_outcomes_24102024.csv")

print(gridSearchData.shape)
print(outcomes.shape)
gridSearchData.isna().sum()


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
                   
                   'age_cat', 'ICS_medication_possesion_ratio_cat', 'numOCS_cat', 'numOCSEvents_cat', 'numOCSwithLRTI_cat', 'numAcuteRespEvents_cat', 
                   'numAntibioticsEvents_cat', 'numAntibioticswithLRTI_cat', 'numAsthmaAttacks_cat', 'numHospEvents_cat', 'numPCS_cat', 'numPCSAsthma_cat', 
                   'numAsthmaManagement_cat', 'numAsthmaReview_cat', 'numAsthmaMedReview_cat', 'numAsthmaReviewRCP_cat', 'average_daily_dose_ICS_cat', 
                   'prescribed_daily_dose_ICS_cat', #use continous vars instead
                   
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



#GRID SEARCH

start_time = time.time()
X = gridSearchData[features_columns]
# X = cudf.DataFrame(X)
outcomes = [
    # 'outcome_12months',
#             'outcome_3months', 
#             'outcome_6months', 
            'outcome_9months', 
           ] 



output = []
for outcome in outcomes:
    print(outcome)
    y = gridSearchData[outcome]
    scale_pos_ratio = y.value_counts()[0]/y.value_counts()[1]
    
    #MODELS
    lr_model = LogisticRegression(class_weight='balanced', random_state=random_state)
    lasso_model = LogisticRegression(class_weight='balanced', penalty='l1', random_state=random_state) #only the LIBLINEAR and SAGA (added in v0.19) solvers handle the L1 penalty
    elastic_model = LogisticRegression(solver='saga', class_weight='balanced', penalty = 'elasticnet', random_state=random_state)
    dt_model = DecisionTreeClassifier(class_weight='balanced', random_state=random_state)
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    xgb_model = xgb.XGBClassifier(objective ='binary:logistic', tree_method = "hist", device = "cuda", verbosity = 3,
                                     importance_type = 'gain', random_state=random_state)

    #PARAMS
    lr_params = {'solver': ['liblinear', 'newton-cholesky'],
                 'C': [0.1, 1.0, 10.0],
                 'max_iter': [80, 100, 120]}
    
    lasso_params = {'solver': ['saga', 'liblinear'],
                    'C': [0.1, 1, 10],
                    'max_iter': [80, 100, 120]}
    
    elastic_params = {'l1_ratio': Real(0.1, 1, 'uniform'),
                      'max_iter': [80, 100, 120]}
    
    dt_params = {'criterion':["gini", "entropy"],
                 'splitter': ['best', 'random'],
                'max_depth': Integer(2,100,"uniform")}
    
    rf_params = {'criterion':["gini", "entropy"],
                 'n_estimators': Integer(100,1000,"uniform"),
                'max_depth': Integer(2,100,"uniform"),
                'min_samples_split': Integer(2,10,"uniform"),
                'min_samples_leaf': Integer(2,100,"uniform"),
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]}
    
    xgb_params = {'n_estimators': Integer(100,1000,"uniform"),
                'max_depth': Integer(2,100,"uniform"),
                 'learning_rate': Real(1e-3, 3e-1, 'log-uniform'),
                 'reg_alpha': Real(0.1, 10, 'log-uniform'),
                 'reg_lambda': Real(0.1, 10, 'log-uniform'),
                 'subsample': Real(0.5, 1, 'uniform'),
                 'colsample_bytree': Real(0.5, 1, 'uniform'),
                 'scale_pos_weight': Real(scale_pos_ratio-2, scale_pos_ratio+2, 'uniform')}

    #Models and params in DICT
    models_to_be_trained = [
        # {'model_name': 'LR', 'model': lr_model, 'params': lr_params},
        # {'model_name': 'Lasso', 'model': lasso_model, 'params': lasso_params},
        # {'model_name': 'ElasticNet', 'model': elastic_model, 'params': elastic_params},
        {'model_name': 'XGB', 'model': xgb_model, 'params': xgb_params},
        {'model_name': 'DT', 'model': dt_model, 'params': dt_params},
        # {'model_name': 'RF', 'model': rf_model, 'params': rf_params},
        
    ]
    

    for item in models_to_be_trained:
        print(item['model_name'])
        gs = BayesSearchCV(item['model'],
                          search_spaces=item['params'],
                          scoring='roc_auc',
                           n_iter = 50,
                          cv=3,
                          verbose=3, 
                           n_jobs=5,
                           n_points=10,
                            random_state = random_state)
        # if item['model_name']=='XGB':
        #     X_xgb = cp.array(X)
        #     y_xgb = cp.array(y)
        #     gs.fit(X_xgb.get(), y_xgb.get())
        gs.fit(X, y)
        # output.append([outcome, item['model_name'], gs.best_score_, gs.best_params_])
        pickle.dump(gs.cv_results_, open('../MODELS/BS/' + outcome.split('_')[-1] + '_' + item['model_name'] + '.sav', 'wb'))

# print(pd.DataFrame(output, columns=['outcome', 'model', 'best_score', 'best_param']))
# pd.DataFrame(output, columns=['outcome', 'model', 'best_score', 'best_param']).to_csv('../MODELS/BS_result.csv', index = False, index_label=False)


# Calculate and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")





























