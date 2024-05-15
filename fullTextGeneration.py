import pickle
import _pickle as cPickle
import pandas as pd
import numpy as np


def fix_ethnic(x):
    if x == 'not_recorded':
        return 'unknown ethnicity'
    elif x == 'White - ethnic group':        
        return 'White ethnicity'
    elif x == 'Other ethnic group':
        return 'other ethnicity'
    elif x == 'Asian - ethnic group':
        return 'Asian ethnicity'
    elif x == 'Black - ethnic group':
        return 'Black ethnicity'
    elif x == 'Mixed ethnic census group':
        return 'Mixed ethnicity'
    
def fix_BMI(x):
    if x == 'normal':
        return 'normal BMI'
    elif x == 'obese':        
        return 'obese BMI'
    elif x == 'overweight':
        return 'overweight BMI'
    elif x == 'not recorded':
        return 'unknown BMI'
    elif x == 'underweight':
        return 'underweight BMI'




data = pickle.load(open('../Clean_data/clinical_data.sav', 'rb'))
dataDesc = pickle.load(open('../Clean_data/NewDataWithDesc.sav', 'rb'))
trainingData, validationData, internalEvaluationData, evaluationData, evaluationDataWales, evaluationDataScotland = pickle.load(open('../FinalData/dataset_2vs1_15022024.sav', 'rb'))
features = pickle.load(open('../FinalData/cleaned_features_2vs1_15022024.sav', 'rb'))
trainingSets = [trainingData, validationData, internalEvaluationData, evaluationData, evaluationDataWales, evaluationDataScotland]
trainingSetsName = ['trainingData', 'validationData', 'internalEvaluationData', 'evaluationData', 'evaluationDataWales', 'evaluationDataScotland']
print('data loaded successfully')


for set, name in zip(trainingSets, trainingSetsName):
    print(name)
    newTrainingData = set
    newTrainingData = trainingData[['patid']].merge(
        dataDesc, on='patid', how='inner').reset_index(drop=True)
    newTrainingData = newTrainingData.merge(features[['patid', 'sex', 'age', 'cat_BMI', 'smokingStatus', 'ethnic_group', 
                                                      'imd_decile', 'OutputAreaClassification']], on='patid',how='inner').reset_index(drop=True)
    print(newTrainingData.shape)

    newTrainingData['sex'] = newTrainingData.sex.apply(lambda x: 'female' if x == 0 else 'male')
    newTrainingData['age'] = newTrainingData.age.apply(lambda x: str(x) + ' years old')
    newTrainingData['OutputAreaClassification'] = newTrainingData.OutputAreaClassification.apply(lambda x: x.replace(';', ', '))
    newTrainingData['imd_decile'] = newTrainingData.imd_decile.apply(lambda x: 'imd decile ' + str(x) + ' area')
    newTrainingData['cat_BMI'] = newTrainingData.cat_BMI.apply(lambda x: fix_BMI(x))
    newTrainingData['ethnic_group'] = newTrainingData.ethnic_group.apply(lambda x: fix_ethnic(x))
    newTrainingData['profile'] = newTrainingData.apply(lambda x: x.ethnic_group + ' ' + x.sex + ', ' + x.age + ', ' 
                                                           + x.smokingStatus + ', with ' + x.cat_BMI + ', and lives in a ' 
                                                           + x.OutputAreaClassification +' area.', axis=1)
    newTrainingData['full_text'] = newTrainingData.apply(lambda x: x.profile + ' ' + x.TERM60, axis=1)
    newTrainingData['seq_length'] = newTrainingData.full_text.apply(lambda x: len(x.split()))
    newTrainingData['profile_length'] = newTrainingData.profile.apply(lambda x: len(x.split()))

    newTrainingData = newTrainingData.merge(data, on='patid', how='left').reset_index(drop=True)
    print(newTrainingData.shape)

    pickle.dump(newTrainingData[['patid', 'full_text', '3months', '6months', '9months',
           '12months']], open('../Clean_data/clinical+profile_' + name + '.sav', 'wb'))
    print('data saved successfully')