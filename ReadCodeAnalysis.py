import pandas as pd
import cudf
import pyreadr
import numpy as np
import torch
import pickle
from os import listdir
from os.path import isfile, join


#patientData
patient = pyreadr.read_r('../ServerData_13Oct2020/d_patient_overall.Rdata')
practice = pyreadr.read_r('../ServerData_13Oct2020/d_practice.Rdata')
patient = patient['d_patient_overall']
practice = practice['d_practice']
patient = patient[['patid', 'practice_id']].merge(practice[['practice_id', 'Country']], how='left', on='practice_id')

#specific codes
target_readcode = pd.read_csv('../FinalData/UniqueReadCodes.csv')


#load clinical information
path = '../ServerData_13Oct2020/'
clinical_files = [join(path, f) for f in listdir(path) if (isfile(join(path, f))) & ('f_clinical_part' in f)]
clinical = pyreadr.read_r('../ServerData_13Oct2020/f_clinical_part1.Rdata')
clinical = clinical['f_clinical_part']
clinical = clinical.dropna(subset=['code_id'])
clinical['event_date'] = pd.to_datetime(clinical['event_date'])
clinical = clinical.loc[(clinical['event_date'] >= '2016-01-01') & (clinical['event_date'] < '2018-01-01')]
# clinical = clinical[clinical.code_id.isin(target_readcode.readcodes.values)]
clinical = clinical.merge(patient[['patid', 'Country']], on='patid', how='left')
clinical = clinical[['patid', 'code_id', 'Country']]
for file in clinical_files:
    print(file)
    temp = pyreadr.read_r(file)
    temp = temp['f_clinical_part']
    temp = temp.dropna(subset=['code_id'])
    temp['event_date'] = pd.to_datetime(temp['event_date'])
    temp = temp.loc[(temp['event_date'] >= '2016-01-01') & (temp['event_date'] < '2018-01-01')]
    # temp = temp[temp.code_id.isin(target_readcode.readcodes.values)]
    temp = temp.merge(patient[['patid', 'Country']], on='patid', how='left')
    temp = temp[['patid', 'code_id', 'Country']]
    clinical = pd.concat([clinical, temp])
    clinical.reset_index(drop=True, inplace=True)
    
    
#load therapy information
path = '../ServerData_13Oct2020/'
therapy_files = [join(path, f) for f in listdir(path) if (isfile(join(path, f))) & ('f_therapy_part' in f)]
therapy = pyreadr.read_r('../ServerData_13Oct2020/f_therapy_part1.Rdata')
therapy = therapy['f_therapy_part']
therapy = therapy.dropna(subset=['code_id'])
therapy['event_date'] = pd.to_datetime(therapy['event_date'])
therapy = therapy.loc[(therapy['event_date'] >= '2016-01-01') & (therapy['event_date'] < '2018-01-01')]
# therapy = therapy[therapy.code_id.isin(target_readcode.readcodes.values)]
therapy = therapy.merge(patient[['patid', 'Country']], on='patid', how='left')
therapy = therapy[['patid', 'code_id', 'Country']]
for file in therapy_files:
    print(file)
    temp = pyreadr.read_r(file)
    temp = temp['f_therapy_part']
    temp = temp.dropna(subset=['code_id'])
    temp['event_date'] = pd.to_datetime(temp['event_date'])
    temp = temp.loc[(temp['event_date'] >= '2016-01-01') & (temp['event_date'] < '2018-01-01')]
    # temp = temp[temp.code_id.isin(target_readcode.readcodes.values)]
    temp = temp.merge(patient[['patid', 'Country']], on='patid', how='left')
    temp = temp[['patid', 'code_id', 'Country']]
    therapy = pd.concat([therapy, temp])
    therapy.reset_index(drop=True, inplace=True)

    
clinical = pd.pivot_table(data=clinical, values='patid', index='code_id', columns='Country', aggfunc=pd.Series.nunique)
clinical = clinical[['England','Scotland','Wales']]
clinical = clinical.fillna(0)
clinical['%England'] = clinical.apply(lambda x: x.England/sum([x.England, x.Scotland, x.Wales])*100, axis=1)
clinical['%Scotland'] = clinical.apply(lambda x: x.Scotland/sum([x.England, x.Scotland, x.Wales])*100, axis=1)
clinical['%Wales'] = clinical.apply(lambda x: x.Wales/sum([x.England, x.Scotland, x.Wales])*100, axis=1)
clinical['%patient'] = clinical.apply(lambda x: (sum([x.England, x.Scotland, x.Wales])/675260)*100, axis=1)
clinical.sort_values('%patient', ascending=False)


therapy = pd.pivot_table(data=therapy, values='patid', index='code_id', columns='Country', aggfunc=pd.Series.nunique)
therapy = therapy[['England','Scotland','Wales']]
therapy = therapy.fillna(0)
therapy['%England'] = therapy.apply(lambda x: x.England/sum([x.England, x.Scotland, x.Wales])*100, axis=1)
therapy['%Scotland'] = therapy.apply(lambda x: x.Scotland/sum([x.England, x.Scotland, x.Wales])*100, axis=1)
therapy['%Wales'] = therapy.apply(lambda x: x.Wales/sum([x.England, x.Scotland, x.Wales])*100, axis=1)
therapy['%patient'] = therapy.apply(lambda x: (sum([x.England, x.Scotland, x.Wales])/675260)*100, axis=1)
therapy.sort_values('%patient', ascending=False)

clinical.to_csv('../FinalData/pivotClinicalCodesbyCountry.csv')
therapy.to_csv('../FinalData/pivotTherapyCodesbyCountry.csv')