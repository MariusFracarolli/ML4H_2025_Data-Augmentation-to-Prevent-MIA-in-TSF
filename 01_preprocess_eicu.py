# %%
import numpy as np
import pandas as pd
import joblib

from tqdm import tqdm
tqdm.pandas()


# %% [markdown]
# # Preprocess data (exclusion gender, age, less than 24h)
# Get 77.704 stays out of 200859 stays

# %%
# load patients and exclude some
pat = pd.read_csv('../eicu-collaborative-research-database-2.0/patient.csv')
pat.loc[pat['age'] =='> 89', 'age'] = 91.4  # change '> 89' to 91.4
pat.loc[pd.isna(pat['age']), 'age'] = -3 # change unknown to -3
pat['age'] = pat['age'].astype(float)
print('Number of patients:',len(pat),'(no exclusion)')
pat = pat[pat['age']>17]
print('Number of patients:',len(pat),'(age exclusion)')
pat = pat[(pat['gender']=='Male')|(pat['gender']=='Female')]
print('Number of patients:',len(pat),'(age and gender exclusion)')
pat = pat[pat['unitdischargeoffset']>24*60-1]
print('Number of patients:',len(pat),'(age and gender and 24h exclusion)')
pat = pat[pat['unitdischargeoffset']>48*60-1]
print('Number of patients:',len(pat),'(age and gender and 48h exclusion)')

log_str = 'Experiment started' + str(len(pat))
with open('log_vars.csv', 'a') as file:
    print(log_str, file=file)


# %% [markdown]
# # Save data to csv (going through each eICU table)

# %% [markdown]
# The following tables ar done and must be controlled:
# - vitalP (7.9GB --> 8 vars, 7.6GB, 316.2M, 15min)
# - vitalAperiodic (1.0GB --> 3 vars, 0.9GB, 40.7M, 2min)
# - lab (2.4GB --> 47 vars, 350MB, 13.7M, 3min)
# - intakeOutput (1.9GB --> 3 vars, 49MB, 2.0M, 1min)
# - nurseCharting (11.5GB --> 12 vars, 1.3GB, 56.1M, 10min)
# - infD (0.3GB --> 36 vars, 84MB, 2.4M, 1min)
# 
# The following should be done:
# - medication (0.6GB)
# 
# The following are not yet reviewed:
# - respiratoryCharting (1.2GB)
# - physical exam (1.4GB)
# - nurseAssesment (2.3 GB)
# - nurseCare (1.2 GB)
# 

# %%
""" 1, vitalPeriodic
contains 17 different variables.
only 8 vars are collected, 
maybe add 'cvp', 'pasystolic','padiastolic', 'pamean', 'st1', 'st2', 'st3', 'icp'
these eight contain 316.485.347 single data from 12.086 to 76.662 patients 


"""
vitalP = pd.read_csv('../eicu-collaborative-research-database-2.0/vitalPeriodic.csv')
# ['vitalperiodicid', 'patientunitstayid', 'observationoffset', 'temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2',
#  'systemicsystolic', 'systemicdiastolic', 'systemicmean', 'pasystolic','padiastolic', 'pamean', 'st1', 'st2', 'st3', 'icp']

vitalP = vitalP[(vitalP['patientunitstayid'].isin(pat['patientunitstayid'])) & (vitalP['observationoffset']<24*60*7)& (vitalP['observationoffset']>=0)]  # reduce to included data

vitalP_list = pd.DataFrame()
for key in ['temperature','sao2','heartrate', 'respiration','etco2','systemicsystolic','systemicdiastolic','systemicmean']:
    aux = vitalP[vitalP[key]>0][['patientunitstayid','observationoffset', key]]
    aux['variable'] = key
    if key == 'systemicmean':
        aux['variable'] = 'MBP'
        aux = aux[aux[key]<= 375]
    elif key == 'systemicsystolic':
        aux['variable'] = 'SBP'
        aux = aux[aux[key]<= 375]
    elif key == 'systemicdiastolic':
        aux['variable'] = 'DBP'
        aux = aux[aux[key]<= 375]
    elif key == 'sao2':
        aux = aux[aux[key]<= 100]
        aux['variable'] = 'O2 Saturation'
    elif key == 'heartrate':
        aux = aux[aux[key]<= 390]
        aux['variable'] = 'HR'
    elif key == 'respiration':
        aux = aux[aux[key]<= 330]
        aux['variable'] = 'RR'
    elif key == 'temperature':
        aux = aux[aux[key].between(14.2,47)]
        aux['variable'] = 'Temperature'
    elif key == 'etco2':
        aux = aux[aux[key]<= 100]
        aux['variable'] = 'EtCO2'
    aux.rename(columns={key: 'value'}, inplace=True)
    vitalP_list = pd.concat([vitalP_list, aux], ignore_index=True)
    log_str =  'vitalPeriodic & ' + key + ' & ' + str(len(aux['patientunitstayid'].unique())) + ' & ' + str(len(aux)) + '//'
    print(log_str)
    with open('log_vars.csv', 'a') as file:
        print(log_str, file=file)
vitalP_list.rename(columns={'patientunitstayid':'stayid','observationoffset':'time','variable':'var'},inplace=True)
vitalP_list = vitalP_list[['stayid','time','var','value']]
vitalP_list.to_csv('vitalP_list.csv', index=False)
log_str =  'vitalPeriodic&                             & ' + str(len(vitalP_list['stayid'].unique())) + ' & ' + str(len(vitalP_list)) + '//'
with open('log_vars.csv', 'a') as file:
    print(log_str, file=file)
print('vitalP read with', len(vitalP_list), 'entries.')  # 342141929 (unfiltered)

# needs 16 minutes and 8.9 GB memory for data. 316.485.347
# 2.5 min for reading
vitalP_list.groupby('var').count().reset_index()[['var','value']]

# %%
"""2, vitalAperiodic
up to 10 variables
we use only 3: MBP, SBP, DBP
don't use: paop, cardiacoutput, cardiacinput, svr, svri, pvr, pvri
    contains 76.224 resp. 76.233 different patients (98%)
contains 40.781.278 different entries 
maybe data from vitalP are also included in these. (then delete vitalP SBP, MBP, DBP?)
"""

vitalA = pd.read_csv('../eicu-collaborative-research-database-2.0/vitalAperiodic.csv')
vitalA = vitalA[(vitalA['patientunitstayid'].isin(pat['patientunitstayid'])) & (vitalA['observationoffset']<24*60*7) & (vitalA['observationoffset']>=0)]  # reduce to included data
vitalA_list = pd.DataFrame()
for key in ['noninvasivesystolic','noninvasivediastolic', 'noninvasivemean']:
    aux = vitalA[vitalA[key]>0][['patientunitstayid','observationoffset', key]]
    # aux['variable'] = key
    if key == 'noninvasivemean':
        aux['variable'] = 'MBP'
    elif key == 'noninvasivesystolic':
        aux['variable'] = 'SBP'
    elif key == 'noninvasivediastolic':
        aux['variable'] = 'DBP'
    aux.rename(columns={key: 'value'}, inplace=True)
    vitalA_list = pd.concat([vitalA_list, aux], ignore_index=True)
    log_str =  'vitalAperiodic & ' + key + ' & ' + str(len(aux['patientunitstayid'].unique())) + ' & ' + str(len(aux)) + '//'
    print(log_str)
    with open('log_vars.csv', 'a') as file:
        print(log_str, file=file)
vitalA_list.rename(columns={'patientunitstayid':'stayid','observationoffset':'time','variable':'var'},inplace=True)
vitalA_list = vitalA_list[['stayid','time','var','value']]
vitalA_list.to_csv('vitalA_list.csv', index=False)
log_str =  'vitalAperiodic&                             & ' + str(len(vitalA_list['stayid'].unique())) + ' & ' + str(len(vitalA_list)) + '//'
with open('log_vars.csv', 'a') as file:
    print(log_str, file=file)
print('vitalA read with', len(vitalA_list), 'entries.')

# needs 2 minutes and 0.9 GB memory for data. 40.733.993
vitalA_list.groupby('var').count().reset_index()[['var','value']]

# %%
"""3, lab
use 51 variables
"""

lab = pd.read_csv('../eicu-collaborative-research-database-2.0/lab.csv')
lab = lab[(lab['patientunitstayid'].isin(pat['patientunitstayid'])) & (lab['labresultoffset']<24*60*7) & (lab['labresultoffset']>=0) & (lab['labresult']>=0)]  # reduce to included data

list_labname = ['bedside glucose', 'potassium', 'sodium', 'glucose', 'Hgb', 'chloride',                             6
       'Hct', 'creatinine', 'BUN', 'calcium', 'bicarbonate',                                                        5
       'platelets x 1000', 'WBC x 1000', 'RBC', 'MCV', 'MCHC', 'MCH', 'RDW',  #                                     7
       'anion gap', 'magnesium', '-lymphs', '-monos', #                                                             4
       '-basos', 'albumin', 'AST (SGOT)', 'ALT (SGPT)', #                                                           4
       'alkaline phos.', 'total bilirubin', 'phosphate', 'paO2', 'paCO2', 'pH',    #                                6
       'PT - INR', 'PT', 'FiO2', 'O2 Sat (%)', 'Base Excess', 'PTT',  #                                             6
       'lactate', 'troponin - I', 'fibrinogen', 'Total CO2', #                                                      4
       'Temperature', 'direct bilirubin', 'Respiratory Rate', 'urinary creatinine','-eos', '-polys', 'HCO3',  #     6
       'Base Deficit','MPV','total protein']

lab_list = pd.DataFrame()
for key in list_labname:
    aux = lab[lab['labname']==key][['patientunitstayid','labresultoffset', 'labresult','labname']]
    # rename and update labname
    len_aux = len(aux)
    if key=='bedside glucose':
        aux['labname'] ='Bedside Glucose' 
        aux = aux[aux['labresult'].between(0,2200)]
    elif key=='potassium':
        aux['labname'] ='Potassium' 
        aux = aux[aux['labresult'].between(0,15)]
    elif key=='sodium':
        aux['labname'] ='Sodium' 
        aux = aux[aux['labresult'].between(0,250)]
    elif key=='glucose':
        aux['labname'] ='Glucose' 
        aux = aux[aux['labresult'].between(0,2200)]
    elif key=='Hgb':
        aux['labname'] ='Hgb' 
        aux = aux[aux['labresult'].between(0,30)]
    elif key=='chloride':
        aux['labname'] ='Chloride' 
        aux = aux[aux['labresult'].between(0, 200)]
    elif key=='Hct':
        aux = aux[aux['labresult'].between(0,100)]
    elif key=='creatinine':
        aux['labname'] ='Creatinine (Blood)' 
        aux = aux[aux['labresult'].between(0,66)]
    elif key=='BUN':
        aux = aux[aux['labresult'].between(0,275)]
    elif key=='calcium':
        aux['labname'] ='Calcium' 
        aux = aux[aux['labresult'].between(0, 40)] # like Calcium Total
    elif key=='bicarbonate':
        aux['labname'] ='Bicarbonate' 
        aux = aux[aux['labresult'].between(0,66)]
    elif key=='HCO3':
        aux['labname'] ='Bicarbonate' 
        aux = aux[aux['labresult'].between(0,66)]
    elif key=='platelets x 1000':
        aux['labname'] ='Platelets' 
        aux = aux[aux['labresult'].between(0,2200)]
    elif key=='WBC x 1000':
        aux['labname'] ='WBC' 
        aux = aux[aux['labresult'].between(0,1100)]
    elif key=='RBC':
        aux = aux[aux['labresult'].between(0,14)]
    elif key=='MCV':
        aux = aux[aux['labresult'].between(0,150)]
    elif key=='MCHC':
        aux = aux[aux['labresult'].between(0,50)]
    elif key=='MCH':
        aux = aux[aux['labresult'].between(0, 50)]
    elif key=='RDW':
        aux = aux[aux['labresult'].between(0,37)]
    elif key=='anion gap':
        aux['labname'] ='Anion Gap' 
        aux = aux[aux['labresult'].between(0,55)]
    elif key=='magnesium':
        aux['labname'] ='Magnesium' 
        aux = aux[aux['labresult'].between(0,22)]
    elif key=='-lymphs':
        aux['labname'] ='Lymphocytes' 
        aux = aux[aux['labresult'].between(0,100)]
    elif key=='-monos':
        aux['labname'] ='Monocytes' 
        aux = aux[aux['labresult'].between(0,100)]
    elif key=='-basos':
        aux['labname'] ='Basophils' 
        aux = aux[aux['labresult'].between(0,100)]
    elif key=='albumin':
        aux['labname'] ='Albumin' 
        aux = aux[aux['labresult'].between(0,10)]
    elif key=='AST (SGOT)':
        aux['labname'] ='AST' 
        aux = aux[aux['labresult'].between(0, 22000)]
    elif key=='ALT (SGPT)':
        aux['labname'] ='ALT' 
        aux = aux[aux['labresult'].between(0, 11000)]
    elif key=='alkaline phos.':
        aux['labname'] ='ALP' 
        aux = aux[aux['labresult'].between(0,4000)]
    elif key=='total bilirubin':
        aux['labname'] ='Bilirubin (Total)' 
        aux = aux[aux['labresult'].between(0,66)]
    elif key=='phosphate':
        aux['labname'] ='Phosphate' 
        aux = aux[aux['labresult'].between(0,22)]
    elif key=='paO2':
        aux['labname'] ='PaO2' 
        aux = aux[aux['labresult'].between(0,770)]
    elif key=='paCO2':
        aux['labname'] ='PaCO2' 
        aux = aux[aux['labresult'].between(0,220)]
    elif key=='pH':
        aux['labname'] ='pH' 
        aux = aux[aux['labresult'].between(0,14)]
    elif key=='PT - INR':
        aux['labname'] ='INR' 
        aux = aux[aux['labresult'].between(0,150)]
    elif key=='PT':
        aux = aux[aux['labresult'].between(0,150)]
    elif key=='FiO2':
        aux.loc[aux['labresult'] > 1000, 'labresult'] /= 10000
        aux.loc[aux['labresult'] > 100, 'labresult'] /= 1000
        aux.loc[aux['labresult'] > 10, 'labresult'] /= 100
        aux.loc[aux['labresult'] > 1, 'labresult'] /= 10
        aux = aux[aux['labresult'].between(0.2,1)]
    elif key=='O2 Sat (%)':
        aux['labname'] ='O2 Saturation' 
        aux = aux[aux['labresult'].between(0,100)]
    elif key=='Base Excess':
        aux = aux[aux['labresult'].between(-31, 28)]
    elif key=='PTT':
        aux = aux[aux['labresult'].between(0,150)]
    elif key=='lactate':
        aux['labname'] ='Lactate' 
        aux = aux[aux['labresult'].between(0,33)]
    elif key=='troponin - I':
        aux['labname'] ='Troponin - I' 
        aux = aux[aux['labresult'].between(0,15000)]  # not available
    elif key=='fibrinogen':
        aux['labname'] ='Fibrinogen' 
        aux = aux[aux['labresult'].between(0,15000)]  # not available
    elif key=='Total CO2':
        aux['labname'] ='CO2 (Total)' 
        aux = aux[aux['labresult'].between(0,65)]
    elif key=='Temperature':
        aux = aux[aux['labresult'].between(14.2,47)]
    elif key=='direct bilirubin':
        aux['labname'] ='Bilirubin (Direct)' 
        aux = aux[aux['labresult'].between(0,66)]
    elif key=='Respiratory Rate':
        aux['labname'] ='RR' 
        aux = aux[aux['labresult'].between(0,330)]
    elif key=='urinary creatinine':
        aux['labname'] ='Creatinine (Urine)' 
        aux = aux[aux['labresult'].between(0,650)]
    elif key=='-eos':
        aux['labname'] ='Eoisinophils' 
        aux = aux[aux['labresult'].between(0,100)]
    elif key=='-polys':
        aux['labname'] ='Neutrophils' 
        aux = aux[aux['labresult'].between(0,100)]
    elif key=='total protein':
        aux['labname'] ='Protein (Total)' 
        # aux = aux[aux['labresult'].between(0,100)] # (0 - 16)
    elif key=='MPV':
        aux = aux[aux['labresult'].between(4,17)]  # deletes 9 & 3 values
    print(aux['labname'].iloc[0], len(aux), len_aux-len(aux))
    log_str =  'lab & ' + key + ' & ' + str(len(aux['patientunitstayid'].unique())) + ' & ' + str(len(aux)) + ' & ' + str(len_aux-len(aux)) + '//'
    with open('log_vars.csv', 'a') as file:
        print(log_str, file=file)
    lab_list = pd.concat([lab_list, aux], ignore_index=True)
lab_list.rename(columns={'patientunitstayid':'stayid','labresultoffset':'time','labname':'var','labresult':'value'},inplace=True)
lab_list = lab_list[['stayid','time','var','value']]
lab_list.to_csv('lab_list.csv', index=False)
log_str =  'lab &                             & ' + str(len(lab_list['stayid'].unique())) + ' & ' + str(len(lab_list)) + '//'
with open('log_vars.csv', 'a') as file:
    print(log_str, file=file)
print('lab read with', len(lab_list), 'entries.')

# 2.5 min, 350 MB, 14.558.261 entries
lab_list.groupby('var').count().reset_index()[['var','value']]

# %%
"""
4, respiratory Charting
FiO2                       & 42720 & 1897060//
"""
respCh = pd.read_csv('../eicu-collaborative-research-database-2.0/respiratoryCharting.csv')
respCh = respCh[(respCh['patientunitstayid'].isin(pat['patientunitstayid'])) & (respCh['respchartoffset']<24*60*7+1) & (respCh['respchartoffset']>=0)]
FiO2_list = respCh[respCh['respchartvaluelabel'].isin(['FiO2', 'FIO2 (%)'])]
FiO2_list['respchartvalue'] = pd.to_numeric(FiO2_list['respchartvalue'], errors='coerce')
FiO2_list = FiO2_list.dropna(subset=['respchartvalue'])
FiO2_list = FiO2_list[FiO2_list['respchartvalue'] >=0]
FiO2_list.loc[FiO2_list['respchartvalue'] > 100, 'respchartvalue'] /= 1000
FiO2_list.loc[FiO2_list['respchartvalue'] > 10, 'respchartvalue'] /= 100
FiO2_list.loc[FiO2_list['respchartvalue'] > 1, 'respchartvalue'] /= 10
FiO2_list.rename(columns={'patientunitstayid':'stayid','respchartoffset':'time','respchartvaluelabel':'var','respchartvalue':'value'},inplace=True) 
FiO2_list = FiO2_list[['stayid','time','var','value']]
FiO2_list['var'] = 'FiO2' # added 240319_1448
len_fio2 = len(FiO2_list)
FiO2_list = FiO2_list[FiO2_list['value'].between(0.2,1)]

log_str =  'respiratoryCharting &  FiO2                       & ' + str(len(FiO2_list['stayid'].unique())) + ' & ' + str(len(FiO2_list)) + '//'

with open('log_vars.csv', 'a') as file:
    print(log_str, file=file)

FiO2_list.to_csv('respCh_list.csv', index=False)
print('respiratoryCharting read with', len(FiO2_list), 'entries.')
print(log_str, len(FiO2_list)-len_fio2)

# %%
"""5, intakeOutput
use only 3: urine, bodyweight, stool 
bodyweight maybe not necessary
2.072.573 entries, mostly urine 1,7M

"""
inOu = pd.read_csv('../eicu-collaborative-research-database-2.0/intakeOutput.csv')
inOu = inOu[(inOu['patientunitstayid'].isin(pat['patientunitstayid'])) & (inOu['intakeoutputentryoffset']<24*60*7) & (inOu['intakeoutputentryoffset']>=0)]  # reduce to included data

inOu_list = pd.DataFrame()
for key in ['Urine','Bodyweight (kg)','Stool']:
    aux = inOu[inOu['celllabel']==key][['patientunitstayid','intakeoutputentryoffset', 'celllabel','cellvaluenumeric']]
    print(key, len(aux))
    # aux.rename(columns={key: 'value'}, inplace=True)
    # print(aux)
    if key == 'Urine':
        aux = aux[(aux['cellvaluenumeric']>=0) & (aux['cellvaluenumeric']<=2500)]
    elif key == 'Bodyweight (kg)':
        aux = aux[(aux['cellvaluenumeric']>=0) & (aux['cellvaluenumeric']<=300)]
    elif key == 'Stool':
        aux = aux[(aux['cellvaluenumeric']>=0) & (aux['cellvaluenumeric']<=4000)]
    inOu_list = pd.concat([inOu_list, aux], ignore_index=True)
    log_str =  'inputOutput & ' + key + ' & ' + str(len(aux['patientunitstayid'].unique())) + ' & ' + str(len(aux)) + '//'
    with open('log_vars.csv', 'a') as file:
        print(log_str, file=file)
inOu_list.rename(columns={'patientunitstayid':'stayid','intakeoutputentryoffset':'time','celllabel':'var','cellvaluenumeric':'value'}, inplace=True)
inOu_list = inOu_list[['stayid','time','var','value']]
inOu_list.to_csv('inOu_list.csv', index=False)
log_str =  'inputOutput &                             & ' + str(len(inOu_list['stayid'].unique())) + ' & ' + str(len(inOu_list)) + '//'
with open('log_vars.csv', 'a') as file:
    print(log_str, file=file)
print('inOu read with', len(inOu_list), 'entries.')

# 1 min, 49MB, 1.952.898 entries

# %%
"""
6, nurseCharting
get 14 values (GCSx3, HR, RR, O2 Sat, DBP,SBP,MBP, temperature, O2 L/$, Bedside Glucose)
"""

nursCh = pd.read_csv('../eicu-collaborative-research-database-2.0/nurseCharting.csv')
nursCh = nursCh[(nursCh['patientunitstayid'].isin(pat['patientunitstayid'])) & (nursCh['nursingchartoffset']<24*60*7) & (nursCh['nursingchartoffset']>=0)]
numeric_mask = pd.to_numeric(nursCh['nursingchartvalue'], errors='coerce').notna()
nursCh = nursCh[numeric_mask]
nursCh['nursingchartvalue'] = pd.to_numeric(nursCh['nursingchartvalue'])
# print('Number of patients:',len(pat),'(age and gender and 24h exclusion)')

key_list = [
    ['Glasgow coma score','Eyes', 'GCS eye'],
    ['Glasgow coma score','Motor', 'GCS motor'],
    ['Glasgow coma score','Verbal', 'GCS verbal'],
    ['Heart Rate','Heart Rate','HR'],
    ['Respiratory Rate','Respiratory Rate','RR'],
    ['O2 Saturation','O2 Saturation','O2 Saturation'],
    ['Non-Invasive BP','Non-Invasive BP Diastolic','DBP'],
    ['Non-Invasive BP','Non-Invasive BP Systolic','SBP'],
    ['Non-Invasive BP','Non-Invasive BP Mean','MBP'],
    ['Temperature','Temperature (C)','Temperature'],
    ['Bedside Glucose','Bedside Glucose','Bedside Glucose'],
    ['O2 L/%','O2 L/%','O2 L/%']
    # ['O2 Admin Device','O2 Admin Device','O2 Admin Device']
]

nursCh_list = pd.DataFrame()
for key in key_list:
    aux = nursCh[(nursCh['nursingchartcelltypevallabel']==key[0]) & (nursCh['nursingchartcelltypevalname']==key[1])][['patientunitstayid','nursingchartoffset','nursingchartvalue']]
    # print(key, len(aux))
    aux['label'] = key[2]
    if key[2] == 'HR':
        aux = aux[(aux['nursingchartvalue']>=0) & (aux['nursingchartvalue']<=390)]
    elif key[2] == 'RR':
        aux = aux[(aux['nursingchartvalue']>=0) & (aux['nursingchartvalue']<=330)]
    elif key[2] == 'O2 Saturation':
        aux = aux[(aux['nursingchartvalue']>=0) & (aux['nursingchartvalue']<=100)]
    elif key[2] == 'DBP':
        aux = aux[(aux['nursingchartvalue']>=0) & (aux['nursingchartvalue']<=375)]
    elif key[2] == 'SBP':
        aux = aux[(aux['nursingchartvalue']>=0) & (aux['nursingchartvalue']<=375)]
    elif key[2] == 'MBP':
        aux = aux[(aux['nursingchartvalue']>=0) & (aux['nursingchartvalue']<=375)]
    elif key[2] == 'Temperature':
        aux = aux[(aux['nursingchartvalue']>=14.2) & (aux['nursingchartvalue']<=47)]
    elif key[2] == 'Bedside Glucose':
        aux = aux[(aux['nursingchartvalue']>=0) & (aux['nursingchartvalue']<=2200)]
    elif key[2] == 'O2 L/%':
        aux = aux[(aux['nursingchartvalue']>=0) & (aux['nursingchartvalue']<=100)]
    nursCh_list = pd.concat([nursCh_list, aux], ignore_index=True)
    log_str =  'nurseCharting & ' + key[2] + ' & ' + str(len(aux['patientunitstayid'].unique())) + ' & ' + str(len(aux)) + '//'
    with open('log_vars.csv', 'a') as file:
        print(log_str, file=file)
nursCh_list.rename(columns={'patientunitstayid':'stayid','nursingchartoffset':'time','label':'var','nursingchartvalue':'value'},inplace=True)
nursCh_list = nursCh_list[['stayid','time','var','value']]
nursCh_list.to_csv('nursCh_list.csv', index=False)
log_str =  'nurseCharting &                             & ' + str(len(nursCh_list['stayid'].unique())) + ' & ' + str(len(nursCh_list)) + '//'
with open('log_vars.csv', 'a') as file:
    print(log_str, file=file)
print('nursCh read with', len(nursCh_list), 'entries.')
nursCh_list.groupby('var').count().reset_index()[['var','value']]

# 10 min, 1.4GB, 56.147.604 entries, 5 min without csv

# %%
"""
7, physicalExam
3 vars (GCS)
325.872 data points, 
"""
physEx = pd.read_csv('../eicu-collaborative-research-database-2.0/physicalExam.csv')
physEx = physEx[(physEx['patientunitstayid'].isin(pat['patientunitstayid'])) & (physEx['physicalexamoffset']<24*60*7+1) & (physEx['physicalexamoffset']>=0)]
GCS_values = physEx[physEx['physicalexampath'].str.contains('GCS', case=False)]
eye_values = GCS_values[GCS_values['physicalexampath'].str.contains('eye', case=False)]
eye_values.loc[:,'labname'] = 'GCS eye'
motor_values = GCS_values[GCS_values['physicalexampath'].str.contains('eye', case=False)]
motor_values.loc[:,'labname'] = 'GCS motor'
verbal_values = GCS_values[GCS_values['physicalexampath'].str.contains('eye', case=False)]
verbal_values.loc[:,'labname'] = 'GCS verbal'
physEx_list = pd.concat([eye_values,motor_values,verbal_values], ignore_index=True)
physEx_list = physEx_list[['patientunitstayid','physicalexamoffset','physicalexamvalue', 'labname']]
physEx_list.rename(columns={'patientunitstayid':'stayid','physicalexamoffset':'time','labname':'var','physicalexamvalue':'value'},inplace=True)
physEx_list = physEx_list[['stayid','time','var','value']]
physEx_list.to_csv('physEx_list.csv', index=False)
log_str =  'nurseCharting &                             & ' + str(len(physEx_list['stayid'].unique())) + ' & ' + str(len(physEx_list)) + '//'
with open('log_vars.csv', 'a') as file:
    print(log_str, file=file)
print('physExamination read with', len(physEx_list), 'entries.')

# %%
""" 
8, infusionDrug
36 variables, thereby
4 SOFA relevant drugs in 10 vars,
12 further drugs in 26 vars
    Amio_list, Fentanyl_dose1, Fentanyl_dose2, Fentanyl_dose3, Furosemide_list, 
    Heparin_dose1, Heparin_dose2, Heparin_dose3, Heparin_dose4, 
    Insulin_dose1, Insulin_dose2, Insulin_dose3, Insulin_dose4,
    Midazolam_dose1, Midazolam_dose2, Milrinone_dose1, Milrinone_dose2,
    Nitrogly_list, Nitrogly_list2, Nitroprus, Pantoprazole_list, 
    Propofol_dose1, Propofol_dose2, Propofol_dose3,
    Vasopressin_dose1, Vasopressin_dose2, Vasopressin_dose3])
2.352.542 data points, available 3.230.649
"""

infusionDrug = pd.read_csv('../eicu-collaborative-research-database-2.0/infusionDrug.csv')
infusionDrug = infusionDrug[(infusionDrug['patientunitstayid'].isin(pat['patientunitstayid'])) & (infusionDrug['infusionoffset']<24*60*7) & (infusionDrug['infusionoffset']>=0)]
infusionDrug['drugrate'] = pd.to_numeric(infusionDrug['drugrate'], errors='coerce')
# infusionDrug  # 3.230.649 entries

# %%
Dobu_ratio = infusionDrug[(infusionDrug['drugname']=='Dobutamine (mcg/kg/min)')][['patientunitstayid','infusionoffset','drugrate']]
Dobu_ratio.rename(columns={'drugrate':'value'}, inplace=True)
Dobu_ratio['variable'] = 'Dobutamine ratio'
print('Dobu_ratio', len(Dobu_ratio['patientunitstayid'].unique()), len(Dobu_ratio))

Dobu_dose = infusionDrug[(infusionDrug['drugname']=='Dobutamine (ml/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Dobu_dose.rename(columns={'drugrate':'value'}, inplace=True)
Dobu_dose['variable'] = 'Dobutamine dose'
print('Dobu_dose', len(Dobu_dose['patientunitstayid'].unique()), len(Dobu_dose))
print('   People with Dobu', len(Dobu_dose['patientunitstayid'].append(Dobu_ratio['patientunitstayid']).unique()), len(Dobu_dose['patientunitstayid'].unique())+len(Dobu_ratio['patientunitstayid'].unique()))


Dopa_ratio = infusionDrug[(infusionDrug['drugname']=='Dopamine (mcg/kg/min)')][['patientunitstayid','infusionoffset','drugrate']]
Dopa_ratio.rename(columns={'drugrate':'value'}, inplace=True)
Dopa_ratio['variable'] = 'Dopamine ratio'
print('Dopa_ratio', len(Dopa_ratio['patientunitstayid'].unique()), len(Dopa_ratio))

Dopa_dose = infusionDrug[(infusionDrug['drugname']=='Dopamine (ml/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Dopa_dose.rename(columns={'drugrate':'value'}, inplace=True)
Dopa_dose['variable'] = 'Dopamine dose'
print('Dopa_dose', len(Dopa_dose['patientunitstayid'].unique()), len(Dopa_dose))
print('   People with Dopa', len(Dopa_dose['patientunitstayid'].append(Dopa_ratio['patientunitstayid']).unique()), len(Dopa_dose['patientunitstayid'].unique())+len(Dopa_ratio['patientunitstayid'].unique()))


Epi_ratio = infusionDrug[(infusionDrug['drugname']=='Epinephrine (mcg/min)')][['patientunitstayid','infusionoffset','drugrate']]
Epi_ratio.rename(columns={'drugrate':'value'}, inplace=True)
Epi_ratio['variable'] = 'Epinephrine ratio'
print('Epi_ratio', len(Epi_ratio['patientunitstayid'].unique()), len(Epi_ratio))

Epi_dose = infusionDrug[(infusionDrug['drugname']=='Epinephrine (ml/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Epi_dose.rename(columns={'drugrate':'value'}, inplace=True)
Epi_dose['variable'] = 'Epinephrine dose'
print('Epi_dose', len(Epi_dose['patientunitstayid'].unique()), len(Epi_dose))
print('   People with Epinephrine', len(Epi_dose['patientunitstayid'].append(Epi_ratio['patientunitstayid']).unique()), len(Epi_dose['patientunitstayid'].unique())+len(Epi_ratio['patientunitstayid'].unique()))


Nor_mcgmin = infusionDrug[(infusionDrug['drugname']=='Norepinephrine (mcg/min)') & (infusionDrug['patientunitstayid'].isin(pat['patientunitstayid']))]
Nor_mcgmin = pd.merge(Nor_mcgmin[['patientunitstayid','infusionoffset','drugname','drugrate']], pat[['patientunitstayid','admissionweight']])#.describe()
Nor_mcgmin.rename(columns={'admissionweight':'weight'}, inplace=True)
Nor_list = Nor_mcgmin[Nor_mcgmin['weight']>0]

Nor_mcgmin = Nor_mcgmin[Nor_mcgmin['weight'].isna()]
Nor_mcgmin = pd.merge(Nor_mcgmin[['patientunitstayid','infusionoffset','drugname','drugrate']], pat[['patientunitstayid','dischargeweight']])#.describe()
Nor_mcgmin.rename(columns={'dischargeweight':'weight'}, inplace=True)
Nor_list = pd.concat([Nor_list, Nor_mcgmin[Nor_mcgmin['weight']>0]], ignore_index=True)

Nor_mcgmin = Nor_mcgmin[Nor_mcgmin['weight'].isna()]
Nor_mcgmin['weight'] = 85.0  # use average weight # for 5794 entries in 62 patients,1473 have drugrate 0.0
Nor_list = pd.concat([Nor_list, Nor_mcgmin[Nor_mcgmin['weight']>0]], ignore_index=True)
Nor_list['variable'] = 'Norepinephrine ratio'
Nor_list['value'] = Nor_list['drugrate']/Nor_list['weight']
Nor_list = Nor_list[['patientunitstayid','infusionoffset','variable','value']]

Nor_mcgkgmin = infusionDrug[(infusionDrug['drugname']=='Norepinephrine (mcg/kg/min)') & (infusionDrug['patientunitstayid'].isin(pat['patientunitstayid']))][['patientunitstayid','infusionoffset','drugrate']]
Nor_mcgkgmin.rename(columns={'drugrate':'value'}, inplace=True)
Nor_mcgkgmin['variable'] = 'Norepinephrine ratio'
Nor_mcgkgmin = Nor_mcgkgmin[['patientunitstayid','infusionoffset','variable','value']]

Norepi_ratio = pd.concat([Nor_list, Nor_mcgkgmin])
print('Norepi_ratio', len(Norepi_ratio['patientunitstayid'].unique()), len(Norepi_ratio))

Norepi_dose1 = infusionDrug[(infusionDrug['drugname']=='Norepinephrine ()')][['patientunitstayid','infusionoffset','drugrate']]
Norepi_dose1.rename(columns={'drugrate':'value'}, inplace=True)
Norepi_dose1['variable'] = 'Norepinephrine 1'
print('Norepi_dose1', len(Norepi_dose1['patientunitstayid'].unique()), len(Norepi_dose1))

Norepi_dose2 = infusionDrug[(infusionDrug['drugname']=='Norepinephrine (ml/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Norepi_dose2.rename(columns={'drugrate':'value'}, inplace=True)
Norepi_dose2['variable'] = 'Norepinephrine 2'
print('Norepi_dose2', len(Norepi_dose2['patientunitstayid'].unique()), len(Norepi_dose2))

Norepi_dose3 = infusionDrug[(infusionDrug['drugname']=='norepinephrine Volume (ml) (ml/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Norepi_dose3.rename(columns={'drugrate':'value'}, inplace=True)
Norepi_dose3['variable'] = 'Norepinephrine 3'
print('Norepi_dose3', len(Norepi_dose3['patientunitstayid'].unique()), len(Norepi_dose3))

all_Nores = pd.concat([Norepi_dose1,Norepi_dose2,Norepi_dose3, Nor_list])
print('   People with Norepinephrine', len(all_Nores['patientunitstayid'].unique()), len(all_Nores))

# %%
Amio_list = pd.DataFrame()
aux = infusionDrug[infusionDrug['drugname'].isin(['Amiodarone ()','Amiodarone (ml/hr)'])][['patientunitstayid','infusionoffset','drugrate']]
aux.rename(columns={'drugrate':'value'}, inplace=True)
Amio_list = pd.concat([Amio_list, aux], ignore_index=True)
aux = infusionDrug[(infusionDrug['drugname'].isin(['Amiodarone (mg/min)'])) & (infusionDrug['infusionrate']>-0.1)][['patientunitstayid','infusionoffset','infusionrate']]
aux.rename(columns={'infusionrate':'value'}, inplace=True)
Amio_list = pd.concat([Amio_list, aux], ignore_index=True)
Amio_list['variable'] = 'Amiodarone'
# Amio_list 
print('Amionide', len(Amio_list['patientunitstayid'].unique()),len(Amio_list))


Fentanyl_dose1 = infusionDrug[(infusionDrug['drugname']=='Fentanyl ()')][['patientunitstayid','infusionoffset','drugrate']]
Fentanyl_dose1.rename(columns={'drugrate':'value'}, inplace=True)
Fentanyl_dose1['variable'] = 'Fentanyl 1'
print('Fentanyl_dose1', len(Fentanyl_dose1['patientunitstayid'].unique()), len(Fentanyl_dose1))

Fentanyl_dose2 = infusionDrug[(infusionDrug['drugname']=='Fentanyl (mcg/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Fentanyl_dose2.rename(columns={'drugrate':'value'}, inplace=True)
Fentanyl_dose2['variable'] = 'Fentanyl 2'
print('Fentanyl_dose2', len(Fentanyl_dose2['patientunitstayid'].unique()), len(Fentanyl_dose2))

Fentanyl_dose3 = infusionDrug[(infusionDrug['drugname']=='Fentanyl (ml/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Fentanyl_dose3.rename(columns={'drugrate':'value'}, inplace=True)
Fentanyl_dose3['variable'] = 'Fentanyl 3'
print('Fentanyl_dose3', len(Fentanyl_dose3['patientunitstayid'].unique()), len(Fentanyl_dose3))

all_Fentanyl = pd.concat([Fentanyl_dose1,Fentanyl_dose2,Fentanyl_dose3])
print('   People with Fentanyl', len(all_Fentanyl['patientunitstayid'].unique()), len(all_Fentanyl))


Furosemide_list = pd.DataFrame()
Furosemide_list = infusionDrug[infusionDrug['drugname'].isin(['Furosemide (mg/hr)','Furosemide (ml/hr)'])][['patientunitstayid','infusionoffset','drugrate']]
Furosemide_list.rename(columns={'drugrate':'value'}, inplace=True)
Furosemide_list['variable'] = 'Furosemide'
print('Furosemide', len(Furosemide_list['patientunitstayid'].unique()),len(Furosemide_list))


Heparin_dose1 = infusionDrug[(infusionDrug['drugname']=='Heparin ()')][['patientunitstayid','infusionoffset','drugrate']]
Heparin_dose1.rename(columns={'drugrate':'value'}, inplace=True)
Heparin_dose1['variable'] = 'Heparin 1'
print('Heparin_dose1', len(Heparin_dose1['patientunitstayid'].unique()), len(Heparin_dose1))

Heparin_dose2 = infusionDrug[(infusionDrug['drugname']=='Heparin (ml/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Heparin_dose2.rename(columns={'drugrate':'value'}, inplace=True)
Heparin_dose2['variable'] = 'Heparin 2'
print('Heparin_dose2', len(Heparin_dose2['patientunitstayid'].unique()), len(Heparin_dose2))

Heparin_dose3 = infusionDrug[(infusionDrug['drugname']=='Heparin (units/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Heparin_dose3.rename(columns={'drugrate':'value'}, inplace=True)
Heparin_dose3['variable'] = 'Heparin 3'
print('Heparin_dose3', len(Heparin_dose3['patientunitstayid'].unique()), len(Heparin_dose3))

vol_hep = ['Volume (ml) Heparin-heparin 25,000 units in dextrose 500 mL infusion (ml/hr)',
    'Volume (ml) Heparin-heparin 25,000 units in 0.45 % sodium chloride 500 mL infusion (ml/hr)']
Heparin_dose4 = infusionDrug[(infusionDrug['drugname'].isin(vol_hep))][['patientunitstayid','infusionoffset','drugrate']]
Heparin_dose4.rename(columns={'drugrate':'value'}, inplace=True)
Heparin_dose4['variable'] = 'Heparin vol'
print('Heparin_dose4', len(Heparin_dose4['patientunitstayid'].unique()), len(Heparin_dose4))

all_Heparin = pd.concat([Heparin_dose1,Heparin_dose2, Heparin_dose3, Heparin_dose4])
print('   People with Heparin', len(all_Heparin['patientunitstayid'].unique()), len(all_Heparin))


Insulin_dose1 = infusionDrug[(infusionDrug['drugname']=='Insulin ()')][['patientunitstayid','infusionoffset','drugrate']]
Insulin_dose1.rename(columns={'drugrate':'value'}, inplace=True)
Insulin_dose1['variable'] = 'Insulin 1'
print('Insulin_dose1', len(Insulin_dose1['patientunitstayid'].unique()), len(Insulin_dose1))

Insulin_dose2 = infusionDrug[(infusionDrug['drugname']=='Insulin (ml/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Insulin_dose2.rename(columns={'drugrate':'value'}, inplace=True)
Insulin_dose2['variable'] = 'Insulin 2'
print('Insulin_dose2', len(Insulin_dose2['patientunitstayid'].unique()), len(Insulin_dose2))

Insulin_dose3 = infusionDrug[(infusionDrug['drugname']=='Insulin (units/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Insulin_dose3.rename(columns={'drugrate':'value'}, inplace=True)
Insulin_dose3['variable'] = 'Insulin 3'
print('Insulin_dose3', len(Insulin_dose3['patientunitstayid'].unique()), len(Insulin_dose3))

Insulin_dose4 = infusionDrug[(infusionDrug['drugname']=='Insulin 250 Units Sodium Chloride 0.9% 250 ml (units/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Insulin_dose4.rename(columns={'drugrate':'value'}, inplace=True)
Insulin_dose4['variable'] = 'Insulin 4'
print('Insulin_dose3', len(Insulin_dose4['patientunitstayid'].unique()), len(Insulin_dose4))

all_Insulin = pd.concat([Insulin_dose1,Insulin_dose2, Insulin_dose3, Insulin_dose4])
print('   People with Insulin', len(all_Insulin['patientunitstayid'].unique()), len(all_Insulin))


Midazolam_dose1 = infusionDrug[(infusionDrug['drugname']=='Midazolam (mg/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Midazolam_dose1.rename(columns={'drugrate':'value'}, inplace=True)
Midazolam_dose1['variable'] = 'Midazolam 1'
print('Midazolam_dose1', len(Midazolam_dose1['patientunitstayid'].unique()), len(Midazolam_dose1))

Midazolam_dose2 = infusionDrug[(infusionDrug['drugname']=='Midazolam (ml/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Midazolam_dose2.rename(columns={'drugrate':'value'}, inplace=True)
Midazolam_dose2['variable'] = 'Midazolam 2'
print('Midazolam_dose2', len(Midazolam_dose2['patientunitstayid'].unique()), len(Midazolam_dose2))

all_Midazolam = pd.concat([Midazolam_dose1,Midazolam_dose2])
print('   People with Midazolam', len(all_Midazolam['patientunitstayid'].unique()), len(all_Midazolam))

Milrinone_dose1 = infusionDrug[(infusionDrug['drugname']=='Milrinone (mcg/kg/min)')][['patientunitstayid','infusionoffset','drugrate']]
Milrinone_dose1.rename(columns={'drugrate':'value'}, inplace=True)
Milrinone_dose1['variable'] = 'Milrinone 1'
print('Milrinone_dose1', len(Milrinone_dose1['patientunitstayid'].unique()), len(Milrinone_dose1))

Milrinone_dose2 = infusionDrug[(infusionDrug['drugname']=='Milrinone (ml/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Milrinone_dose2.rename(columns={'drugrate':'value'}, inplace=True)
Milrinone_dose2['variable'] = 'Milrinone 2'
print('Milrinone_dose2', len(Milrinone_dose2['patientunitstayid'].unique()), len(Milrinone_dose2))

all_Milrinone = pd.concat([Milrinone_dose1,Milrinone_dose2])
print('   People with Milrinone', len(all_Milrinone['patientunitstayid'].unique()), len(all_Milrinone))

Nitrogly_list = pd.DataFrame()
Nitrogly_list = infusionDrug[infusionDrug['drugname'].isin(['NitroGLYCERIN IVF Infused (ml/hr)','Nitroglycerin (ml/hr)'])][['patientunitstayid','infusionoffset','drugrate']]
Nitrogly_list.rename(columns={'drugrate':'value'}, inplace=True)
Nitrogly_list['variable'] = 'Nitroglycerin 1'
print('Nitrogly', len(Nitrogly_list['patientunitstayid'].unique()),len(Nitrogly_list))

Nitrogly_list2 = pd.DataFrame()
Nitrogly_list2 = infusionDrug[infusionDrug['drugname']=='Nitroglycerin (mcg/min)'][['patientunitstayid','infusionoffset','drugrate']]
Nitrogly_list2.rename(columns={'drugrate':'value'}, inplace=True)
Nitrogly_list2['variable'] = 'Nitroglycerin 2'
print('Nitrogly list 2', len(Nitrogly_list2['patientunitstayid'].unique()),len(Nitrogly_list2))

all_Nitrogly = pd.concat([Nitrogly_list,Nitrogly_list2])
print('   People with Nitrogly', len(all_Nitrogly['patientunitstayid'].unique()), len(all_Nitrogly))


Nitroprus = infusionDrug[(infusionDrug['drugname']=='Nitroprusside (ml/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Nitroprus.rename(columns={'drugrate':'value'}, inplace=True)
Nitroprus['variable'] = 'Nitroprusside'
# Nitroprus 
print('Nitroprusside', len(Nitroprus['patientunitstayid'].unique()), len(Nitroprus))


Pantoprazole_list = pd.DataFrame()
aux = infusionDrug[(infusionDrug['drugname'].isin(['Pantoprazole (mg/hr)'])) & (infusionDrug['infusionrate']>-0.1)][['patientunitstayid','infusionoffset','infusionrate']]
aux.rename(columns={'infusionrate':'value'}, inplace=True)
Pantoprazole_list = pd.concat([Pantoprazole_list, aux], ignore_index=True)
aux = infusionDrug[infusionDrug['drugname']=='Pantoprazole (ml/hr)'][['patientunitstayid','infusionoffset','drugrate']]
aux.rename(columns={'drugrate':'value'}, inplace=True)
Pantoprazole_list = pd.concat([Pantoprazole_list, aux], ignore_index=True)
Pantoprazole_list['variable'] = 'Pantoprazole'
# Pantoprazole_list 
print('Pantoprazole', len(Pantoprazole_list['patientunitstayid'].unique()),len(Pantoprazole_list))

Propofol_dose1 = infusionDrug[(infusionDrug['drugname']=='Propofol ()')][['patientunitstayid','infusionoffset','drugrate']]
Propofol_dose1.rename(columns={'drugrate':'value'}, inplace=True)
Propofol_dose1['variable'] = 'Propofol 1'
print('Propofol_dose1', len(Propofol_dose1['patientunitstayid'].unique()), len(Propofol_dose1))

Propofol_dose2 = infusionDrug[(infusionDrug['drugname']=='Propofol (mcg/kg/min)')][['patientunitstayid','infusionoffset','drugrate']]
Propofol_dose2.rename(columns={'drugrate':'value'}, inplace=True)
Propofol_dose2['variable'] = 'Propofol 2'
print('Propofol_dose2', len(Propofol_dose2['patientunitstayid'].unique()), len(Propofol_dose2))

Propofol_dose3 = infusionDrug[(infusionDrug['drugname']=='Propofol (ml/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Propofol_dose3.rename(columns={'drugrate':'value'}, inplace=True)
Propofol_dose3['variable'] = 'Propofol 3'
print('Propofol_dose3', len(Propofol_dose3['patientunitstayid'].unique()), len(Propofol_dose3))

all_Propofol = pd.concat([Propofol_dose1,Propofol_dose2,Propofol_dose3])
print('   People with Propofol', len(all_Propofol['patientunitstayid'].unique()), len(all_Propofol))


Vasopressin_dose1 = infusionDrug[(infusionDrug['drugname']=='Vasopressin ()')][['patientunitstayid','infusionoffset','drugrate']]
Vasopressin_dose1.rename(columns={'drugrate':'value'}, inplace=True)
Vasopressin_dose1['variable'] = 'Vasopressin 1'
print('Vasopressin_dose1', len(Vasopressin_dose1['patientunitstayid'].unique()), len(Vasopressin_dose1))

Vasopressin_dose2 = infusionDrug[(infusionDrug['drugname']=='Vasopressin (ml/hr)')][['patientunitstayid','infusionoffset','drugrate']]
Vasopressin_dose2.rename(columns={'drugrate':'value'}, inplace=True)
Vasopressin_dose2['variable'] = 'Vasopressin 2'
print('Vasopressin_dose2', len(Vasopressin_dose2['patientunitstayid'].unique()), len(Vasopressin_dose2))

Vasopressin_dose3 = infusionDrug[(infusionDrug['drugname']=='Vasopressin (units/min)')][['patientunitstayid','infusionoffset','drugrate']]
Vasopressin_dose3.rename(columns={'drugrate':'value'}, inplace=True)
Vasopressin_dose3['variable'] = 'Vasopressin 3'
print('Vasopressin_dose3', len(Vasopressin_dose3['patientunitstayid'].unique()), len(Vasopressin_dose3))

all_Vasopressin = pd.concat([Vasopressin_dose1,Vasopressin_dose2,Vasopressin_dose3])
print('   People with Vasopressin', len(all_Vasopressin['patientunitstayid'].unique()), len(all_Vasopressin))


# %%


infD_list = pd.concat([Dobu_ratio, Dobu_dose, Dopa_ratio, Dopa_dose, 
    Epi_ratio, Epi_dose, Norepi_ratio, Norepi_dose1, Norepi_dose2, Norepi_dose3, 
    Amio_list, Fentanyl_dose1, Fentanyl_dose2, Fentanyl_dose3, Furosemide_list, 
    Heparin_dose1, Heparin_dose2, Heparin_dose3, Heparin_dose4, 
    Insulin_dose1, Insulin_dose2, Insulin_dose3, Insulin_dose4,
    Midazolam_dose1, Midazolam_dose2, Milrinone_dose1, Milrinone_dose2,
    Nitrogly_list, Nitrogly_list2, Nitroprus, Pantoprazole_list, 
    Propofol_dose1, Propofol_dose2, Propofol_dose3,
    Vasopressin_dose1, Vasopressin_dose2, Vasopressin_dose3])
                       
infD_list.rename(columns={'patientunitstayid':'stayid','infusionoffset':'time','variable':'var'},inplace=True)
infD_list = infD_list[['stayid','time','var','value']]
    
numeric_mask = pd.to_numeric(infD_list['value'], errors='coerce').notna()
infD_list = infD_list[numeric_mask]

infD_list['var'] = "d " + infD_list['var'].astype(str)

infD_list.to_csv('infD_list.csv', index=False)
    
log_str =  'infusionDrug &                             & ' + str(len(infD_list['stayid'].unique())) + ' & ' + str(len(infD_list)) + '//'
with open('log_vars.csv', 'a') as file:
    print(log_str, file=file)
print('infD read with', len(infD_list), 'entries.')
infD_list.groupby('var').count()
# 0.5 min, 84 MB, 2.352.524 entries

# %%
"""
9, patient
'gender', 'age','admissionheight','hospitaladmittime24', 'admissionweight', 
'hospitaldischargeyear', 'unitvisitnumber','patienthealthsystemstayid', 'hospitaladmitoffset'
time is always 0, value sometimes either 0 or 1. 
only one value per patient
774.199 data points and 17 vars (8 ICU types)
"""

pat_list = pd.DataFrame()
for key in ['gender', 'age','admissionheight','hospitaladmittime24', 'admissionweight', 'hospitaldischargeyear', 
            'unitvisitnumber','patienthealthsystemstayid', 'hospitaladmitoffset']:
    aux = pat[['patientunitstayid', key]].copy()
    if key == 'gender':
        aux['gender'] = np.where(pat['gender'] == 'Male', 1, 0)
    aux['var'] = "s " + key.capitalize()
    aux.rename(columns={key: 'value','patientunitstayid':'stayid'}, inplace=True)
    if key == 'hospitaladmittime24':
        aux['value'] = aux['value'].apply(lambda x: (int(x[0:2]) * 60 +  int(x[3:5]))/60)
    pat_list = pd.concat([pat_list, aux], ignore_index=True)
pat_list['time'] = 0

pat_list = pat_list[['stayid','time','var','value']]

list_unittype = pat[['patientunitstayid','unittype']].copy()
list_unittype[['value', 'time']] = 1, 0
list_unittype.rename(columns={'patientunitstayid':'stayid', 'unittype':'var'}, inplace=True)
list_unittype['var'] = "i " + list_unittype['var'].astype(str)

pat_list = pd.concat([pat_list, list_unittype])
numeric_mask = pd.to_numeric(pat_list['value'], errors='coerce').notna()
# print(pat_list[~numeric_mask].groupby('var').count())
pat_list = pat_list[numeric_mask]

pat_list.to_csv('pat_list.csv', index=False)
log_str =  'pat&                             & ' + str(len(pat_list['stayid'].unique())) + ' & ' + str(len(pat_list)) + '//'
with open('log_vars.csv', 'a') as file:
    print(log_str, file=file)
print('pat read with', len(pat_list), 'entries and', len(pat_list['var'].unique()), 'vars (8 ICU types).')  # 621632 (unfiltered)
pat_list.groupby('var').count().reset_index()
#2 sec, 18 MB, 618.791 entries

# %% [markdown]
# # Save datapoints to joblib

# %% [markdown]
# Following steps (25min):
# - load data (3.5min)
# - reset patient index
# - build thirds of each data set (otherwise crash)
# - sort values (2.5min)
# - remove duplicates (3min) [if data point is copied in vitalA and vitalP]
# - build vind (variable index) and apply
# - create train, valid and test indices
# - recreate time 
# - average if mulitple values per ['id','time','var'] (6min)
# - write to eicu_preprocessed.pkl (2min)
# - load eicu_preprocessed.pkl (2min)
# - calc mean and std and normalize (3min)
# - write to final eicu_preprocessed_n.pkl file (3min)
# - show variable names and frequency

# %%
# load data

vitalP_list = pd.read_csv('vitalP_list.csv')
print('vitalP read')
vitalA_list = pd.read_csv('vitalA_list.csv')
print('vitalA read')
lab_list = pd.read_csv('lab_list.csv')
rC_list = pd.read_csv('respCh_list.csv')
inOu_list = pd.read_csv('inOu_list.csv')
nursCh_list = pd.read_csv('nursCh_list.csv')
print('lab, rC, inOu, nursCh read')
pE_list = pd.read_csv('physEx_list.csv')
infD_list = pd.read_csv('infD_list.csv')
pat_list = pd.read_csv('pat_list.csv')
print('physEx, infD, pat read -- all read')


# %%
# create patient index (from 0 to 77704)
pat_index = pat_list['stayid'].sort_values().unique()
pat_index = pd.DataFrame(pat_index).reset_index()
pat_index.rename(columns={'index':'id', 0:'stayid'}, inplace=True)
pat_index.to_csv('pat_index.csv', index=False)

# %%
# bring data to three subsets, otherwise kernel crashes
one_3 = pat_index['stayid'].loc[len(pat_index)//3]
two_3 = pat_index['stayid'].loc[(2*len(pat_index))//3]

vitalP_list1 = vitalP_list[vitalP_list['stayid']<one_3]
vitalP_list2 = vitalP_list[(vitalP_list['stayid']>=one_3) & (vitalP_list['stayid']<two_3)]
vitalP_list3 = vitalP_list[vitalP_list['stayid']>=two_3]
vitalA_list1 = vitalA_list[vitalA_list['stayid']<one_3]
vitalA_list2 = vitalA_list[(vitalA_list['stayid']>=one_3) & (vitalA_list['stayid']<two_3)]
vitalA_list3 = vitalA_list[vitalA_list['stayid']>=two_3]
lab_list1 = lab_list[lab_list['stayid']<one_3]
lab_list2 = lab_list[(lab_list['stayid']>=one_3) & (lab_list['stayid']<two_3)]
lab_list3 = lab_list[lab_list['stayid']>=two_3]
rC_list1 = rC_list[rC_list['stayid']<one_3]
rC_list2 = rC_list[(rC_list['stayid']>=one_3) & (rC_list['stayid']<two_3)]
rC_list3 = rC_list[rC_list['stayid']>=two_3]
inOu_list1 = inOu_list[inOu_list['stayid']<one_3]
inOu_list2 = inOu_list[(inOu_list['stayid']>=one_3) & (inOu_list['stayid']<two_3)]
inOu_list3 = inOu_list[inOu_list['stayid']>=two_3]
nursCh_list1 = nursCh_list[nursCh_list['stayid']<one_3]
nursCh_list2 = nursCh_list[(nursCh_list['stayid']>=one_3) & (nursCh_list['stayid']<two_3)]
nursCh_list3 = nursCh_list[nursCh_list['stayid']>=two_3]
pE_list1 = pE_list[pE_list['stayid']<one_3]
pE_list2 = pE_list[(pE_list['stayid']>=one_3) & (pE_list['stayid']<two_3)]
pE_list3 = pE_list[pE_list['stayid']>=two_3]
infD_list1 = infD_list[infD_list['stayid']<one_3]
infD_list2 = infD_list[(infD_list['stayid']>=one_3) & (infD_list['stayid']<two_3)]
infD_list3 = infD_list[infD_list['stayid']>=two_3]
pat_list1 = pat_list[pat_list['stayid']<one_3]
pat_list2 = pat_list[(pat_list['stayid']>=one_3) & (pat_list['stayid']<two_3)]
pat_list3 = pat_list[pat_list['stayid']>=two_3]


# %%
# concatenate data to one dataframe
all_data_1 = pd.concat([vitalP_list1,vitalA_list1,lab_list1,inOu_list1, nursCh_list1, infD_list1, pat_list1, rC_list1, pE_list1])
all_data_2 = pd.concat([vitalP_list2,vitalA_list2,lab_list2,inOu_list2, nursCh_list2, infD_list2, pat_list2, rC_list2, pE_list2])
all_data_3 = pd.concat([vitalP_list3,vitalA_list3,lab_list3,inOu_list3, nursCh_list3, infD_list3, pat_list3, rC_list3, pE_list3])

all_data_1['id'] = all_data_1['stayid'].map(pat_index.set_index('stayid')['id'])
all_data_2['id'] = all_data_2['stayid'].map(pat_index.set_index('stayid')['id'])
all_data_3['id'] = all_data_3['stayid'].map(pat_index.set_index('stayid')['id'])


# %%
all_data_1 = all_data_1[['id','time','var','value']]
all_data_1.sort_values(by=['id', 'time'], inplace=True)
print('all_data points 1', len(all_data_1))
all_data_2 = all_data_2[['id','time','var','value']]
all_data_2.sort_values(by=['id', 'time'], inplace=True)
print('all_data points 2', len(all_data_2))
all_data_3 = all_data_3[['id','time','var','value']]
all_data_3.sort_values(by=['id', 'time'], inplace=True)
print('all_data points 3', len(all_data_3))

# 2.5 min

# %%
all_data_no_duplicates_1 = all_data_1.drop_duplicates()  # 141061593/146413757
print('all data points 1 (without duplicates)', len(all_data_no_duplicates_1))
all_data_no_duplicates_2 = all_data_2.drop_duplicates()  # 131291032/136477010
print('all data points 2 (without duplicates)', len(all_data_no_duplicates_2))
all_data_no_duplicates_3 = all_data_3.drop_duplicates()  # 145305237/150750457
print('all data points 3 (without duplicates)', len(all_data_no_duplicates_3))
# 3 min

# %%

S = len(pat_index)
bp1, bp2 = int(0.64*S), int(0.8*S)
train_ind = pat_index['id'][:bp1]
valid_ind = pat_index['id'][bp1:bp2]
test_ind = pat_index['id'][bp2:]

# %%
all_data_1['time'] = all_data_1['time']/60
all_data_2['time'] = all_data_2['time']/60
all_data_3['time'] = all_data_3['time']/60

# %%
print(len(all_data_1))
all_data_1 = all_data_1.groupby(['id', 'time', 'var']).agg({'value': 'mean'}).reset_index()
print('-->', len(all_data_1))
print(len(all_data_2))
all_data_2 = all_data_2.groupby(['id', 'time', 'var']).agg({'value': 'mean'}).reset_index()
print('-->', len(all_data_2))
print(len(all_data_3))
all_data_3 = all_data_3.groupby(['id', 'time', 'var']).agg({'value': 'mean'}).reset_index()
print('-->', len(all_data_3))
# 6 min (average if multiple values)
# here are no duplicates anymore. But still ~21 mio data points with different values for id time var


# %%
ts = pd.concat([all_data_1, all_data_2, all_data_3], ignore_index=True)
joblib.dump([ts, train_ind, valid_ind, test_ind], 'eicu_preprocessed.pkl')

# 2 min, write to file first

# %%
ts, train_ind, valid_ind, test_ind = joblib.load('eicu_preprocessed.pkl')

# 2 min, read again

# %%
means_stds = ts.groupby('var').agg({'value': ['mean', 'std']})
means_stds.columns = [col[1] for col in means_stds.columns]
means_stds.loc[means_stds['std'] == 0, 'std'] = 1
ts = ts.merge(means_stds.reset_index(), on='var', how='left')
ts['value'] = (ts['value']-ts['mean'])/ts['std']
ts.rename(columns={'id':'ts_ind', 'var':'variable', 'time':'hour'}, inplace=True)  # value bleibt gleich 

# 2 min, build mean and std

# %%
joblib.dump([ts, train_ind, valid_ind, test_ind], 'eicu_preprocessed_n.pkl')

# 3 min, write to file. 

# %%
var_list = ts[['variable','mean','std']].drop_duplicates()
var_list.sort_values('mean')

# %%
pd.set_option('display.max_rows', 120)
ts, train_ind, valid_ind, test_ind = joblib.load('eicu_preprocessed_n.pkl')
var_num = ts.groupby('variable').count().reset_index()
var_pat = ts.groupby(['ts_ind','variable']).count().groupby('variable').count().reset_index()
var_num = pd.merge(var_pat[['variable','hour']], var_num[['variable','hour']], on='variable', how='inner')
var_num.rename(columns={'hour_x': 'number patients', 'hour_y': 'number'}, inplace=True)
var_num



def inv_list(l, start=0):  # Create vind
    d = {}
    for i in range(len(l)):
        d[l[i]] = i+start
    return d


def f(x):
    mask   = [0 for i in range(V)]
    values = [0 for i in range(V)]
    for vv in x:  # tuple of ['vind','value']
        v = int(vv[0])-1  # shift index of vind
        mask[v] = 1
        values[v] = vv[1]  # get value
    return values+mask  # concat


def pad(x):
    if len(x) > 3*880:
        print('bigger than 880*3', len(x))
    return x+[0]*(fore_max_len-len(x))


## for train and val set:
# Read data.
data_path = 'eicu_preprocessed_n.pkl'
data, train_ind, valid_ind, test_ind = joblib.load(open(data_path, 'rb'))
# Remove test patients.
test_sub = data.loc[data.ts_ind.isin(test_ind)].ts_ind.unique()
data = data.loc[~data.ts_ind.isin(test_sub)]

# Get static data with mean fill and missingness indicator.
static_varis = ['i CCU-CTICU', 'i CSICU', 'i CTICU', 'i Cardiac ICU', 'i MICU', 'i Med-Surg ICU', 'i Neuro ICU', 'i SICU', 
                's Admissionheight', 's Admissionweight', 's Age', 's Gender', 's Hospitaladmitoffset', 's Hospitaladmittime24', 
                's Hospitaldischargeyear', 's Patienthealthsystemstayid', 's Unitvisitnumber']  # 17
ii = data.variable.isin(static_varis)
static_data = data.loc[ii]
data = data.loc[~ii]  # ~ binary flip
# print('data\n',data)

static_var_to_ind = inv_list(static_varis)
D = len(static_varis)  # 17 demographic variables
N = data.ts_ind.max()+1  # 77.704 number of stays
demo = np.zeros((int(N), int(D)))
for row in tqdm(static_data.itertuples()):
    demo[int(row.ts_ind), static_var_to_ind[row.variable]] = row.value
# Normalize static data.
means = demo.mean(axis=0, keepdims=True)  # quite sparse
stds = demo.std(axis=0, keepdims=True)
stds = (stds == 0)*1 + (stds != 0)*stds
demo = (demo-means)/stds
# Get variable indices.
varis = sorted(list(set(data.variable)))
V = len(varis)
print('varis', varis, V)
var_to_ind = inv_list(varis, start=1)
data['vind'] = data.variable.map(var_to_ind)
data = data[['ts_ind', 'vind', 'hour', 'value']].sort_values(by=['ts_ind', 'vind', 'hour'])
# Find max_len.
fore_max_len = 2640  # hard coded max_len of vars
# Get forecast inputs and outputs.
fore_times_ip, fore_values_ip, fore_varis_ip = [], [], []
fore_op, fore_op_awesome, fore_inds = [], [], []
for w in tqdm(range(25, 124, 4)):
    pred_data = data.loc[(data.hour>=w)&(data.hour<=w+24)]
    pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value':'first'}).reset_index()
    pred_data['vind_value'] = pred_data[['vind', 'value']].values.tolist()
    pred_data = pred_data.groupby('ts_ind').agg({'vind_value':list}).reset_index()
    pred_data['vind_value'] = pred_data['vind_value'].apply(f)   

    obs_data = data.loc[(data.hour < w) & (data.hour >= w-24)]
    obs_data = obs_data.loc[obs_data.ts_ind.isin(pred_data.ts_ind)]
    obs_data = obs_data.groupby('ts_ind').head(fore_max_len)
    obs_data = obs_data.groupby('ts_ind').agg({'vind': list, 'hour': list, 'value': list}).reset_index()
    for pred_window  in range(-24, 24, 1):
        pred_data = data.loc[(data.hour >= w+pred_window) & (data.hour <= w+1 +pred_window)]
        pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value': 'first'}).reset_index()
        pred_data['vind_value'+str(pred_window)] = pred_data[['vind', 'value']].values.tolist()
        pred_data = pred_data.groupby('ts_ind').agg({'vind_value'+str(pred_window): list}).reset_index()
        pred_data['vind_value'+str(pred_window)] = pred_data['vind_value'+str(pred_window)].apply(f)  # 721 entries with 2*129 vind_values
        obs_data = obs_data.merge(pred_data, on='ts_ind')

    for col in ['vind', 'hour', 'value']:
        obs_data[col] = obs_data[col].apply(pad)
    fore_op_awesome.append(np.array(list([list(obs_data['vind_value'+str(pred_window)]) for pred_window in range(-24, 24, 1)])))
    fore_inds.append(np.array([int(x) for x in list(obs_data.ts_ind)]))
    fore_times_ip.append(np.array(list(obs_data.hour)))
    fore_values_ip.append(np.array(list(obs_data.value)))
    fore_varis_ip.append(np.array(list(obs_data.vind)))
    
del data
fore_times_ip = np.concatenate(fore_times_ip, axis=0)
fore_values_ip = np.concatenate(fore_values_ip, axis=0)
fore_varis_ip = np.concatenate(fore_varis_ip, axis=0)

fore_op_awesome = np.concatenate(fore_op_awesome, axis=1)
fore_op = np.swapaxes(fore_op_awesome, 0, 1)
print(fore_op.shape)

fore_inds = np.concatenate(fore_inds, axis=0)
fore_demo = demo[fore_inds]
# Generate sets of inputs and outputs.
train_ind = np.argwhere(np.in1d(fore_inds, train_ind)).flatten()
valid_ind = np.argwhere(np.in1d(fore_inds, valid_ind)).flatten()
fore_train_ip = [ip[train_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
fore_valid_ip = [ip[valid_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
del fore_times_ip, fore_values_ip, fore_varis_ip, demo, fore_demo
fore_train_op = fore_op[train_ind]
fore_valid_op = fore_op[valid_ind]
del fore_op

joblib.dump(fore_train_op, 'fore_train_op_dense.pkl')
joblib.dump(fore_valid_op, 'fore_valid_op_dense.pkl')
joblib.dump(fore_train_ip, 'fore_train_ip_dense.pkl')
joblib.dump(fore_valid_ip, 'fore_valid_ip_dense.pkl')

# print('lengths of rem_sub, fore_train_ip[1], fore_valid_ip[0]')
# print(len(rem_sub), fore_train_ip[1].shape, fore_valid_ip[0].shape)
# 214 min



## for test sets
# Read data.
data_path = 'eicu_preprocessed_n.pkl'
data, train_ind, valid_ind, test_ind = joblib.load(open(data_path, 'rb'))

# Only test patients
data = data.loc[data.ts_ind.isin(test_ind)]
data = data.loc[(data.hour>=0) & (data.hour<=48)]

means_stds = data.groupby("variable").agg({"mean":"first", "std":"first"})
mean_std_dict = dict()
for pos, row in means_stds.iterrows():
    mean_std_dict[pos] = (float(row["mean"]), float(row["std"]))
joblib.dump(mean_std_dict, 'mean_std_dict.pkl')

# Get static data with mean fill and missingness indicator.
static_varis = ['i CCU-CTICU', 'i CSICU', 'i CTICU', 'i Cardiac ICU', 'i MICU', 'i Med-Surg ICU', 'i Neuro ICU', 'i SICU', 
                's Admissionheight', 's Admissionweight', 's Age', 's Gender', 's Hospitaladmitoffset', 's Hospitaladmittime24', 
                's Hospitaldischargeyear', 's Patienthealthsystemstayid', 's Unitvisitnumber']  # 17
ii = data.variable.isin(static_varis)
static_data = data.loc[ii]
data = data.loc[~ii]  # ~ binary flip

static_var_to_ind = inv_list(static_varis)
D = len(static_varis)  # 17 demographic variables
N = data.ts_ind.max()+1  # 77.704 number of stays
demo = np.zeros((int(N), int(D)))
for row in tqdm(static_data.itertuples()):
    demo[int(row.ts_ind), static_var_to_ind[row.variable]] = row.value

# Normalize static data.
means = demo.mean(axis=0, keepdims=True)  # quite sparse
stds = demo.std(axis=0, keepdims=True)
stds = (stds == 0)*1 + (stds != 0)*stds
demo = (demo-means)/stds

# Get variable indices.
varis = sorted(list(set(data.variable)))
V = len(varis)
print('V', V, 'varis', varis)
var_to_ind = inv_list(varis, start=1)
data['vind'] = data.variable.map(var_to_ind)
data = data[['ts_ind', 'vind', 'hour', 'value']].sort_values(by=['ts_ind', 'vind', 'hour'])
# Find max_len.
fore_max_len = 880*3  # hard coded max_len of vars
# Get forecast inputs and outputs.
fore_times_ip, fore_values_ip, fore_varis_ip = [], [], []
fore_op, fore_op_awesome, fore_inds = [], [], []
pred_data = data.loc[(data.hour>=24)&(data.hour<=48)]
pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value':'first'}).reset_index()
pred_data['vind_value'] = pred_data[['vind', 'value']].values.tolist()
pred_data = pred_data.groupby('ts_ind').agg({'vind_value':list}).reset_index()
pred_data['vind_value'] = pred_data['vind_value'].apply(f)   

obs_data = data.loc[(data.hour < 24) & (data.hour >= 0)]
obs_data = obs_data.loc[obs_data.ts_ind.isin(pred_data.ts_ind)]
obs_data = obs_data.groupby('ts_ind').head(fore_max_len)
obs_data = obs_data.groupby('ts_ind').agg({'vind': list, 'hour': list, 'value': list}).reset_index()
# Take 24 hours before and after a fixed timepoint (w=24 hours)
for pred_window  in range(-24, 24, 1):
    pred_data = data.loc[(data.hour >= 24+pred_window) & (data.hour <= 25 + pred_window)]
    pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value': 'first'}).reset_index()
    pred_data['vind_value'+str(pred_window)] = pred_data[['vind', 'value']].values.tolist()
    pred_data = pred_data.groupby('ts_ind').agg({'vind_value'+str(pred_window): list}).reset_index()
    pred_data['vind_value'+str(pred_window)] = pred_data['vind_value'+str(pred_window)].apply(f)  # 721 entries with 2*129 vind_values
    obs_data = obs_data.merge(pred_data, on='ts_ind')

for col in ['vind', 'hour', 'value']:
    obs_data[col] = obs_data[col].apply(pad)
fore_op_awesome.append(np.array(list([list(obs_data['vind_value'+str(pred_window)]) for pred_window in range(-24, 24, 1)])))
fore_inds.append(np.array([int(x) for x in list(obs_data.ts_ind)]))
fore_times_ip.append(np.array(list(obs_data.hour)))
fore_values_ip.append(np.array(list(obs_data.value)))
fore_varis_ip.append(np.array(list(obs_data.vind)))
    
del data
fore_times_ip = np.concatenate(fore_times_ip, axis=0)
fore_values_ip = np.concatenate(fore_values_ip, axis=0)
fore_varis_ip = np.concatenate(fore_varis_ip, axis=0)

fore_op_awesome = np.concatenate(fore_op_awesome, axis=1)
fore_op = np.swapaxes(fore_op_awesome, 0, 1)
print(fore_op.shape)

fore_inds = np.concatenate(fore_inds, axis=0)
fore_demo = demo[fore_inds]

# Generate 3 sets of inputs and outputs.
test_ind = np.argwhere(np.in1d(fore_inds, test_ind)).flatten()
fore_test_ip = [ip[test_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
del fore_times_ip, fore_values_ip, fore_varis_ip, demo, fore_demo
fore_test_op = fore_op[test_ind]
del fore_op

joblib.dump(fore_test_op, 'fore_test_op_dense.pkl')
joblib.dump(fore_test_ip, 'fore_test_ip_dense.pkl')
