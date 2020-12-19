import pandas as pd
import numpy as np

def data_merge(df, ext_df, suffix):
    ext_df = ext_df.copy()
    ext_df.columns = [f'{c}{suffix}'  if c != 'id' else c for c in ext_df.columns]
    return df.merge(ext_df, how='left', on='id')
    
def merge_all(train, test, em, ws, ed):
    train = data_merge(train, em, '_em')
    train = data_merge(train, ws, '_ws')
    train = data_merge(train, ed, '_ed')
    
    test = data_merge(test, em, '_em')
    test = data_merge(test, ws, '_ws')
    test = data_merge(test, ed, '_ed')
    
    train['locality'] = train['locality'].astype(str)
    test['locality'] = test['locality'].astype(str)
    
    train['modification_date'] = train['modification_date'].apply(convert_to_date)
    train['finish_date_em'] = train['finish_date_em'].apply(convert_to_date)
    train['start_date_em'] = train['start_date_em'].apply(convert_to_date)
    
    train['we_age_mnth'] = train[['start_date_em', 'finish_date_em', 'modification_date']].apply(calc_we, axis=1)
    
    train['we_age_mnth'] = train[['we_age_mnth', 'age']].apply(adj_we_ag, axis = 1)
    
    test['modification_date'] = test['modification_date'].apply(convert_to_date)
    test['finish_date_em'] = test['finish_date_em'].apply(convert_to_date)
    test['start_date_em'] = test['start_date_em'].apply(convert_to_date)
    
    test['we_age_mnth'] = test[['start_date_em', 'finish_date_em', 'modification_date']].apply(calc_we, axis=1)
    
    test['we_age_mnth'] = test[['we_age_mnth', 'age']].apply(adj_we_ag, axis = 1)
    
    train['unemp'] = ~train.finish_date_em.isna()
    test['unemp'] = ~test.finish_date_em.isna()

    return train, test


def set_age(train, test):
    min_date = pd.to_datetime(train['publish_date'], format="%Y-%m-%d").min()
    train['pub_age_mnth'] = (pd.to_datetime(train['publish_date'], format="%Y-%m-%d") - min_date)//np.timedelta64(1, 'M')
    test['pub_age_mnth'] = (pd.to_datetime(test['publish_date'], format="%Y-%m-%d") - min_date)//np.timedelta64(1, 'M')
    
    return train, test

def set_age2(train, test):
    min_date = pd.to_datetime(train['modification_date'], format="%Y-%m-%d").min()
    train['mod_age_mnth'] = (pd.to_datetime(train['modification_date'], format="%Y-%m-%d") - min_date)//np.timedelta64(1, 'M')
    test['mod_age_mnth'] = (pd.to_datetime(test['modification_date'], format="%Y-%m-%d") - min_date)//np.timedelta64(1, 'M')
    
    return train, test

# position - слишком вариативна - можно кластеризировать\
CAT_FEATURES = ['position', 
                'region', 
                'industry', 
                'locality', 
                'locality_name', 
                'education_type', 
                'drive_licences',
                'citizenship',
                'schedule',
                'employement_type',
                'gender',
                'relocation_ready',
                'travel_ready',
                'retraining_ready',
                'is_worldskills_participant',
                'has_qualifications',
                'status_ws',
                'unemp'
               ]
NUM_FEATURES = [
    'age',
    'experience',
    'salary_desired',
    'completeness_rate',
    'we_age_mnth',
    'graduation_year_ed',
    'pub_age_mnth'
]
BOOL_FEAT = ['relocation_ready', 'travel_ready',
             'retraining_ready',
             'is_worldskills_participant'  
            ]
IMP_SPACE = ['region', 'locality_name', 'education_type', 'drive_licences', 'gender',
            'relocation_ready', 'travel_ready','retraining_ready',
            'is_worldskills_participant',
             'has_qualifications','status_ws'            
            ]
IMP_NUM = ['age', 'we_age_mnth', 'finish_date_em', 'graduation_year_ed']

DROP_FEATURES = ['creation_date',
                 'modification_date',
                 'publish_date',
                 'achievements_em',
                 'position_em',
                 'employer_em',
                 'responsibilities_em',
                 'code_ws',
                 'is_international_ws',
                 'int_name_ws',
                 'ru_name_ws',
                 'institution_ed',
                 'description_ed',
                 'start_date_em',
                 'finish_date_em'
                ]
HIGH_GENERALITY = ['position',
                   'employer_em',
                   'position_em',
                   'achievements_em',
                   'responsibilities_em',
                   'institution_ed'
                  ]
def print_det(feat, train, test):
    print(f'Feature {feat}')
    print('\ntrain shape', train.shape[0])
    print('\ndetails train', train[feat].describe())
    if feat in test.columns:
        print('\ntest shape', test.shape[0])
        print('\ndetails test', test[feat].describe())
    print('\n---------------------------------')

    
def calc_we(x):
    st_date, fn_date, pub_date = x[0], x[1], x[2]
    if st_date is not None:
        if fn_date is not None:
            return (fn_date - st_date) // np.timedelta64(1, 'M')
        else:
            return (pub_date - st_date) // np.timedelta64(1, 'M')
    return -1


def convert_to_date(x):
    try:
        return pd.to_datetime(x, format="%Y-%m-%d")
    except:
        return np.NaN

def adj_we_ag(x):
    we_ag, age = x[0], x[1]
    if age is not None and we_ag is not None:
        if we_ag < 0:
            return np.NaN
        if (age - 20) > (we_ag / 12):
            return we_ag
        else: 
            return np.NaN
    else:
        return we_ag