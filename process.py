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
    
    return train, test