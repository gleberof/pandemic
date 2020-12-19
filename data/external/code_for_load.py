

# КЛАДР
from sklearn.preprocessing import LabelEncoder 
locality = data["locality"].astype(str).str[8:11] 
locality[locality.str.len()!=3] = "0" 
le = LabelEncoder().fit(locality) 
le.transform(locality)



# Региональные кластеры по зарплате с дифференциацией по релоку, командировкам, полу, обучению
region_code_features = pd.read_csv("region_code_features.csv") 
data = data.merge(region_code_features, on="region", how="left")

for col in ['rfs', 'rts', 'rr_k', 'rr_cl',
            'tfs', 'tts', 'tr_k', 'tr_cl',
            'gfs', 'gts', 'gr_k', 'gr_cl',
            'rrfs', 'rrts', 'rrr_k', 'rrr_cl']:
    if col.endswith("_cl"):
        data.loc[:, col] = data.loc[:, col].fillna(-1)
    else:
        median = data.loc[:, col].median()
        data.loc[:, col] = data.loc[:, col].fillna(median)



# Агрегаты по образованию
agg_education_mult = pd.read_csv("agg_education_mult.csv")
data = data.merge(agg_education_mult, on="id", how="left")
data.loc[:, ["ins_all", "ins_diff", "year_min", "year_max"]] = \
    data.loc[:, ["ins_all", "ins_diff", "year_min", "year_max"]].fillna(-1)



# Агрегаты по работе
work_gr = pd.read_csv("work_gr.csv")
data = data.merge(work_gr, on="id", how="left")

work_cols = ["pos_count", "pos_uniq", "emp_uniq", "day_exp", "day_free"]
data.loc[:, work_cols] = data.loc[:, work_cols].fillna(-1)
