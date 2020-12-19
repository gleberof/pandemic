

# КЛАДР
from sklearn.preprocessing import LabelEncoder 
locality = data["locality"].astype(str).str[8:11] 
locality[locality.str.len()!=3] = "0" 
le = LabelEncoder().fit(locality) 
le.transform(locality)



# Региональные кластеры по зарплате с дифференциацией по релоку, командировкам, полу, обучению
region_code_features = pd.read_csv("region_code_features.csv") 
data = data.merge(region_code_features, on="region", how="left")




