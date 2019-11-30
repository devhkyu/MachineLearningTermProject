import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN

DATA_DIR = Path('../data/world_development_indicators')
key_word_dict = {}
key_word_dict['Demography'] = ['population','birth','death','fertility','mortality','expectancy']
key_word_dict['Food'] = ['food','grain','nutrition','calories']
key_word_dict['Trade'] = ['trade','import','export','good','shipping','shipment']
key_word_dict['Health'] = ['health','desease','hospital','mortality','doctor']
key_word_dict['Economy'] = ['income','gdp','gni','deficit','budget','market','stock','bond','infrastructure']
key_word_dict['Energy'] = ['fuel','energy','power','emission','electric','electricity']
key_word_dict['Education'] = ['education','literacy']
key_word_dict['Employment'] =['employed','employment','umemployed','unemployment']
key_word_dict['Rural'] = ['rural','village']
key_word_dict['Urban'] = ['urban','city']

origin_dataset = pd.read_csv(DATA_DIR/"Indicators.csv")
origin_dataset["Country"] = origin_dataset["Year"].map(str) + origin_dataset["CountryName"]

# Reshape data
pivot_data = origin_dataset.pivot("Country","IndicatorName","Value")
pivot_data.columns = [x.lower() for x in pivot_data.columns]
print(pivot_data.columns)

# Find not nan data over than 700
over_7000_indicator = pivot_data.count()[pivot_data.count()>7000].keys()
print(over_7000_indicator)

# Find include keyword column upper data
contain_keyword_col = list()
for i in over_7000_indicator:
    if any(ext in i for ext in keyword_list):
        contain_keyword_col.append(i)
print(len(contain_keyword_col))

# Drop feature of above columns
not_keyword_col = [x if x not in contain_keyword_col else "" for x in pivot_data.columns]
not_keyword_col = list(filter(lambda a: a != "", not_keyword_col))
print(not_keyword_col)
pivot_data = pivot_data.drop(columns = not_keyword_col)
pivot_data = pivot_data.dropna()

# More delete columns

# Preprocessing
# Use more scaler
tmp = MinMaxScaler().fit_transform(pivot_data)
pivot_data = pd.DataFrame(tmp, columns=pivot_data.columns, index=range(len(pivot_data.index.values)))
print(pivot_data.shape)

# KMeans
km = KMeans()
km_result = km.fit_predict(pivot_data)
print(km_result)

# EM
gm = GaussianMixture()
gm_result = gm.fit_predict(pivot_data)
print(gm_result)

# DBSCAN
ds = DBSCAN()
ds_result = ds.fit_predict(pivot_data)
print(ds_result)