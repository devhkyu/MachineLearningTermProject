import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def powerset(iterable):
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

DATA_DIR = Path('../data/world_development_indicators')
key_word_dict = {}
key_word_dict['Demography'] = ['population','birth','death','fertility','mortality','expectancy']
#key_word_dict['Food'] = ['food','grain','nutrition','calories']
#key_word_dict['Trade'] = ['trade','import','export','good','shipping','shipment']
key_word_dict['Health'] = ['health','desease','hospital','mortality','doctor']
#key_word_dict['Economy'] = ['income','gdp','gni','deficit','budget','market','stock','bond','infrastructure']
key_word_dict['Energy'] = ['fuel','energy','power','emission','electric','electricity']
key_word_dict['Education'] = ['education','literacy']
#key_word_dict['Employment'] =['employed','employment','umemployed','unemployment']
#key_word_dict['Rural'] = ['rural','village']
#key_word_dict['Urban'] = ['urban','city']

origin_dataset = pd.read_csv(DATA_DIR/"Indicators.csv")
origin_dataset["Country"] = origin_dataset["Year"].map(str) + origin_dataset["CountryName"]

# Reshape data
pivot_data = origin_dataset.pivot("Country","IndicatorName","Value")
pivot_data.columns = [x.lower() for x in pivot_data.columns]
print(pivot_data.columns)

# Find not nan data over than 700
over_7000_indicator = pivot_data.count()[pivot_data.count()>7000].keys()
print(over_7000_indicator)

contain_keyword_col = dict()
for i in key_word_dict.keys():
    contain_keyword_col[i] = list()

# Find include keyword column upper data
for kw in key_word_dict.keys():
    for i in over_7000_indicator:
        if any(ext in i for ext in key_word_dict[kw]):
            contain_keyword_col[kw].append(i)
print(contain_keyword_col)

# Drop feature of above columns
pivot_data_each_keyword = dict()
for kw in key_word_dict.keys():
    not_keyword_col = [x if x not in contain_keyword_col[kw] else "" for x in pivot_data.columns]
    not_keyword_col = list(filter(lambda a: a != "", not_keyword_col))
    pivot_data_each_keyword[kw] = pivot_data.drop(columns = not_keyword_col)
    pivot_data_each_keyword[kw] = pivot_data_each_keyword[kw].dropna()
    print(kw)
    print(pivot_data_each_keyword[kw].columns)
    print(len(pivot_data_each_keyword[kw].columns))
# More delete columns

keyword_powerset = powerset(key_word_dict.keys())

for keyList in keyword_powerset:
    tmp_table = pd.DataFrame()
    if len(keyList) == 0:
        continue
    for kw in keyList:
        if len(tmp_table) == 0:
            tmp_table = pivot_data_each_keyword[kw]
        else:
           tmp_table = pd.merge(tmp_table,pivot_data_each_keyword[kw],right_index=True,left_index=True)
    print(keyList,"\nrow length = ",len(tmp_table),"\ncolumn count = ",len(tmp_table.columns))

    # Preprocessing
    # Use more scaler
    tmp = MinMaxScaler().fit_transform(tmp_table)
    tmp_table = pd.DataFrame(tmp, columns=tmp_table.columns, index=range(len(tmp_table.index.values)))
    print(tmp_table.shape)

    # KMeans
    km = KMeans(n_clusters=6)
    km_result = km.fit_predict(tmp_table)
    print(km_result)
    print(silhouette_score(tmp_table,km_result))

    # EM
    gm = GaussianMixture(n_components=6)
    gm_result = gm.fit_predict(tmp_table)
    print(gm_result)
    print(silhouette_score(tmp_table,gm_result))

    # DBSCAN
    ds = DBSCAN(eps=0.1, min_samples=4)
    ds_result = ds.fit_predict(tmp_table)
    print(ds_result)
    print(silhouette_score(tmp_table,ds_result))
'''
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
'''