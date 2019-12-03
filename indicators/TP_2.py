import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from collections import Counter
import operator
import json

def powerset(iterable):
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def silhouette_scoring(estimator, x):
    cluster = estimator.fit_predict(x)
    try:
        score = silhouette_score(x,cluster)
    except ValueError:
        score = 0
    return score

def cal_purity(model, x, gold):
    cluster_result = model.fit_predict(x)
    count_el = dict(sorted(Counter(cluster_result).items(), key=operator.itemgetter(0)))  # count each cluster's volume
    clustering_result = dict()  # collect each cluster's number of each elements
    gold["cluster"] = cluster_result
    popul = 0

    all_len = 0
    # cal each cluster
    for i in count_el.keys():
        cluster_counter = Counter(gold.loc[gold["cluster"] == i, "Income-Level"])
        clustering_result[i] = dict(cluster_counter)
        popul += max(cluster_counter.values())
        all_len += sum(cluster_counter.values())

    return float(popul)/all_len

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

km_hpTune = {"n_clusters":[4,6,8,10,12],"max_iter":[50,100,200]}
gm_hpTune = {"n_components":[4,6,8,10,12],"covariance_type":['full','tied','diag','spherical'], "max_iter":[50,100,200]}
dc_hpTune ={"eps":[0.001,0.01,0.05,0.1,0.2,0.5],"min_samples":[2,5,10,15]}

origin_dataset = pd.read_csv(DATA_DIR/"Indicators.csv")
origin_dataset["Country"] = origin_dataset["Year"].map(str) + origin_dataset["CountryName"]
incomeLevel_dataset = pd.read_csv(DATA_DIR/"Country_Income_Level.csv",index_col=0)
incomeLevel_dataset["Country"] = incomeLevel_dataset["Year"].map(str) + incomeLevel_dataset["CountryName"]
incomeLevel_dataset = incomeLevel_dataset.drop(columns=["Year","CountryName"])
# Reshape data
pivot_data = origin_dataset.pivot("Country","IndicatorName","Value")
pivot_data.columns = [x.lower() for x in pivot_data.columns]

# Find not nan data over than 700
over_7000_indicator = pivot_data.count()[pivot_data.count()>7000].keys()

contain_keyword_col = dict()
for i in key_word_dict.keys():
    contain_keyword_col[i] = list()

# Find include keyword column upper data
for kw in key_word_dict.keys():
    for i in over_7000_indicator:
        if any(ext in i for ext in key_word_dict[kw]):
            contain_keyword_col[kw].append(i)

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

hyper_tune = dict()
purity_result = dict()
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
    tmp_table = pd.DataFrame(tmp, columns=tmp_table.columns, index=tmp_table.index.values)
    print(tmp_table.shape)

    hyper_tune["-".join(keyList)] = dict()
    purity_result["-".join(keyList)] = dict()

    tmp_incomeLevel = incomeLevel_dataset[incomeLevel_dataset["Country"].isin(tmp_table.index.values)].reset_index()
    print(tmp_table.index.values)
    tmp_table = tmp_table.reindex(tmp_incomeLevel["Country"])

    # DBSCAN
    print(tmp_table.shape)
    ds = DBSCAN()
    ds_search = GridSearchCV(estimator=ds, param_grid=dc_hpTune, scoring=silhouette_scoring, n_jobs=3, cv=5, verbose=10)
    ds_result = ds_search.fit(tmp_table)
    print(ds_result.best_params_)
    print(ds_result.best_score_)
    hyper_tune["-".join(keyList)]["ds"] = ds_result.best_params_

    ds_best = DBSCAN(**(ds_result.best_params_))
    purity = cal_purity(ds_best,tmp_table,tmp_incomeLevel)
    print(purity)
    purity_result["-".join(keyList)]["ds"] = purity

    # KMeans
    km = KMeans()
    km_search = GridSearchCV(estimator=km, param_grid=km_hpTune, scoring=silhouette_scoring, n_jobs=3, cv=5, verbose=10)
    km_result = km_search.fit(tmp_table)
    print(km_result.best_params_)
    print(km_result.best_score_)
    hyper_tune["-".join(keyList)]["km"] = km_result.best_params_

    km_best = KMeans(**(km_result.best_params_))
    purity = cal_purity(km_best,tmp_table,tmp_incomeLevel)
    print(purity)
    purity_result["-".join(keyList)]["km"] = purity


    # gausian navie basis
    gm = GaussianMixture()
    gm_search = GridSearchCV(estimator=gm, param_grid=gm_hpTune, scoring=silhouette_scoring, n_jobs=3, cv=5, verbose=10)
    gm_result = gm_search.fit(tmp_table)
    print(gm_result.best_params_)
    print(gm_result.best_score_)
    hyper_tune["-".join(keyList)]["gm"] = gm_result.best_params_

    gm_best = GaussianMixture(**(gm_result.best_params_))
    purity = cal_purity(gm_best,tmp_table,tmp_incomeLevel)
    print(purity)
    purity_result["-".join(keyList)]["gm"] = purity


print(hyper_tune)
print(purity_result)

with open("result/hyper_tune.json", "w") as json_file:
    json.dump(hyper_tune, json_file)


with open("result/purity_result.json", "w") as json_file:
    json.dump(purity_result, json_file)