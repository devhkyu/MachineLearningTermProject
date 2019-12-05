import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from collections import Counter
import operator
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def powerset(iterable):
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def visualizeClusters(X, labels):
    estimator = PCA(n_components=2)
    PCA_x = estimator.fit_transform(X)
    x, y = PCA_x[:, 0], PCA_x[:, 1]

    PCA_df = pd.DataFrame({'x': x, 'y': y})

    scatterPlot(PCA_df, labels)

def scatterPlot(df, labels):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    colors.append([0, 0, 0, 1])

    # make color vector
    cvec = [colors[label] for label in labels]

    
    plt.scatter(df['x'], df['y'], c=cvec)
    plt.show()
    return

def plotCluster(df,predict,model):
    df_plot=df.copy()
    country=list()
    for i in range(len(df)):
        country.append(df.index.array[i].split("_")[0])
    df_plot['country']=country
    if model!="origin":
        df_plot['predict']=label
        df_plot.groupby('country').agg(mode)
        title="Clustering of Countries based on"+model
        ti=np.unique(label)
        d=dict(type='choropleth',locations=df_plot.index,locationmode="country names",
               z=df_plot['predict'],text=df_plot.index,colorbar={"title":"cluster","tickmode":"array","tickvals":ti},
               colorscale="Viridis" 
              )
    else:
        df_plot=df_plot.groupby('country').mean()
        d=dict(type='choropleth',locations=df_plot.index,locationmode="country names",
           z=df_plot[target_id],text=df_plot.index,colorbar={"title":"f" ,"tickmode":"array","tickvals":[0,20,40,60,80,100,120,140,160,180,200]},
           colorscale="Viridis"
          )
        title=target_id
    layout = dict(title=title,height=500,
                  geo=dict(showframe = False,
                           projection = {'type':'mercator'}))
    go.Figure(data=[d],layout=layout).show()

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
# More delete columns

keyword_powerset = powerset(key_word_dict.keys())

hyper_tune = dict()
sliho_result = dict()
purity_result = dict()
scaler = ["minmax","robust","standard"]

for i in scaler:
    with open("result/hyper_tune_" + i + ".json", "r") as f:
        hyper_tune[i] = json.load(f)
    with open("result/purity_result_"+i+".json") as f:
        purity_result[i] = json.load(f)
    with open("result/silhouette_result_"+i+".json") as f:
        sliho_result[i] = json.load(f)

for keyList in keyword_powerset:
    tmp_table = pd.DataFrame()
    if len(keyList) == 0:
        continue
    for kw in keyList:
        if len(tmp_table) == 0:
            tmp_table = pivot_data_each_keyword[kw]
        else:
            tmp_table = pd.merge(tmp_table,pivot_data_each_keyword[kw],right_index=True,left_index=True)
    
    # Preprocessing
    # Use more scaler
    for i in scaler:
        if i == "minmax":
            scale = MinMaxScaler()
        elif i == "robust":
            scale = RobustScaler()
        elif i == "standard":
            scale = StandardScaler()
        tmp = scale.fit_transform(tmp_table)
        tmp_table_each_scale = pd.DataFrame(tmp, columns=tmp_table.columns, index=tmp_table.index.values)

        tmp_incomeLevel = incomeLevel_dataset[incomeLevel_dataset["Country"].isin(tmp_table_each_scale.index.values)].reset_index()
        tmp_table_each_scale = tmp_table_each_scale.reindex(tmp_incomeLevel["Country"])

        gold = tmp_incomeLevel["Income-Level"]

        #DBSCAN
        ds = DBSCAN(**(hyper_tune[i]["-".join(keyList)]["ds"]))
        ds_result = ds.fit_predict(tmp_table_each_scale)
        ds_purity = purity_result[i]["-".join(keyList)]["ds"]
        df_silhouette = sliho_result[i]["-".join(keyList)]["ds"]

        #GMM
        gm = GaussianMixture(**(hyper_tune[i]["-".join(keyList)]["gm"]))
        gm_result = gm.fit_predict(tmp_table_each_scale)
        gm_purity = purity_result[i]["-".join(keyList)]["gm"]
        gm_silhouette = sliho_result[i]["-".join(keyList)]["gm"]

        #KMeans
        km = KMeans(**(hyper_tune[i]["-".join(keyList)]["km"]))
        km_result = km.fit_predict(tmp_table_each_scale)
        km_purity = purity_result[i]["-".join(keyList)]["km"]
        km_silhouette = sliho_result[i]["-".join(keyList)]["km"]
        plt.figure(figsize=(8, 8))
        plt.title("-".join(keyList))
        visualizeClusters(tmp_table_each_scale,LabelEncoder().fit_transform(gold))
        '''
        plt.figure(figsize=(8, 8))
        plt.title("-".join(keyList)+"\nDBSCAN"+"\n"+i+"\n"+str(hyper_tune[i]["-".join(keyList)]["ds"])+"\n{silhouette:"+str(sliho_result[i]["-".join(keyList)]["ds"])+", purity:"+str(purity_result[i]["-".join(keyList)]["ds"])+"}")
        visualizeClusters(tmp_table_each_scale,ds_result)
        plt.figure(figsize=(8, 8))
        plt.title("-".join(keyList)+"\nGMM"+"\n"+i+"\n"+str(hyper_tune[i]["-".join(keyList)]["gm"])+"\n{silhouette:"+str(sliho_result[i]["-".join(keyList)]["gm"])+", purity:"+str(purity_result[i]["-".join(keyList)]["gm"])+"}")
        visualizeClusters(tmp_table_each_scale,gm_result)
        print(gm_result)
        plt.figure(figsize=(8, 8))
        plt.title("-".join(keyList)+"\nKMeans"+"\n"+i+"\n"+str(hyper_tune[i]["-".join(keyList)]["km"])+"\n{silhouette:"+str(sliho_result[i]["-".join(keyList)]["km"])+", purity:"+str(purity_result[i]["-".join(keyList)]["km"])+"}")
        visualizeClusters(tmp_table_each_scale,km_result)
        print(km_result)
        '''