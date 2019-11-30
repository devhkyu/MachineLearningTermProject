from pathlib import Path
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

DATA_DIR = Path('../data/world_development_indicators')
df_level = pd.read_csv(DATA_DIR/'income_level.csv', delimiter='|')
# print(df_level[df_level['level'] == 'Low'])
# print(df_level['1987'][0])

# Read csv from data
df = pd.read_csv(DATA_DIR/'Indicators.csv')

# CountryName, Year, IndicatorName
df_indicator = df[df['IndicatorCode'] == 'NY.GNP.PCAP.CD']
df_indicator = df_indicator[['CountryName', 'Year', 'Value']]
df_pivot = df_indicator.pivot('CountryName', 'Year', 'Value')

# Fill na method using interpolate
# df_pivot = df_pivot.interpolate(method='linear', limit_direction='both', axis=1)
# a = pd.melt(df_pivot, id_vars=["CountyName"], value_vars=df_pivot.columns[1:])
# print(a)

"""
for i in df_pivot.index:
    temp = df_pivot.loc[i, :]
"""