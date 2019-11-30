# Import Modules
from pathlib import Path
import pandas as pd

DATA_DIR = Path('../data/world_development_indicators')
df_level = pd.read_csv(DATA_DIR/'income_level.csv', delimiter='|')

# Read csv from data
df = pd.read_csv(DATA_DIR/'Indicators.csv')

# CountryName, Year, IndicatorName
df_indicator = df[df['IndicatorCode'] == 'NY.GNP.PCAP.CD']
df_indicator = df_indicator[['CountryName', 'Year', 'Value']]
df_pivot = df_indicator.pivot('CountryName', 'Year', 'Value')

# Fill na method using interpolate
df_pivot = df_pivot.interpolate(method='linear', limit_direction='both', axis=1)
list_year = df_indicator[df_indicator['Year'] >= 1987]['Year'].tolist()
list_value = df_indicator[df_indicator['Year'] >= 1987]['Value'].tolist()
list_countryName = df_indicator[df_indicator['Year'] >= 1987]['CountryName'].tolist()

# Comparing with income_level.csv
list_result = []
for inx in range(len(list_value)):
    if list_year[inx] >= 1987:
        if int(list_value[inx]) <= int((df_level[str(list_year[inx])])[0]):
            temp = 'Low'
        elif int(list_value[inx]) <= int((df_level[str(list_year[inx])])[1]):
            temp = 'Lower-Middle'
        elif int(list_value[inx]) <= int((df_level[str(list_year[inx])])[2]):
            temp = 'Upper-Middle'
        else:
            temp = 'High-Level'
        list_result.append(temp)

# Make dataFrame and save to csv
df_final = pd.DataFrame({'CountryName': list_countryName, 'Year': list_year, 'Income-Level': list_result})
df_final.to_csv(DATA_DIR/'Country_Income_Level.csv')