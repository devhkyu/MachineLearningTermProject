from pathlib import Path
import pandas as pd
import numpy as np
import re

# Set data path
DATA_DIR = Path('../data/world_development_indicators')

# Read csv to df
country = pd.read_csv(DATA_DIR/'Country.csv', delimiter=',')
countryNotes = pd.read_csv(DATA_DIR/'CountryNotes.csv', delimiter=',')
footNotes = pd.read_csv(DATA_DIR/'Footnotes.csv', delimiter=',')
indicators = pd.read_csv(DATA_DIR/'Indicators.csv', delimiter=',')
series = pd.read_csv(DATA_DIR/'Series.csv', delimiter=',')
seriesNotes = pd.read_csv(DATA_DIR/'SeriesNotes.csv', delimiter=',')

indicator_array = indicators[['IndicatorName', 'IndicatorCode']].drop_duplicates().values
modified_indicators = []
unique_indicator_codes = []

for ele in indicator_array:
    indicator = ele[0]
    indicator_code = ele[1].strip()
    if indicator_code not in unique_indicator_codes:
        # delete , ( ) from the IndicatorNames
        new_indicator = re.sub('[,()]', "", indicator).lower()
        # replace - with "to" and make all words into lower case
        new_indicator = re.sub('-', " to ", new_indicator).lower()
        modified_indicators.append([new_indicator, indicator_code])
        unique_indicator_codes.append(indicator_code)

Indicators = pd.DataFrame(modified_indicators,columns=['IndicatorName','IndicatorCode'])
Indicators = Indicators.drop_duplicates()

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

feature = 'Economy'
for indicator_ele in Indicators.values:
    for ele in key_word_dict[feature]:
        word_list = indicator_ele[0].split()
        if ele in word_list or ele+'s' in word_list:
            print(indicator_ele[0])
            break