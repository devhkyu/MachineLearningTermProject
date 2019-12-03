# Import Modules
from pathlib import Path
from sklearn.utils import resample
import pandas as pd

# Set data path
DATA_DIR = Path('../data/bankmarketing')

# Read csv to df
bank_origin = pd.read_csv(DATA_DIR/'bank-additional-full.csv', delimiter=';')

# Drop rows having unknown value
bank_origin = bank_origin[bank_origin.job != 'unknown']
bank_origin = bank_origin[bank_origin.marital != 'unknown']
bank_origin = bank_origin[bank_origin.loan != 'unknown']
bank_origin = bank_origin[bank_origin.housing != 'unknown']

# Set age as range
"""
bank_origin['age'].loc[(bank_origin['age'] >= 0) & (bank_origin['age'] < 10)] = 0
bank_origin['age'].loc[(bank_origin['age'] >= 10) & (bank_origin['age'] < 20)] = 1
bank_origin['age'].loc[(bank_origin['age'] >= 20) & (bank_origin['age'] < 30)] = 2
bank_origin['age'].loc[(bank_origin['age'] >= 30) & (bank_origin['age'] < 40)] = 3
bank_origin['age'].loc[(bank_origin['age'] >= 40) & (bank_origin['age'] < 50)] = 4
bank_origin['age'].loc[(bank_origin['age'] >= 50) & (bank_origin['age'] < 60)] = 5
bank_origin['age'].loc[(bank_origin['age'] >= 60) & (bank_origin['age'] < 70)] = 6
bank_origin['age'].loc[(bank_origin['age'] >= 70) & (bank_origin['age'] < 80)] = 7
bank_origin['age'].loc[(bank_origin['age'] >= 80) & (bank_origin['age'] < 90)] = 8
bank_origin['age'].loc[(bank_origin['age'] >= 90) & (bank_origin['age'] < 100)] = 9
"""

# Resampling Data
df_majority = bank_origin[bank_origin['y'] == 'yes']
df_minority = bank_origin[bank_origin['y'] == 'no']
df_minority_upsampled = resample(df_minority, replace=True, n_samples=int(2000), random_state=123)
df_majority_downsampled = resample(df_majority, replace=True, n_samples=int(2000), random_state=123)
bank = pd.concat([df_minority_upsampled, df_majority_downsampled])

# Save to csv file
# bank_origin.to_csv(DATA_DIR/'bank_prep.csv')
bank.to_csv(DATA_DIR/'bank_res4.csv')