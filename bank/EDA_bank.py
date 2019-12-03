from pathlib import Path
import pandas as pd

# Set data path
DATA_DIR = Path('../data/bankmarketing')

# Read csv to df
bank = pd.read_csv(DATA_DIR/'bank-additional-full.csv', delimiter=';')

# Check whether there is null value
print('===================================')
print('Check wheter there is null value')
print(bank.isnull().sum())
print('===================================')

for inx in bank.columns:
    print(inx)
    print(bank[inx].unique())
    print('===================================')

# Check people who replied unknown: dirty data
print('Unknown count')
unknown_list = ['job', 'marital', 'loan', 'housing', 'default']
for inx in unknown_list:
    print("{}: {}".format(inx, bank[inx].value_counts()['unknown']))
print(bank[(bank['loan'] == 'unknown')][bank['housing'] == 'unknown'].head())
print('===================================')

"""
age = bank['age']
job = bank['job']
plt.bar(job, age)
plt.xlabel('job')
plt.ylabel('age')
plt.show()
"""