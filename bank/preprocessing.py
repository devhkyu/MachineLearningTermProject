from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler
from pathlib import Path
import pandas as pd

# Set data path
DATA_DIR = Path('../data/bankmarketing')

# Read csv to df
bank = pd.read_csv(DATA_DIR/'bank-additional-full.csv', delimiter=';')

# Making Categorical data: Whether pdays are
"""
list_pdays_cat = []
for inx in range(len(bank)):
    if bank['pdays'][inx] == 999:
        list_pdays_cat.append(0)
    else:
        list_pdays_cat.append(1)
print(list_pdays_cat)
"""

# MinMaxScaler
scale_min_max = MinMaxScaler()
min_max = bank.copy()
min_max[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']] = scale_min_max.fit_transform(bank[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']])
min_max.to_csv(DATA_DIR/'bank_minmax.csv')

# RobustScaler
scale_robust = RobustScaler()
robust = bank.copy()
robust[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']] = scale_robust.fit_transform(bank[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']])
robust.to_csv(DATA_DIR/'bank_robust.csv')

# MaxAbsScaler
scale_max_abs = MaxAbsScaler()
max_abs = bank.copy()
max_abs[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']] = scale_max_abs.fit_transform(bank[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']])
max_abs.to_csv(DATA_DIR/'bank_maxabs.csv')

# StandardScaler
scale_std = StandardScaler()
std = bank.copy()
std[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']] = scale_std.fit_transform(bank[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']])
std.to_csv(DATA_DIR/'bank_std.csv')