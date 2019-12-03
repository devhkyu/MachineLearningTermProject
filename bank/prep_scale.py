from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler
from pathlib import Path
import pandas as pd

# Set data path
DATA_DIR = Path('../data/bankmarketing')

# Read csv to df
# bank = pd.read_csv(DATA_DIR/'bank-additional-full.csv', delimiter=';')
bank = pd.read_csv(DATA_DIR/'bank_prep.csv')

# MinMaxScaler
scale_min_max = MinMaxScaler()
min_max = bank.copy()
min_max[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']] = scale_min_max.fit_transform(bank[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']])
# min_max[['duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']] = scale_min_max.fit_transform(bank[['duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']])
min_max.to_csv(DATA_DIR/'bank_prep_minmax.csv')

# RobustScaler
scale_robust = RobustScaler()
robust = bank.copy()
robust[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']] = scale_robust.fit_transform(bank[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']])
# robust[['duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']] = scale_robust.fit_transform(bank[['duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']])
robust.to_csv(DATA_DIR/'bank_prep_robust.csv')

# MaxAbsScaler
scale_max_abs = MaxAbsScaler()
max_abs = bank.copy()
max_abs[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']] = scale_max_abs.fit_transform(bank[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']])
# max_abs[['duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']] = scale_max_abs.fit_transform(bank[['duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']])
max_abs.to_csv(DATA_DIR/'bank_prep_maxabs.csv')

# StandardScaler
scale_std = StandardScaler()
std = bank.copy()
std[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']] = scale_std.fit_transform(bank[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']])
# std[['duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']] = scale_std.fit_transform(bank[['duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']])
std.to_csv(DATA_DIR/'bank_prep_std.csv')