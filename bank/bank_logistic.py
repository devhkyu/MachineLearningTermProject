# Import Modules
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Multi Label Encoding
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# Read csv to df
DATA_DIR = Path('../data/bankmarketing')

# bank_origin = pd.read_csv(DATA_DIR/'bank-additional-full.csv', delimiter=';')
# bank_origin = pd.read_csv(DATA_DIR/'bank_std.csv')
bank_origin = pd.read_csv(DATA_DIR/'bank_prep_robust.csv')

# Encoding Label for categorical data
le = LabelEncoder()
bank = MultiColumnLabelEncoder(columns=['job', 'marital', 'education', 'default', 'housing', 'loan',
                                      'contact', 'month', 'day_of_week', 'poutcome']).fit_transform(bank_origin)
# Select X, Y
# x = bank.drop(columns='y')
x = bank[['duration', 'pdays', 'nr.employed', 'euribor3m', 'campaign', 'age']]
y = bank['y']

# Split train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Parameter
param_logistic = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'max_iter': [50, 100, 200]}
# param_logistic = {'solver': ['saga'], 'max_iter': [50]}
CV = 10

# Logistic Regression
start = time.time()
clf_log = LogisticRegression()
grid_log = GridSearchCV(estimator=clf_log, param_grid=param_logistic, scoring='accuracy', n_jobs=4, cv=CV, verbose=10)
grid_log.fit(x, y)
print('\nStandard Scaled Data\n')
print('Parameter:', grid_log.best_params_)
print('Score:', grid_log.best_score_)
print('Time:', time.time()-start, "sec")