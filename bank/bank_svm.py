# Import Modules
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
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
bank_origin = pd.read_csv(DATA_DIR/'bank_res4.csv')

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
# param_svm = {'C': [0.1, 1.0, 10.0], 'gamma': [1, 10, 100], 'kernel': ['linear', 'rbf', 'sigmoid']}
param_svm = {'C': [0.1, 0.5, 1, 1.5, 2], 'gamma': [0.1, 0.5, 1, 1.5, 2], 'kernel': ['linear']}
CV = 5

# Support Vector Machine
start = time.time()
clf_svm = SVC(probability=True)
grid_svm = GridSearchCV(estimator=clf_svm, param_grid=param_svm, scoring='accuracy', n_jobs=4, cv=CV, verbose=10)
grid_svm.fit(x, y)
print('Parameter:', grid_svm.best_params_)
print('Score:', grid_svm.best_score_)
print('Time:', time.time()-start, "sec")

