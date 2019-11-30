from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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
bank_origin = pd.read_csv(DATA_DIR/'bank-additional-full.csv', delimiter=';')
print(bank_origin.isnull().sum())

# Encoding Label for categorical data
le = LabelEncoder()
bank = MultiColumnLabelEncoder(columns=['job', 'marital', 'education', 'default', 'housing', 'loan',
                                      'contact', 'month', 'day_of_week', 'poutcome', 'y']).fit_transform(bank_origin)
# Select X, Y
x = bank.drop(columns='y')
y = bank['y']

# Split train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Parameter
param_logistic = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'max_iter': [50, 100, 200]}
param_svm = {'C': [0.1, 1.0, 10.0], 'gamma': [1, 10, 100], 'kernel': ['linear', 'rbf', 'sigmoid', 'poly']}
param_xgb = {'eta': [0.1, 0.2, 0.3], 'gamma': [0, 1, 10], 'alpha': [0, 1, 10]}
param_nb = {"priors":[None],"var_smoothing":[1e-9,1e-7,1e-5,1e-3,1e-1]}
CV = 10
# naive baysian
start = time.time()
nb_log = GaussianNB(priors=None)
grid_nb_log = GridSearchCV(estimator=nb_log, param_grid=param_nb, scoring='accuracy', n_jobs=1, cv=CV, verbose=10)
grid_nb_log.fit(x, y)
print('Parameter:', grid_nb_log.best_params_)
print('Score:', grid_nb_log.best_score_)
print('Time:', time.time()-start, "sec")

# Logistic Regression
start = time.time()
clf_log = LogisticRegression()
grid_log = GridSearchCV(estimator=clf_log, param_grid=param_logistic, scoring='accuracy', n_jobs=1, cv=CV, verbose=10)
grid_log.fit(x, y)
print('Parameter:', grid_log.best_params_)
print('Score:', grid_log.best_score_)
print('Time:', time.time()-start, "sec")


'''
# Support Vector Machine
start = time.time()
clf_svm = SVC(probability=True)
grid_svm = GridSearchCV(estimator=clf_svm, param_grid=param_svm, scoring='accuracy', n_jobs=1, cv=CV)
grid_svm.fit(x, y)
print('Parameter:', grid_svm.best_params_)
print('Score:', grid_svm.best_score_)
print('Time:', time.time()-start, "sec")
'''