# Import Modules
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
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
bank_origin = pd.read_csv(DATA_DIR/'bank_prep_robust.csv')
# bank_origin = pd.read_csv(DATA_DIR/'bank_robust.csv')   # 0.9099

# Encoding Label for categorical data
le = LabelEncoder()
bank = MultiColumnLabelEncoder(columns=['job', 'marital', 'education', 'default', 'housing', 'loan',
                                      'contact', 'month', 'day_of_week', 'poutcome', 'y']).fit_transform(bank_origin)
# Select X, Y
# x = bank.drop(columns='y')
x = bank[['duration', 'pdays', 'nr.employed', 'euribor3m', 'campaign', 'age']]
y = bank['y']

# Split train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Parameter
# param_xgb = {"priors": [None], "var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]}
param_xgb = {"eta": [0.20, 0.25, 0.30, 0.35, 0.40], "gamma": [0, 1, 2, 3, 4, 5], "max_depth": [3, 4, 5, 6, 7]}
CV = 10

# XGBoost Classifier
start = time.time()
clf_log = LogisticRegression(max_iter=50, solver='liblinear')
clf_log.fit(x_train, y_train)
log_y = clf_log.predict(x_test)
print("Logistic Regression: %.4f" % (accuracy_score(y_test, log_y)))

"""
clf_nb = GaussianNB(priors=None, var_smoothing=0.1)
clf_nb.fit(x_train, y_train)
nb_y = clf_nb.predict(x_test)
print("Naive Bayesian: %.4f" % (accuracy_score(y_test, nb_y)))
"""

clf_svm = SVC(C=0.1, gamma=1, kernel='sigmoid')
clf_svm.fit(x_train, y_train)
svm_y = clf_svm.predict(x_test)
print("Support Vector Machine: %.4f" % (accuracy_score(y_test, svm_y)))

clf_xgbc = XGBClassifier(eta=0.10286751649448647, gamma=1.5458648766632133, max_depth=5)
clf_xgbc.fit(x_train, y_train)
xgbc_y = clf_xgbc.predict(x_test)
print("XGBoost Classifier: %.4f" % (accuracy_score(y_test, xgbc_y)))

y_pred = list()
y_pred.append(log_y)
# y_pred.append(nb_y)
y_pred.append(svm_y)
y_pred.append(xgbc_y)

clf_xgb = XGBClassifier(eta=0.5243, gamma=1.9861, max_depth=4)
xgcv = 10
print("XGBoost: %.4f" % sum((cross_val_score(clf_xgb, pd.DataFrame(y_pred).T, y_test, cv=xgcv))/xgcv))