# Import Modules
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pandas as pd


# Multi Label Encoding
class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns# array of column names to encode

    def fit(self, X, y=None):
        return self # not relevant here

    def transform(self, X):
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


DATA_DIR = Path('../data/bankmarketing')
bank_origin = pd.read_csv(DATA_DIR/'bank-additional-full.csv', delimiter=';')
bank_labeled = MultiColumnLabelEncoder(columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']).fit_transform(bank_origin)
corr = bank_labeled.corr(method='pearson')
corr.to_excel(DATA_DIR/'bank_correlation.xlsx')