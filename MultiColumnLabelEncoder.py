""" Inpired from src: https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
But for my case, I want all the columns to be encoded
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MultiColumnLabelEncoder:
    def __init__(self, colnames=None):
        self.colnames = colnames  # array of column names to encode. This is a DataFrame
        self.les = []  # a list of label encoders, one for each encoded column
        self.classes = []  # a 2d list of the classes in an ecndoded form

    def fit(self, X, y=None):  # Now: only adapted to when we want all the columns of the DataFrame to be encoded
        for colname in X.columns.values:
            # "unk": We take care of possible values in the test set that may not be encountered during training
            le = LabelEncoder().fit(list(X[colname]) + ["unk"])
            # print("list(X[colname]) + [unk]:", list(X[colname]) + ["unk"])
            self.les.append(le)
            self.classes.append(le.classes_)

    def transform(self, X):
        '''
        X the DataFrame to encode. X is a DataFrame.
        '''
        output = X.copy()
        if self.colnames is not None: # NOT USED IN OUR CASE
            for col in self.colnames:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for ind in range(len(self.les)):
                # We take care of possible values in the test set that may not be encountered during training
                output[X.columns.values[ind]] = output[X.columns.values[ind]].map(
                    lambda s: "unk" if s not in self.les[ind].classes_ else s)
                output[X.columns.values[ind]] = self.les[ind].transform(output[X.columns.values[ind]])
        return output

    def inverse_transform(self, X):  # Useful ???
        """X the DataFrame to decode. X is a DataFrame.
        Here: only adapted when self.colnames = None"""
        output = X.copy()
        for ind in range(len(self.les)):
            output[X.columns.values[ind]] = self.les[ind].inverse_transform(output[X.columns.values[ind]])
        return output


# Create some toy data in a Pandas dataframe
fruit_data = pd.DataFrame({
    'fruit': ['apple', 'orange', 'pear', 'orange'],
    'color': ['red', 'orange', 'green', 'green'],
    'weight': [5, 6, 3, 4]
})