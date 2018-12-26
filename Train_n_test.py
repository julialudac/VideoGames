import csv
import numpy as np # linear algebra
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection  import  train_test_split

# personal imports
import data_extractor as de
from multi_labels_encoder import ThreeFeaturesEncoder
import accuracy_extractor as ae


class TrainValidateTest:
    """Train and test... and predict. How to use:
    1/ Train
    2/ Validate on a validation data (ie see accuracy) or test on a test data (ie output estimated labels)
    Note: Since there is some random into the learning process, the acuracy result may change with same datasets
    and same parameters n_estimators and max_depth.
    """

    def __init__(self, n_estimators, max_depth):
        self.decision_tree = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)

    def train(self, df_training_numerical):
        """Train a random forest
        :param df_training_numerical: Encoded train DataFrame (the DataFrame includes the labels) whose values are the counts of the
        actions, the player id and the played race
        """
        self.decision_tree.fit(df_training_numerical.iloc[:, 1:].values, df_training_numerical.id_player.values)

    def validate(self, df_validation_numerical):
        """Validate => displays the score of the learnt classifier on a validation data.
        :param df_validation_numerical: Encoded train DataFrame (the DataFrame includes the labels) whose values are the counts of the
        actions, the player id and the played race
        """
        nolabel_df_validation = df_validation_numerical.drop(axis=1, labels="id_player")
        predicted = self.decision_tree.predict(nolabel_df_validation.values)
        labels = df_validation_numerical.id_player.values
        print("accuracy:", ae.get_accuracy(labels, predicted))

    def test(self, df_testing_numerical, encoder):
        """Test => Output labels in a csv file.
        :param df_testing_numerical: Encoded test DataFrame (the DataFrame includes the labels that are all 0's) whose
        values are the counts of the actions, the player id and the played race
        :param encoder: The encoder that was used to ecode the DataFrame passed as a parameter
        """
        nolabel_df_testing = df_testing_numerical.drop(axis=1, labels="id_player")
        predicted = self.decision_tree.predict(nolabel_df_testing.values)
        decoded_predicted = encoder.decode_labels(predicted)  # We decode the encoded predictions
        indices = range(1, len(predicted) + 1)
        output_df = pd.DataFrame({"RowId": indices, "prediction": decoded_predicted})
        output_df.to_csv("test_labels.CSV", index=False)


class ExtractTrainValidateTest:
    """
    1/ Extract and process DataFrames
    2/ Train and validate => it will yield an accuracy
    This class is not really to output estimations about players' ids but more used to find the best
    hyperparameters: t, n_estimators, max_depth, df_training, df_validation to get the best accuracy.
    Then, when you are happy with the parameters, you can use them into notebook
    "extracting_training_validating_testing" to output results for a test dataset in a csv file.
    """

    def __init__(self):
        self.tnv = None

    def extract_and_process_df(self, t):
        """
        :param t: The time limit where to stop extracting the data. When not a multiple of 5, it is rounded to
        upper 5.
        """
        # Prepare train and test original datasets (+ labels)
        self.df_training = de.get_dataframe("../input/minitrain.CSV", training=True, limit_seconds=t)  # OK
        self.df_validation = de.get_dataframe("../input/minitest.CSV", training=True, limit_seconds=t)  # OK

        # Encoding
        # -- Learning dataset encoding
        self.encoder = ThreeFeaturesEncoder(self.df_training)
        # -- Encoding dataset : training and validation
        self.encoded_df_training = self.encoder.encode_df(self.df_training)
        self.encoded_df_validation = self.encoder.encode_df(self.df_validation)

        # Getting the datasets we're gonna work on: Convert dataset into another one
        self.df_training_numerical = de.transform_sample(self.encoded_df_training, True)
        self.df_validation_numerical = de.transform_sample(self.encoded_df_validation, True)

        # Conform validation DataFrame columns to training DataFrame columns
        self.df_validation_numerical = de.conform_test_to_training(self.df_training_numerical,
                                                                   self.df_validation_numerical)
        self.df_validation_numerical = self.df_validation_numerical.fillna(0)

    def train_and_validate(self, n_estimators, max_depth):
        self.tnv = TrainValidateTest(n_estimators, max_depth)
        self.tnv.train(self.df_training_numerical)
        self.tnv.validate(self.df_validation_numerical)

