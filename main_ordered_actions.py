
# coding: utf-8


import csv
import numpy as np # linear algebra
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection  import  train_test_split

# personal imports
import data_extractor
from encoder_categorical_numerical import Encoder_Categorical_Numerical
import accuracy_extractor


validation = True
training_file = "./input/TRAIN.CSV" if not validation else "./input/minitrain.csv"
validation_file = "./input/minitest.csv"
#validation_file
testing_file = "./input/TEST.CSV" if not validation else "./input/minitest.csv"


class Transform_Data:
	def __init__(self, training_file, operation_type, validation_file=None, testing_file=None):
		self.operation_type = operation_type
		self.path_training_file = training_file
		self.path_validation_file = validation_file
		self.path_testing_file = testing_file

    def __ceil_to_five__(self, seconds):
        """Approximates the number of seconds to the upper multiple of 5.
        :param seconds: number of seconds to approximate
        :return: approxilated seconds
        """
        modulo = seconds % 5
        if modulo == 0:
            return seconds
        else:
            return seconds - modulo + 5

    def __remove_tX__(self, row):
        """Get a new list of words with words of type tX (X a number) removed
        :param myrow: a list of words"""
        tX = re.compile("t\d")
        new_row = []
        for cell in row:
            if tX.match(cell):
                pass
            else:
                new_row.append(cell)
        return new_row

    def __get_maximum_time_match__(self, row, default_time=0):
        """Get the approximate time of a match
        :param myrow: one match"""
        tX = re.compile("t\d")
        for cell in row[::-1]:
            if tX.match(cell):
                return int(cell[1:])
        return default_time

    def __get_dataframe__(self, path_file, training, limit_seconds):
        """
        :param path_file:
        :param training: boolean
        :param limit_seconds: int
        :return: A DataFrame with columns named id_player, played_race, 0... n, with n the number of kept actions.
        """
        limit_seconds = __ceil_to_five__(limit_seconds)
        stop_word = "t" + str(limit_seconds)
        extracted = []
        largest_column_count = 0

        # Loop the data lines
        with open(path_file) as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\n')
            for row in spamreader:
                myrow = row[0].split(',')
                try:
                    stop_index = myrow.index(stop_word)
                except:
                    stop_index = -1
                #compute the time of a match and insert it to the row
                time_match = __get_maximum_time_match__(myrow)
                #delete useless tX
                myrow = __remove_tX__(myrow[0:stop_index])
                #count the number of columns
                column_count = len(myrow)
                largest_column_count = column_count if largest_column_count < column_count else largest_column_count
                #insert time_match as feature
                index_time_match = 2 if training else 1
                myrow.insert(index_time_match, time_match)
                extracted.append(myrow)
        column_names = []
        if training:
            column_names = ["id_player", "played_race", "time_match"] + [i for i in range(0, largest_column_count - 2)]
        else:
            column_names = ["played_race", "time_match"] + [i for i in range(0, largest_column_count - 1)]
        return pd.DataFrame(extracted, columns = column_names)

	def process_data(self, observed_time):
		self.df_training = self.__get_dataframe__(self.path_training_file, training=True, limit_seconds=observed_time)
		self.encoder = Encoder_Categorical_Numerical(self.df_training)
		self.df_training = self.encoder.encode_df(df_training)

		if self.operation_type == "TESTING":
			self.df_testing = self.__get_dataframe__(self.path_testing_file, training=False, limit_seconds=observed_time)
			self.df_testing = self.encoder.encode_df(df_testing, False)
		elif self.operation_type == "VALIDATION":
			self.df_testing = self.__get_dataframe__(self.path_validation_file, training=True, limit_seconds=observed_time)
			self.df_testing = self.encoder.encode_df(df_testing, True)
		self.__align_number_columns__()

	def __align_number_columns__(self):
		if self.operation_type == "TESTING":			
			columns_to_add = (set(self.df_training.columns) - set(self.df_testing.columns))-set(["id_player"])
			print(columns_to_add)
			for column in columns_to_add: 
			    self.df_testing[column] = np.nan
		elif self.operation_type == "VALIDATION":
		    training_nb_columns = len(self.df_training.columns)-1
		    testing_nb_columns = len(self.df_testing.columns)
		    if testing_nb_columns>training_nb_columns:
		        columns_to_add = set(self.df_testing.columns) - set(self.df_training.columns)
		        print(columns_to_add)
		        for column in columns_to_add: 
		            self.df_training[column] = np.nan
		    else:
		    	columns_to_add = (set(self.df_training.columns) - set(self.df_testing.columns))-set(["id_player"])
		        print(columns_to_add)
		        for column in columns_to_add: 
		            self.df_testing[column] = np.nan
		    
		self.df_training = self.df_training.fillna(-1)
		self.df_testing = self.df_testing.fillna(-1)
		
	def __str__(self):
		if self.df_training:
			print("TRAINING DATASET")
			print(self.df_training.head())
		if self.df_testing:
			print(self.operation_type + " DATASET")
			print(self.df_testing.head())


if __name__ == '__main__':
	operation_type = "VALIDATION"
	datasets = Transform_Data("./input/minitrain.CSV", operation_type, "./input/minitest.CSV", None)
	datasets.process_data()
	print(datasets)

	decision_tree = RandomForestClassifier(n_estimators = 100, max_depth = 500)
	decision_tree.fit(datasets.df_training.iloc[:, 1:].values, datasets.df_training.id_player.values)

	if operation_type == "VALIDATION":
		df_validation_without_labels = datasets.df_testing.drop(axis=1, labels="id_player")
		predicted = decision_tree.predict(df_validation_without_labels.values)
		labels = datasets.df_testing.id_player.values
		print("accuracy:", accuracy_extractor.get_accuracy(labels, predicted))
	elif operation_type == "TESTING":
		predicted = encoder.decode_labels(decision_tree.predict(datasets.df_testing.values))
		indices = range(1, len(predicted) + 1)
		output_df = pd.DataFrame({"RowId": indices, "prediction": predicted})
		output_df.to_csv("result.CSV", index=False)
