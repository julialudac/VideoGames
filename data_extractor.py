import re
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def __ceil_to5_up__(seconds):
    """Approximates the number of seconds to the upper multiple of 5.
    :param seconds: number of seconds to approximate
    :return: approxilated seconds
    """
    modulo = seconds % 5
    if modulo == 0:
        return seconds
    else:
        return seconds - modulo + 5


def __remove_tX__(myrow):
    """Get a new list of words with words of type tX (X a number) removed
    :param myrow: a list of words"""
    tX = re.compile("t\d")
    newrow = []
    for word in myrow:
        if tX.match(word):
            # print("word", word, "to remove")
            pass
        else:
            newrow.append(word)
    return newrow


def get_dataframe(path_file, training, limit_seconds):
    """
    :param path_file:
    :param training:
    :return: A DataFrame with columns named id_player, played_race, 0... n, with n the number of kept actions.
    """
    limit_seconds = __ceil_to5_up__(limit_seconds)
    extracted = []
    largest_column_count = 0

    # Loop the data lines
    with open(path_file) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\n')
        for row in spamreader:
            myrow = row[0].split(',')
            stop_word = "t" + str(limit_seconds)
            try:
                stop_index = myrow.index(stop_word)
            except:
                stop_index = -1
            myrow = myrow[0:stop_index]
            myrow = __remove_tX__(myrow)
            column_count = len(myrow)
            extracted.append(myrow)
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count
    column_names = []
    if training:
        column_names = ["id_player", "played_race"] + [i for i in range(0, largest_column_count - 2)]
    else:
        column_names = ["played_race"] + [i for i in range(0, largest_column_count - 1)]
    return pd.DataFrame(extracted, columns = column_names)


def transform_sample(df, training):
    """
    BEWARE: Doesn't work on the shuffled data, even reidexing them <=> Only works with a df from a whole csv.
    Not fatal for the result to handle, but quite penalizing to online test.
    :return: a DataFrame resulting from df where for each sample,
    we will have the frequency of actions + played_race (+ id_player).
    Are the values encoded or not? It depends on your goal:
    - not encoded is better to debug this function and have an outlook at what it does
    - encoded allows to move on into the program: training and testing
    """
    df_training_numerical = __subtransform_sample__(0, df, training)
    for index, _ in df.iterrows():
        if index != 0:
            df_sample = __subtransform_sample__(index, df, training)
            df_training_numerical = pd.concat([df_training_numerical, df_sample], sort=False)
    df_training_numerical.fillna(0, inplace=True)
    return df_training_numerical


def __subtransform_sample__(index, df, training):
    column_index_start = 2 if training else 1
    old_row = df.iloc[index,column_index_start:]
    actions, counts = np.unique(old_row.dropna().values, return_counts=True)
    first_part_row = df.iloc[index,0:column_index_start].values
    second_part_row = np.array([count for action, count in zip(actions, counts)])
    row = np.append(first_part_row, second_part_row).reshape(1, -1)
    columns_for_row = ["played_race"] + list(actions)
    if training:
        columns_for_row = ["id_player"] + columns_for_row
    return pd.DataFrame(row, index=[index], columns=columns_for_row)


""" BAD IDEA: Training data must be consistent wrt test/validation data!!!
def conform_transformed_df(df_train, df_test):
    :param df_test: A transformed (ie DataFrame with the counts) test DataFrame
    :param df_train: A transformed train DataFrame
    :return: returns transformed DataFrame whose columns names are intersections of column names of train and test
    test_cols = set(df_test.columns.values)
    train_cols = set(df_train.columns.values)
    kept_cols = test_cols.intersection(train_cols)
    print("kept_cols:",kept_cols)
    conformed_train = df_train[list(kept_cols)]
    conformed_test = df_test[list(kept_cols)]
    return conformed_train, conformed_test
"""


# Difficulty to implement this function: ***** => may have errors
def conform_test_to_training(df_train, df_test, train=True):
    """
    :param df_train: A transformed (ie DataFrame with the counts) train DataFrame
    :param df_test: A transformed (ie DataFrame with the counts) test DataFrame
    :return: returns a transformed DataFrame whose columns names are left join of column names of train and test
    """
    """Let's create a dummy element that has values in the lef join of column names of train and test
    The dummy element must be added to the array composing the test dataset
    """
    test_cols = set(df_test.columns.values)
    #print("test cols:", test_cols, "size:", len(test_cols))
    train_cols = set(df_train.columns.values) # Columns to keep for the conformed test DataFrame
    #print("train_cols:", train_cols, "size:", len(train_cols))
    intersection_cols = test_cols.intersection(train_cols)
    df_test = df_test[list(intersection_cols)]  # We drop the cols that are not in train
    test_array = df_test.values.tolist() # df_test.values is a np-array

    """But to build the conformed DataFrame, we need to have the attributes in the same order as the intersection_cols
    is used to discard the columns of df_test that are not in df_train. So to build the final conformed DataFrame,
    We need a list of attributes composed of the intersection + the attributes on df_train that are not in intersection
    """
    extra_cols = train_cols - intersection_cols
    if not train:
        extra_cols.remove('id_player')

    # The column from train DataFrame that ordered according to te columns of test DataFrame minus id_player if we want
    # to conform a test DataFrame (and not a validation DataFrame)
    testordered_train_cols = list(intersection_cols) + list(extra_cols)

    # To maintain the columns, we append a dummy element that we will remove further
    one_el_list = [-1] * len(testordered_train_cols)
    test_array.append(one_el_list)

    conformed_test = pd.DataFrame(test_array, columns=testordered_train_cols)
    conformed_test = conformed_test[conformed_test['played_race'] != -1] # Condition on whatever existing column
    """ Now, it's also important to reorder the columns of the conformed DataFrame like the train DataFrame 
    to predict correctly (since it uses
    arrays and not DataFrame, we must have same columns order for train and test dataframes.
    """
    train_cols = df_train.columns.values.tolist()
    if not train:
        train_cols.remove('id_player')
    conformed_test = conformed_test[train_cols]

    # print("train columns:", df_test.columns.values)
    # print("test columns:", df_test.columns.values)
    # print("intersection:", intersection_cols)
    # print("extra:", )

    print("We have", len(testordered_train_cols), "features.")

    return conformed_test

