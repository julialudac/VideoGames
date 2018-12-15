import pandas as pd
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
    # TODO: for more than t5, we have to remove all the tX!!!
    newrow = []
    for word in myrow:
        # if (word == ) # regex to do
        pass
    return newrow


def __extract_data_till_time_list__(datafile, limit_seconds, data_role = 0, n_features=None):
    """TODO: use remove_tX
    :param datafile: file containing the data in a csv format (with columns separated by commas)
    :param limit_seconds: seconds limit to pick the data
    :param data_role: 0 if it's data from train file used for training, 1 if it's data from train file
    used for testing, 2 if it's data from test file.
    :param n_features: None if data_role = 0.
    Otherwise, indicates the number of features for each element.
    :return a 2d list whose rows are the elements and non-empty labels if n_features=None
    """
    limit_seconds = __ceil_to5_up__(limit_seconds)
    extracted = []
    labels = []
    for line in datafile:
        if line == "\n":
            continue
        myrow = line.split(",")
        if data_role == 0 or data_role == 1:
            labels.append(myrow[0])
        # print(myrow)
        stop_word = "t" + str(limit_seconds)
        #print("my row:", myrow)
        try:
            stop_index = myrow.index(stop_word)
        except:
            stop_index = -1
        myrow = myrow[1:stop_index]
        if data_role > 0:
            if len(myrow) > n_features:
                myrow = myrow[:n_features]
            elif len(myrow) < n_features:
                myrow = myrow + [None]*(n_features - len(myrow))
        extracted.append(myrow)
    return extracted, labels


def extract_data_till_time_df(datafile, limit_seconds, data_role = 0, n_features=None):
    """
    :param datafile: file containing the data in a csv format (with columns separated by commas)
    :param limit_seconds: seconds limit to pick the data
    :param is_training: True if data is from training data csv
    :return a DataFrame whose rows are the elements and non-empty (list of) labels if is_training=True.
    Missing values are replaced by "unk" (Cons: "unk" is the garbage class => large amount of it for each feature)
    """
    extracted, labels =__extract_data_till_time_list__(datafile, limit_seconds, data_role=data_role,
                                                       n_features=n_features)
    extracted_df = pd.DataFrame(extracted)
    extracted_df.fillna(inplace=True, value="unk")
    return extracted_df, labels


