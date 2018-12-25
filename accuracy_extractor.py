import numpy as np


def get_accuracy(labels, predicted_labels):
    compare_result = np.unique(np.equal(labels,predicted_labels), return_counts=True)
    good_predictions = 0
    total_predictions = len(predicted_labels)
    for index, value in enumerate(compare_result[0]):
        if value:
            good_predictions = compare_result[1][index]
    return good_predictions/total_predictions*100


# ???
def get_accuracy_from_categorical(indexes, predicted_labels, id_player_unique_values):
    """???
    :param indexes:
    :param predicted_labels:
    :return:
    """
    name_players_random_forest = id_player_unique_values.iloc[indexes]