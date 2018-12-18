import numpy as np
import sklearn as sk
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

# personal imports
import data_extractor as de
from MultiColumnLabelEncoder import MultiColumnLabelEncoder


class Train_n_test:
    """Train and test... and predict. How to use:
    1/ Train
    2/ Test on a test data (known labels) or/and estimate on a data without telling the labels"""

    def __init__(self):
        """Initialize the attributes (that are later filled) to None
        TODO: remove the unnecessary if needed. The 1/ Not needed if. Will be done if the attribute is only used in one method and not outside"""
        self.train_df = None
        self.train_labels = None
        self.test_df = None
        self.test_labels = None
        self.label_encoder = None
        self.features_encoder = None
        self.estimator = None
        self.predicted = None # encoded predictions

    def train(self, train_df, train_labels, estimator):
        """
        :param train_df: (Not yet encoded) train DataFrame
        :param train_labels: list of the train labels
        :param estimator: an initialized instance of sklearn estimator
        :return:
        """

        # 1/ Get the training dataset
        self.train_df = train_df.copy()
        self.train_labels = train_labels.copy()

        # 2/ Learn an encoding of the dataset with the values from the training set as reference
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.train_labels)  # The encoder learns the encoding
        self.features_encoder = MultiColumnLabelEncoder()
        self.features_encoder.fit(self.train_df)  # The encoder learns the encoding

        # 3/ Encode the training dataset: encode labels and features
        # Since we don't need the original-not-encoded labels and features, we can overwrite them
        # (for the labels, the original values are already learnt in the encoder so it's Ok)
        self.train_labels = self.label_encoder.transform(self.train_labels)
        self.train_df = self.features_encoder.transform(self.train_df)

        # 4/ Train/Learn the classifier on the encoded train dataset
        self.estimator = estimator
        self.estimator.fit(self.train_df.values, self.train_labels)

    def test(self, test_df, test_labels):
        """Test or validate, same => displays the score of the learnt classifier on a test data
        :param test_df: Not yet encoded) test DataFrame
        :param test_labels: list of the test labels
        :return:
        """

        # 1/ Get the test dataset
        self.test_df = test_df.copy()
        self.test_labels = test_labels.copy()

        # 2/ Encode the test dataset: encode labels and features
        # Again, we can overwrite the original data with encoded data
        self.test_labels = self.label_encoder.transform(self.test_labels)
        self.test_df = self.features_encoder.transform(self.test_df)

        # 3/ Test the classifier on the encoded train dataset
        self.predicted = self.estimator.predict(self.test_df.values)  # encoded labels

        # 4/ Display results in the naive way: correct / total
        comparison_list = self.predicted == self.test_labels
        print("Accuracy:", np.sum(comparison_list), "/", len(comparison_list),
              "that means", np.sum(comparison_list)/len(comparison_list)*100, "%")

    def estimate(self, to_estimate, saved_file=None):
        """Estimate labels on a data whose labels may not be known (usually the case)
        Either displays the estimated classes or save them to a csv file. These
        estimation are in their decoded form.
        :param to_estimate: a DataFrame of features whose labels are to be estimated
        :return:
        """

        # Get and encode to_estimate
        encoded_to_estimate = self.features_encoder.transform(to_estimate)

        # Get encoded predictions
        predicted = self.estimator.predict(encoded_to_estimate.values)

        # Decode predictions and build DataFrame
        predicted = self.label_encoder.inverse_transform(predicted)
        indices = range(1, len(predicted) + 1)
        output_df = DataFrame({"RowId": indices, "prediction": predicted})

        # Save/Display
        if saved_file:
            output_df.to_csv(saved_file, index=False)
        else:
            print("Decoded predictions:\n", output_df)