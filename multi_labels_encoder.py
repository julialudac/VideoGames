
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ThreeFeaturesEncoder:
    """
    It contains three encoders to encode the 3 features player_id, played_race and action (a player's
    click or keyboard touch).
    """
    def __init__(self, df_training):
        """ Learning encodings are done here.
        :param df_training: DataFrame whose values are used to encode
        Properties:
        - id_player_unique_values: a Series of all the player's ids (that are strings)
        - played_race_unique_values: a Series of all the possible races
        -
        """

        # Unique values
        self.id_player_unique_values = pd.Series(df_training.id_player.unique())
        self.played_race_unique_values = pd.Series(df_training.played_race.unique())
        self.possible_actions = set() # It is more complex to find that the previous sets
        for column in df_training.columns[2:]:
            for action in df_training.loc[:, column].unique():
                if pd.notna(action):
                    self.possible_actions.add(action)

        # Dictionaries that are mappings from not encoded values to encoded values into integers.
        # So keys are original values and mappings are encoded keys.
        self.id_player_from_categ_to_num = {value: index for value, index in
                                       zip(self.id_player_unique_values, range(0, len(self.id_player_unique_values)))}
        self.played_race_from_categ_to_num = {value: index for value, index in
                                         zip(self.played_race_unique_values, range(0, len(self.played_race_unique_values)))}
        self.possible_actions_from_categ_to_num = {value: index for value, index in
                                      zip(self.possible_actions, range(0, len(self.possible_actions)))}

    def encode_df(self, to_encode):
        """
        :param to_encode: The DataFrame with columns id_player, played_race, 0... n,
        with n the number of kept actions. Values are not encoded
        :return: The same DataFrame with its values encoded
        If this is a test DataFrame (ie no column named 'id_player'), a new column id_player is added with 0's.
        """

        # The dataframe with the encoded ids
        encoded = to_encode.replace(self.id_player_from_categ_to_num)

        # The dataframe with the encoded ids and races
        encoded = encoded.replace(self.played_race_from_categ_to_num)

        # The dataframe with the encoded ids, races and actions
        encoded = encoded.replace(self.possible_actions_from_categ_to_num)

        return encoded

    def decode_labels(self, to_decode):
        """
        :param to_decode: encoded labels to decode
        :return: decoded labels
        """
        decoded = []
        for encoded in to_decode:
            decoded.append(self.id_player_unique_values[encoded])
        return decoded



