import pandas as pd

class Encoder_Categorical_Numerical:
    """
    It contains three encoders to encode the 3 features player_id, played_race and action (a player's
    click or keyboard touch).
    """
    def __init__(self, df_training, index_beginning_actions, index_end_actions):
        """ Learning encodings are done here.
        :param df_training: DataFrame whose values are used to encode
        Properties:
        - id_player_unique_values: a Series of all the player's ids (that are strings)
        - played_race_unique_values: a Series of all the possible races
        -
        """
        self.index_beginning_actions = index_beginning_actions
        self.index_end_actions = index_end_actions
        # Unique values
        self.id_player_unique_values = pd.Series(df_training.id_player.unique())
        self.played_race_unique_values = pd.Series(df_training.played_race.unique())
        self.possible_actions = set() # It is more complex to find that the previous sets
        for column in df_training.columns[self.index_beginning_actions:self.index_end_actions]:
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

    def encode_df(self, to_encode, train=True):
        """
        :param to_encode: The DataFrame with columns id_player, played_race, 0... n,
        with n the number of kept actions. Values are not encoded
        :param train: If the DataFrame has column "id_player_from_categ_to_num".
        :return: The same DataFrame with its values encoded
        """
        index_beginning_actions = self.index_beginning_actions if train else self.index_beginning_actions-1
        index_end_actions = self.index_end_actions if train else self.index_end_actions-1

        encoded = to_encode
        # encode the race
        encoded.loc[:, "played_race"] = encoded.loc[:, "played_race"].replace(self.played_race_from_categ_to_num)
        # encode actions
        encoded.iloc[:, index_beginning_actions:index_end_actions] = encoded.iloc[:, index_beginning_actions:index_end_actions].replace(self.possible_actions_from_categ_to_num)
        #encode ids
        if train:
            encoded = encoded.replace(self.id_player_from_categ_to_num)

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



