import logging

import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np


from collections import Counter
import config

class Model:
    """
    the model builder
    """

    def __init__(self, df_demo, df_x):
        """

        :param pd.DataFrame df_demo: the demographic data
        :param pd.DataFrame df_x: the actual sequences
        """
        self.type = 'MEMM'
        self.train_feature_matrix = None
        self.train_feature_all_posible_labels = None
        # todo: decide where to initial this
        self.all_tags_list = None
        self.build_features_head_modifier(df_demo, df_x)
        self.device_indexes_map = df_x.groupby(config.x_device_id).groups

    def build_features_head_modifier(self, df_demo, df_x):
        """

        :param df_demo:
        :param df_x:
        :return:
        """
        # todo: make sure that df_x is sorted by device -> time
        logger.debug("DependencyParsing: build_features_head_modifier -------------->")

        df_cols_dict = {'x': {name_: id_ for id_, name_ in enumerate(df_x.columns)}}
        df_cols_dict.update({'demo': {name_: id_ for id_, name_ in enumerate(df_demo.columns)}})

        x_matrix = df_x.as_matrix()

        relevant_demo = None
        device_id = None
        for node in x_matrix:
            node_index = df_cols_dict['x'][config.x_row_index]
            node_label = df_cols_dict['x'][config.x_label]

            if device_id != node[df_cols_dict['x'][config.x_device_id]]:
                device_id = node[df_cols_dict['x'][config.x_device_id]]
                relevant_demo = df_demo.query("{}=={}".format(config.demo_device_id, device_id)).values

            word_matrix_rows_index = []
            word_matrix_columns_index = []

            for possible_genre in self.all_tags_list:

                columns_index = []
                # todo: create a feature class
                # todo: create the feature position builder
                current_features = self.features.create_features(modifier, possible_head, tree)
                for feature in current_features:
                    if feature in self.features_position:
                        word_matrix_rows_index.append(possible_genre.counter)
                        columns_index.append(self.features_position[feature])

                word_matrix_columns_index += columns_index

                # True label
                if node[node_index] == node[node_label]:
                    rows_index = np.zeros(shape=len(columns_index), dtype=np.int32)
                    cols_index = np.asarray(a=columns_index, dtype=np.int32)
                    data_to_insert = np.ones(len(columns_index), dtype=np.int8)
                    train_word_vector = csr_matrix((data_to_insert, (rows_index, cols_index)),
                                                   shape=(1, self.feature_vector_len))
                    self.training_features_matrix[(sentence_num, modifier.counter)] = train_word_vector

            rows_index = np.asarray(a=word_matrix_rows_index, dtype=np.int32)
            cols_index = np.asarray(a=word_matrix_columns_index, dtype=np.int32)
            data_to_insert = np.ones(len(word_matrix_rows_index), dtype=np.int8)
            word_matrix = csr_matrix((data_to_insert, (rows_index, cols_index)),
                                     shape=(len(tree), self.feature_vector_len))

            self.words_features_matrix[(sentence_num, modifier.counter)] = word_matrix

        logger.debug("DependencyParsing: build_features_head_modifier <--------------")



    def feature_from_demo(house_dev_dict, df_demographic):
        """
        :param house_dev_dict dictionary devices for every house
               df_demographic
        :return feature_demo dictionary the names of demographic feature
        """
        # dictionary: count devices for every house
        dev_per_house = {}
        for key in house_dev_dict:
            dev_per_house[key] = len(house_dev_dict[key])

        # replace every 1 in row for the number of device
        for key, value in dev_per_house.items():
            df_demographic.loc[[key]] = df_demographic.loc[[key]].replace(1, value)

        # create dictionary for feature and how many device with this feature
        l = df_demographic[df_demographic.columns].sum(axis=0).to_dict()
        feature_demo = [key for (key, val) in l.items() if val > config.min_amount_demo]
        feature_demo = {key: idx for idx, key in enumerate(feature_demo)}

        return feature_demo

    def feature_builder(self, feature_list):
        """

        :param list(int) feature_list:
        :return:
        """
        pass


if __name__ == '__main__':
    logger = logging.getLogger("dependency_parsing")
