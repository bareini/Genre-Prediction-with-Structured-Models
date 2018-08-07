import logging

import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

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

        # functions as words_features_matrix - a dict which saves all possible labels for node
        self.possible_generes_per_node_matrix = {}
        # todo: decide where to initial this
        self.all_tags_list = None

        # functions as build_features_head_modifier
        self.build_features_matrices(df_demo, df_x)
        self.device_indexes_map = df_x.groupby(config.x_device_id).groups

        # todo: set
        self.tags_seen_in_train = []
        self.feature_vector_len = 0
        self.features_position = []


    def build_features_matrices(self, df_demo, df_x):
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
        prev1_node_label = None
        prev2_node_label = None

        training_matrix_rows_index = 0
        training_matrix_columns_index = 0
        training_matrix_rows_index_counter = 0

        for node in x_matrix:

            node_index = df_cols_dict['x'][config.x_row_index]
            node_label = df_cols_dict['x'][config.x_label]

            if device_id != node[df_cols_dict['x'][config.x_device_id]]:
                device_id = node[df_cols_dict['x'][config.x_device_id]]
                relevant_demo = df_demo.query("{}=={}".format(config.demo_device_id, device_id)).values


            # indexes for the training ('truth') matrix
            training_matrix_rows_index_counter = 0
            training_matrix_rows_index = []
            training_matrix_columns_index = []

            # indexes for possible labels matrix
            word_matrix_rows_index_counter = 0
            word_matrix_rows_index = []
            word_matrix_columns_index = []



            for genere_counter, possible_genre in enumerate(self.all_tags_list):

                #
                columns_index = []
                # todo: create a feature class
                # todo: create the feature position builder
                # todo: fearture creater that returns the relevant indexes
                #     current_features = self.features.create_features(modifier, possible_head, tree)
                #     for feature in current_features:
                #         if feature in self.features_position:
                #             word_matrix_rows_index.append(possible_genre.counter)
                #             columns_index.append(self.features_position[feature])
                #
                #     word_matrix_columns_index += columns_index
                #
                #     # True label
                #     if node[node_index] == node[node_label]:
                #         rows_index = np.zeros(shape=len(columns_index), dtype=np.int32)
                #         cols_index = np.asarray(a=columns_index, dtype=np.int32)
                #         data_to_insert = np.ones(len(columns_index), dtype=np.int8)
                #         train_word_vector = csr_matrix((data_to_insert, (rows_index, cols_index)),
                #                                        shape=(1, self.feature_vector_len))
                #         self.training_features_matrix[(sentence_num, modifier.counter)] = train_word_vector
                #
                # rows_index = np.asarray(a=word_matrix_rows_index, dtype=np.int32)
                # cols_index = np.asarray(a=word_matrix_columns_index, dtype=np.int32)
                # data_to_insert = np.ones(len(word_matrix_rows_index), dtype=np.int8)
                # word_matrix = csr_matrix((data_to_insert, (rows_index, cols_index)),
                #                          shape=(len(tree), self.feature_vector_len))


                # used to be self.features.create_features
                current_features = self.potetial_features(node, prev1_node_label, prev2_node_label, possible_genre)

                for feature in current_features:
                    if feature in self.features_position:
                        # the indecator location in the mini matrix - row (same for all) and column (the features from the map)
                        word_matrix_rows_index.append(genere_counter)
                        columns_index.append(self.features_position[feature])

                word_matrix_columns_index += columns_index

                if possible_genre == node_label:

                    temp = list(np.ones(len(columns_index), dtype=np.int32) * training_matrix_rows_index_counter)
                    training_matrix_rows_index += temp
                    training_matrix_columns_index += columns_index
                    training_matrix_rows_index_counter += 1

            rows_index = np.asarray(a=word_matrix_rows_index, dtype=np.int32)
            cols_index = np.asarray(a=word_matrix_columns_index, dtype=np.int32)
            data_to_insert = np.ones(len(word_matrix_rows_index), dtype=np.int8)
            possible_genre_matrix = csr_matrix((data_to_insert, (rows_index, cols_index)),
                                     shape=(len(self.tags_seen_in_train), self.feature_vector_len))

            # Initialize the dict where key : (sentence,word), value : word_matrix
            # original keys - [(device_id, node_index)] - BUT assuming that for each view we have a unigue index

            self.possible_generes_per_node_matrix[ node_index] = possible_genre_matrix

            prev2_node_label = prev1_node_label
            prev1_node_label = node_label

        rows_index = np.asarray(a=training_matrix_rows_index, dtype=np.int32)
        cols_index = np.asarray(a=training_matrix_columns_index, dtype=np.int32)
        data_to_insert = np.ones(len(training_matrix_rows_index), dtype=np.int8)
        self.train_feature_matrix = csr_matrix((data_to_insert, (rows_index, cols_index)),
                                              shape=(training_matrix_rows_index_counter, self.feature_vector_len))

        logger.info("create_features_vector_for_train <-------------- ")

        logger.debug("DependencyParsing: build_features_head_modifier <--------------")


    def potetial_features( self, node, prev1_node_label, prev2_node_label, target_genere):
        """

        :param prev1_node_label: genre of the previous view - string
        :param prev2_node_label:  genre of the 2-previous view - string
        :param target_genere:  the genere we are considering
        :return: a list of string which represents the potential - features for verifying weather its
        """
        # todo: take feature formats from dafna's code

    def feature_vec_builder(self, feature_list):

        """
        :param list(string) feature_list: potential features names
        :return:
        """
        feature_vec = np.zeros(len(self.features_position))

        for feature in feature_list:
            if feature in self.features_position:
                feature_vec[self.features_position[feature]] = 1

        return feature_vec



if __name__ == '__main__':
    logger = logging.getLogger("dependency_parsing")
