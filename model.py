import logging
import time
from itertools import count
from collections import defaultdict
from typing import List, Any

import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import config

logger = logging.getLogger("dependency_parsing")


class Model:
    """
    the model builder
    """
    def __init__(self, df_demo, df_x, house_device, device_house, model_type, test_df=None):
        """


        :param house_device:
        :param dict device_house:
        :param pd.DataFrame df_demo: the demographic data
        :param pd.DataFrame df_x: the actual sequences
        """

        self.type = 'MEMM'
        self.train_feature_matrix = None
        self.df_cols_dict = None
        self.df_x = df_x
        self.df_demo = df_demo
        self.test_df = test_df
        self.model_type = model_type

        self.house_device = house_device
        self.device_house = device_house

        # functions as words_features_matrix - a dict which saves all possible labels for node
        self.possible_genres_per_node_matrix = {}
        # todo: decide where to initial this
        self.all_tags_list = None

        self.device_indexes_map = df_x.groupby(config.x_device_id).groups

        # todo: set
        self.tags_seen_in_train = []
        self.feature_vector_len = 0
        self.atomic_tags = set()
        self.features_position = {}
        self.tags_seen_in_station = defaultdict(set)
        self.true_genres = df_x[config.x_program_genre].tolist()
        self.test_true_genres = test_df[config.x_program_genre].tolist()
        self.tags_seen_in_part_of_day = defaultdict(list)

        self.feature_position_counter = count()
        self.prec_positions = []  # cols which are valid also for preceptron
        self.init_features()
        # functions as build_features_head_modifier
        self.build_features_matrices()
        self.dict_nodes_per_device = None
        self.notes_per_device()
        self.test_feature_matrix = None

    def init_features(self):
        """
        this function initializes the self.features_position attribute, by getting all the possible
        values in each defined feature, and if the existence of it is bigger than the threshold (in config.threshold)
        it being saved to feature_position attribute, with a unique position

        :return None:
        """
        self.feature_from_demo()  # run the demographic feature init
        for dict_cols in config.All_cols:
            for col, (action, prefix) in dict_cols.items():
                if action == 'counter':
                    # todo: take it out to function
                    temp_df = self.df_x[col].value_counts()
                    self.features_position.update({"{}_{}".format(prefix, feature_val): next(self.feature_position_counter)
                                                   for feature_val in temp_df[temp_df > config.thresholds[col]].index})
                    if col == config.x_program_genre:
                        self.all_tags_list = self.df_x[col].unique().tolist()
                        for genres in self.df_x[col].unique():
                            if type(genres) == str:
                                self.atomic_tags.update(set(genres.split(',')))
                            else:
                                self.atomic_tags.add(genres)
                elif action == 'unique':
                    self.features_position.update(
                        {"{}_{}".format(prefix, feature_val): next(self.feature_position_counter)
                         for feature_val in self.df_x[col].astype(str).unique()})
                elif action == 'interact':
                    col_1, col_2 = col
                    temp_df = self.df_x.groupby([col_1, col_2], as_index=True).size()
                    # if prefix == config.part_of_day_genre:
                    #     temp_df = temp_df.reset_index()
                    #     for col1, col2, val in temp_df.values:
                    #         self.tags_seen_in_part_of_day[col1].append(col2)
                    #     continue
                    temp_df = temp_df[temp_df > config.thresholds[col]].reset_index()
                    self.features_position.update(
                        {"{}_{}_{}".format(prefix, col1, col2): next(self.feature_position_counter)
                         for col1, col2, val in temp_df.values})

                elif action == 'double_interact':
                    col_1, col_2, col_3 = col
                    temp_df = self.df_x.groupby([col_1, col_2, col_3], as_index=True).size()  # .reset_index()
                    temp_df = temp_df[temp_df > config.thresholds[col]].reset_index()
                    self.features_position.update(
                        {"{}_{}_{}_{}".format(prefix, col1, col2, col3): next(self.feature_position_counter)
                         for col1, col2, col3, val in temp_df.values})
                    if prefix == config.station_time_genre:
                        temp_df = temp_df.reset_index()
                        for col0, col1, col2, col3, col4 in temp_df.values:
                            self.tags_seen_in_station[col1, col2].update({col3})
                        continue

        self.prec_positions = list(set(self.prec_positions))
        for key in self.features_position.keys():
                if key[:2] not in config.genre_prefixes:
                    self.prec_positions.append(self.features_position[key])
        self.feature_vector_len = next(self.feature_position_counter)
        self.calc_top_k_seen_in_part_of_day()

    def build_features_matrices(self):
        """
        Construct train_feature_matrix - a training matrix of actual genere sequences as they appear in the data
        and mini matrices for per node of all possible genre - possible_genres_per_node_matrix

        :return:
        """
        # todo: make sure that df_x is sorted by device -> time
        logger.debug("DependencyParsing: build_features_head_modifier -------------->")
        # todo: question everything
        self.df_cols_dict = {'x': {name_: id_ for id_, name_ in enumerate(self.df_x.columns)}}
        self.df_cols_dict.update({'demo': {name_: id_ for id_, name_ in enumerate(self.df_demo.columns)}})

        x_matrix = self.df_x.values  # type: np.matrix

        relevant_demo_id = None
        demo_features = None
        relevant_demo = None
        device_id = None
        prev1_node_label = None
        prev2_node_label = None
        demo_feature_count = 0

        # indexes for the training ('truth') matrix
        training_matrix_rows_index_counter = 0
        training_matrix_rows_index = []
        training_matrix_columns_index = []

        for idx in range(x_matrix.shape[0]):

            node = x_matrix[idx]
            node_index = node[self.df_cols_dict['x'][config.x_row_index]]
            node_label = node[self.df_cols_dict['x'][config.x_program_genre]]

            if device_id != node[self.df_cols_dict['x'][config.x_device_id]]:
                device_id = node[self.df_cols_dict['x'][config.x_device_id]]

                if device_id is not None:
                    relevant_demo_id = self.device_house[device_id]
                    demo_features, demo_feature_count = self.demo_positions(relevant_demo_id)
                    non_gernre_features, non_genre_feature_count = self.node_position_no_genre(node_index)
                    device_features = demo_features + non_gernre_features


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

                # used to be self.features.create_features
                current_features, ones_counter = self.node_positions_genre(node_index, possible_genre)
                current_features.extend(device_features)
                ones_counter += demo_feature_count + non_genre_feature_count

                # for feature in current_features:
                #     if feature in self.features_position:
                #         # the indecator location in the mini matrix - row (same for all) and column (the features from the map)
                #         word_matrix_rows_index.append(genere_counter)
                #         columns_index.append(self.features_position[feature])

                additional_row_index = [genere_counter] * ones_counter
                word_matrix_rows_index.extend(additional_row_index)
                columns_index.extend(current_features)

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
                                               shape=(len(self.all_tags_list), self.feature_vector_len))

            # Initialize the dict where key : (sentence,word), value : word_matrix
            # original keys - [(device_id, node_index)] - BUT assuming that for each view we have a unigue index

            self.possible_genres_per_node_matrix[node_index] = possible_genre_matrix

            prev2_node_label = prev1_node_label
            prev1_node_label = node_label

        rows_index = np.asarray(a=training_matrix_rows_index, dtype=np.int32)
        cols_index = np.asarray(a=training_matrix_columns_index, dtype=np.int32)
        data_to_insert = np.ones(len(training_matrix_rows_index), dtype=np.int8)
        self.train_feature_matrix = csr_matrix((data_to_insert, (rows_index, cols_index)),
                                               shape=(training_matrix_rows_index_counter, self.feature_vector_len))

        logger.info("create_features_vector_for_train <-------------- ")

        logger.debug("DependencyParsing: build_features_head_modifier <--------------")

    def node_position_no_genre(self,device_id):
        """
        extract all potential features for a specific node before selection

        :param node the line from the dfs which describes a specific view

        :param prev1_node_label: genre of the previous view - string
        :param prev2_node_label:  genre of the 2-previous view - string
        :param target_genere:  the genere we are considering
        :return: a list of string which represents the potential - features for verifying weather its
        """

        if self.model_type == "Advanced":
            config.col_action.update(config.advanced_household32)
            config.genere_cols.update(config.advanced_household32)

        node = self.df_x.loc[device_id]
        feature_vector_positions = []

        for col, (action, prefix) in config.col_action.items():

            if col in config.genere_cols:
                continue
            if action == 'unique' or action == 'counter':
                name = "{}_{}".format(prefix, node[col])

            elif action == 'interact':
                col_1, col_2 = col
                name = "{}_{}_{}".format(prefix, node[col_1], node[col_2])

            elif action == 'double_interact':
                col_1, col_2, col_3 = col
                name = "{}_{}_{}_{}".format(prefix, node[col_1], node[col_2], node[col_3])

            if name in self.features_position:
                feature_vector_positions.append(self.features_position[name])

        return feature_vector_positions, len(feature_vector_positions)


    def node_positions_genre(self, device_id, target_genere):
        """
        extract all potential features for a specific node before selection

        :param node the line from the dfs which describes a specific view

        :param prev1_node_label: genre of the previous view - string
        :param prev2_node_label:  genre of the 2-previous view - string
        :param target_genere:  the genere we are considering
        :return: a list of string which represents the potential - features for verifying weather its
        """

        node = self.df_x.loc[device_id]
        feature_vector_positions = []

        if self.model_type != 'perceptron':

            if self.model_type == 'advanced2':
                config.col_action.update(config.advanced_pattern2)
                config.genere_cols.update(config.advanced_pattern2)
            elif self.model_type == "advanced":
                config.col_action.update(config.advanced_household32)
                config.genere_cols.update(config.advanced_household32)
            if self.model_type == "creative":
                config.col_action.update(config.cluster_cols)
                config.genere_cols.update(config.cluster_cols)

            for col in config.genere_cols:
                action, prefix = config.col_action[col]

                # can be only 'Program Genre'
                if action == 'unique' or action == 'counter':
                    name = "{}_{}".format(prefix, target_genere)

                elif action == 'interact':
                    col_1, col_2 = col
                    name = "{}_{}_{}".format(prefix, node[col_1], target_genere)

                elif action == 'double_interact':
                    col_1, col_2, col_3 = col
                    name = "{}_{}_{}_{}".format(prefix, node[col_1], node[col_2], target_genere)

                if name in self.features_position:
                    feature_vector_positions.append(self.features_position[name])

        return feature_vector_positions, len(feature_vector_positions)

    def create_test_matrix(self):
        """
        temporary method, to create the test feature matrix, for the use of the perceptron
        will be replaced by build_features_matrices

        :return: None
        """
        # todo: embed in the build_features_matrices
        x_matrix = self.test_df.values  # type: np.matrix

        # todo: make sure this is changed back to MEMM!!
        temp_type = self.model_type
        self.model_type = 'perceptron'

        relevant_demo_id = None
        demo_features = None
        device_id = None

        # indexes for the training ('truth') matrix
        test_matrix_rows_index_counter = 0
        test_matrix_rows_index = []
        test_matrix_columns_index = []

        for idx in range(x_matrix.shape[0]):

            node = x_matrix[idx]
            node_index = node[self.df_cols_dict['x'][config.x_row_index]]

            if device_id != node[self.df_cols_dict['x'][config.x_device_id]]:
                device_id = node[self.df_cols_dict['x'][config.x_device_id]]

                if device_id is not None:
                    relevant_demo_id = self.device_house[device_id]
                    demo_features, demo_feature_count = self.demo_positions(relevant_demo_id)

            columns_index = []

            current_features, _ = self.test_node_positions(node_index, None, None, None)
            current_features.extend(demo_features)

            columns_index.extend(current_features)

            temp = list(np.ones(len(columns_index), dtype=np.int32) * test_matrix_rows_index_counter)
            test_matrix_rows_index.extend(temp)
            test_matrix_columns_index.extend(columns_index)
            test_matrix_rows_index_counter += 1

        rows_index = np.asarray(a=test_matrix_rows_index, dtype=np.int32)
        cols_index = np.asarray(a=test_matrix_columns_index, dtype=np.int32)
        data_to_insert = np.ones(len(test_matrix_rows_index), dtype=np.int8)
        self.test_feature_matrix = csr_matrix((data_to_insert, (rows_index, cols_index)),
                                              shape=(test_matrix_rows_index_counter, len(self.prec_positions)))
        self.model_type = temp_type

    def test_node_positions(self, device_id, target_genere, prev1_genre, prev2_genre):
        """
        extract all potential features for a specific node before selection for test!

        :param node the line from the dfs which describes a specific view

        :param prev1_node_label: genre of the previous view - string
        :param prev2_node_label:  genre of the 2-previous view - string
        :param target_genere:  the genere we are considering
        :return: a list of string which represents the potential - features for verifying weather its
        """
        if self.model_type == 'advanced2':
            config.col_action.update(config.advanced_pattern2)
            config.genere_cols.update(config.advanced_pattern2)
        elif self.model_type == "advanced":
            config.col_action.update(config.advanced_household32)
            config.genere_cols.update(config.advanced_household32)
        if self.model_type == "creative":
            config.col_action.update(config.cluster_cols)
            config.genere_cols.update(config.cluster_cols)

        node = self.test_df.loc[device_id]
        feature_vector_positions = []

        for col, (action, prefix) in config.col_action.items():

            if col in config.genere_cols:
                continue
            if action == 'unique' or action == 'counter':
                name = "{}_{}".format(prefix, node[col])

            elif action == 'interact':
                col_1, col_2 = col
                name = "{}_{}_{}".format(prefix, node[col_1], node[col_2])

            elif action == 'double_interact':
                col_1, col_2, col_3 = col
                name = "{}_{}_{}_{}".format(prefix, node[col_1], node[col_2], node[col_3])

            if name in self.features_position:
                feature_vector_positions.append(self.features_position[name])
        if self.model_type != 'perceptron':
            for col in config.genere_cols:
                action, prefix = config.col_action[col]

                # can be only 'Program Genre'
                if action == 'unique' or action == 'counter':
                    name = "{}_{}".format(prefix, target_genere)

                elif action == 'interact':
                    col_1, col_2 = col
                    if prefix == 'g1':
                        name = "{}_{}_{}".format(prefix, prev1_genre, target_genere)
                    else:
                        name = "{}_{}_{}".format(prefix, node[col_1], target_genere)

                elif action == 'double_interact':
                    col_1, col_2, col_3 = col
                    if prefix == 'g2':
                        name = "{}_{}_{}".format(prefix, prev2_genre, prev1_genre, target_genere)
                    else:
                        name = "{}_{}_{}_{}".format(prefix, node[col_1], node[col_2], target_genere)

                if name in self.features_position:
                    feature_vector_positions.append(self.features_position[name])

        return feature_vector_positions, len(feature_vector_positions)

    def demo_positions(self, relevant_demo_id):

        demo = self.df_demo.loc[relevant_demo_id].values.flatten()
        feature_vector_positions = []

        for demo_col in self.df_cols_dict['demo'].keys():
            if demo_col == config.household_id or demo_col == config.voter:
                continue
            name = 'd_{}'.format(demo_col)
            if demo[self.df_cols_dict['demo'][demo_col]] == 1 and name in self.features_position:
                feature_vector_positions.append(self.features_position[name])

        return feature_vector_positions, len(feature_vector_positions)

    def feature_vec_builder(self, feature_list):

        """
        NOT SURE IF NEEDED - creates a feature vector from a feature list
        :param list(string) feature_list: potential features names
        :return:
        """
        feature_vec = np.zeros(len(self.features_position))

        for feature in feature_list:
            if feature in self.features_position:
                feature_vec[self.features_position[feature]] = 1

        return feature_vec

    def feature_from_demo(self):
        """


        :param house_dev_dict: dictionary devices for every house
               df_demographic
        :return
        """
        # dictionary: count devices for every house
        self.df_demo = self.df_demo.set_index(config.household_id)
        arr_dev_per_house = np.zeros(shape=(self.df_demo.shape[0], 1))
        i = 0
        for key in self.house_device:
            if self.df_demo.index[i] == key:
                arr_dev_per_house[i] = int(len(self.house_device[key]))
                i += 1

        # replace every 1 in row for the number of device
        # for key, value in dev_per_house.items():
        #     self.df_demo.loc[[key]] = self.df_demo.loc[[key]].replace(1, value)
        temp_matrix = np.multiply(self.df_demo.values, arr_dev_per_house)
        # create dictionary for feature and how many device with this feature
        # temp_matrix = self.df_demo.drop(columns=config.household_id).as_matrix()
        temp_matrix = temp_matrix.sum(axis=0)
        self.features_position.update({'d_{}'.format(coll): next(self.feature_position_counter)
                                       for coll in
                                       self.df_demo.columns[np.where(temp_matrix > config.min_amount_demo)[0]]})

    def notes_per_device(self):
        """

        :return: dict the notes for every device.
        """
        self.dict_nodes_per_device = self.test_df.groupby(['Device ID'])['df_id'].apply(list).to_dict()

    def calc_top_k_seen_in_part_of_day(self):
        """
        updates tags seen in part of day

        :return: nothing
        """
        k = config.num_of_top_k # the top k seen in part of day
        df = self.df_x[[config.part_of_day, config.x_program_genre]]
        b = df.groupby([config.part_of_day])[config.x_program_genre].value_counts()
        c = b.groupby([config.part_of_day]).head(k).rename(columns={config.x_program_genre: 'a'}).reset_index().drop(columns=0)
        d = c.groupby([config.part_of_day], as_index=True)[config.x_program_genre].apply(sorted).apply(tuple).reset_index()
        self.tags_seen_in_part_of_day = d.set_index(config.part_of_day).to_dict()[config.x_program_genre]

    def propagate_predicted_cluster(self, predicted):
        """

        :return:
        """
        logger.info("{}: propagate predicted cluster".format(time.asctime(time.localtime(time.time()))))
        logging.info('{}: before clusters viterbi'.format(time.asctime(time.localtime(time.time()))))
        reversed_clustered = {}

        # df_program_viewing['program_genre_clustered'] = df_program_viewing['Program Genre'].map(revresed_clustered)

        # df_program_viewing['prev_1_genre_clustered'] = df_program_viewing.groupby('Device ID')[
        #     'program_genre_clustered'].shift(1).fillna('*')
        # df_program_viewing['prev_2_genre_clustered'] = df_program_viewing.groupby('Device ID')[
        #     'program_genre_clustered'].shift(2).fillna('**')
        # df_program_viewing.loc[(df_program_viewing['prev_2_genre_clustered'] == '**')
        #                        & (df_program_viewing['prev_1_genre_clustered'] != '*'), 'prev_2_genre'] = '*'
        # df_program_viewing['gen_in_dev_hh_1_clustered'] = df_program_viewing['gen_in_dev_hh_1'].map(
        #     revresed_clustered).fillna(-1)
        # df_program_viewing['gen_in_advance_1_clustered'] = df_program_viewing['gen_in_advance_1'].map(
        #     revresed_clustered).fillna(-1)


if __name__ == '__main__':
    logger = logging.getLogger("dependency_parsing")
