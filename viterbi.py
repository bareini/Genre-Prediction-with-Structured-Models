import os
import logging
import numpy as np
from collections import defaultdict
import config
from scipy.sparse import csr_matrix
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import accuracy_score, confusion_matrix as conf_matrix
import logging
import time
from datetime import datetime


logger = logging.getLogger("viterbi")


class Viterbi:

    def __init__(self, model, weights_vec, perceptron, most_common):
        """

        :param Model model:
        :param weights_vec:
        :param perceptron:
        :param most_common:
        """
        self.weights = weights_vec
        self.model = model
        self.perceptron = perceptron
        self.most_common = most_common

    def probability(self, Sk_2, Sk_1, Sk, device_id, k):
        """
        :param Sk_2:
        :param Sk_1:
        :param Sk:
        :param sentence:
        :param k: the current position
        :return:
        """

        probability_table = defaultdict(tuple)
        weights = self.weights

        # get demo features
        if device_id is not None:
            relevant_demo_id = self.model.device_house[device_id]
            demo_features, demo_feature_count = self.model.demo_positions(relevant_demo_id)

        for t in Sk_2:
            for u in Sk_1:
                for v in Sk:
                    features_positions, _ = self.model.test_node_positions(k, v, u, t )
                    features_positions.extend(demo_features)

                    probability_table[(t, u, v)] = np.exp(sum(weights[features_positions]))

                # Constant Denominator
                denominator = np.sum(probability_table[(t, u, v)] for v in Sk)
                for v in Sk:
                    probability_table[(t, u, v)] /= denominator

        return probability_table

    def viterbi_algorithm(self, sequence):
        """
        :param sequence:
        :type sentence: list of words
        """
        seq_len = len(sequence)

        tags_seen_in_train = self.model.all_tags_list
        tags_seen_in_station = self.model.tags_seen_in_station


        pie = {}
        bp = {}

        # logger.info("viterbi_algorithm --------------> ")
        # Base Case
        pie[(0, "**", "*")] = 1.0
        Sk_2, Sk_1 = ["**"], ["*"]

        for k in range(1, seq_len + 1):

            # Sk = tags_seen_in_train
            # if Sk_1 != ["*"]:
            #     Sk_1 = Sk
            # if Sk_2 != ["**"]:
            #     Sk_2 = Sk_1
            # Sk_1 = tags_seen_in_train

            # Sk_2 = tags_seen_in_train

            # Load household features
            if k == 1:
                device_id = self.model.test_df.loc[sequence[k-1], config.x_device_id]
                relevant_demo_id = self.model.device_house[device_id]
                demo_positions, _ = self.model.demo_positions(relevant_demo_id)

            current_station = self.model.test_df.loc[sequence[k-1], config.station_num]
            current_part_of_day = self.model.test_df.loc[sequence[k-1], config.part_of_day]
            current_index = self.model.test_df.loc[sequence[k-1], 'df_id']

            current_token = tuple([current_part_of_day, current_station])
            # if the word appeared in the training data with tags we assign this tags to S
            # todo: put comments~!!@##@!Q##$^@#$!!!
            if current_token in tags_seen_in_station:
                Sk = tags_seen_in_station[current_part_of_day, current_station]
            else:
                # Sk = self.model.tags_seen_in_part_of_day[current_part_of_day]
                # Sk = self.create_for_preceptron(current_index, demo_positions)
                Sk = self.model.tags_seen_in_part_of_day[current_part_of_day]
                #
                # if len(Sk) > 1:
                #     print('oy!')
            # if k != 1:
            #     prev1_sation = self.model.test_df.loc[sequence[k - 2], config.station_num]
            #     if prev1_sation in tags_seen_in_station:
            #         Sk_1 = tags_seen_in_station[prev1_sation]
            #     else:
            #         # Sk_1 = self.model.tags_seen_in_part_of_day[current_part_of_day]
            #         prev1_id = self.model.test_df.loc[sequence[k - 2], 'df_id']
            #         Sk_1 = self.create_for_preceptron(prev1_id, demo_positions)
            #
            # else:
            #     Sk_2, Sk_1 = ["**"], ["*"]
            #
            #
            # if k > 2:
            #     prev2_sation = self.model.test_df.loc[sequence[k - 3], config.station_num]
            #     if prev2_sation in tags_seen_in_station:
            #         Sk_2 = tags_seen_in_station[prev2_sation]
            #     else:
            #         prev2_id = self.model.test_df.loc[sequence[k - 3], 'df_id']
            #         Sk_1 = self.create_for_preceptron(prev2_id, demo_positions)
            #
            # elif k == 2:
            #     Sk_2 = ["*"]

            probability_table = self.probability(Sk_2, Sk_1, Sk, device_id, sequence[k-1])

            for u in Sk_1:
                for v in Sk:

                    pie_max = 0
                    bp_max = None
                    for t in Sk_2:
                        pie_temp = pie[(k - 1, t, u)] * probability_table[(t, u, v)]

                        if pie_temp > pie_max:
                            pie_max = pie_temp
                            bp_max = t

                    pie[(k, u, v)] = pie_max
                    bp[(k, u, v)] = bp_max
            if k < seq_len:
                Sk_2 = Sk_1
                Sk_1 = Sk

        t = {}
        n = seq_len
        pie_max = 0
        for u in Sk_1:
            for v in Sk:
                curr_pie = pie[(n, u, v)]
                if curr_pie > pie_max:
                    pie_max = curr_pie
                    t[n] = v
                    t[n - 1] = u

        for k in range(n - 2, 0, -1):
            t[k] = bp[k + 2, t[k + 1], t[k + 2]]

        genre_sequence = []
        for i in range(1, len(t)+1):
            try:
                genre_sequence.append(t[i])
            except KeyError as e:
                print("i: {}, seq={}, genre_sequence={}, t[i]={}, n={}".format(i, sequence, genre_sequence, t, n))
        if n == 1:
            genre_sequence = [genre_sequence[n]]

        logging.info('{}: viterbi_algorithm <--------------'.format(time.asctime(time.localtime(time.time()))))
        return genre_sequence

    def create_for_preceptron(self, ind, demo_positions):
        poistions, _ = self.model.test_node_positions(ind, None, None, None)
        feauture_vec = np.zeros(len(self.model.prec_positions))
        poistions.extend(demo_positions)
        np.put(feauture_vec, poistions, [1])
        Sk = [self.perceptron.labels_mapping[self.perceptron.predict(np.matrix(feauture_vec)).item()]]
        return Sk