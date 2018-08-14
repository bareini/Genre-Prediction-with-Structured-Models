import os
import logging
import numpy as np
from collections import defaultdict
import config
from scipy.sparse import csr_matrix
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import accuracy_score, confusion_matrix as conf_matrix
import logging

logger = logging.getLogger("viterbi")


class Viterbi:

    def __init__(self, model, weights_vec):
        self.weights = weights_vec
        self.model = model

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
                    features_positions = self.model.test_node_positions(device_id, v, u, t )
                    features_positions.extend(demo_features)

                    probability_table[(t, u, v)] = np.exp(sum(weights[features_positions]))

                # Constant Denominator
                denominator = np.sum(probability_table[(t, u, v)] for v in Sk)
                for v in Sk:
                    probability_table[(t, u, v)] /= denominator

        return probability_table

    def viterbi_algorithm(self, sequence ):
        """
        :type sentence: list of words
        """

        tags_seen_in_train = self.model.all_tags_list
        tags_seen_in_station = self.model.tags_seen_in_station

        pie = {}
        bp = {}

        # logger.info("viterbi_algorithm --------------> ")
        # Base Case
        pie[(0, "*", "*")] = 1.0

        for k in range(1, len(sequence)):

            Sk = tags_seen_in_train
            Sk_1 = tags_seen_in_train
            Sk_2 = tags_seen_in_train

            current_station = self.model.test_df.loc[sequence[k], config.station_num]

            # if the word appeared in the training data with tags we assign this tags to S
            if current_station in tags_seen_in_station:
                Sk = tags_seen_in_station[current_station]

            if k == 1:
                Sk_2, Sk_1 = ["*"], ["*"]
                prev1_sation = self.model.test_df.loc[sequence[k - 2], config.station_num]
            elif prev1_sation in tags_seen_in_station:
                Sk_1 = tags_seen_in_station[prev1_sation]

            if k == 2:
                Sk_2 = ["*"]
                prev2_sation = self.model.test_df.loc[sequence[k - 3]]
            elif k > 2 and prev2_sation in tags_seen_in_station:
                Sk_2 = tags_seen_in_station[prev2_sation]

            probability_table = self.probability(Sk_2, Sk_1, Sk, sequence[k], k)

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

        t = {}
        n = len(sequence)
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
        for i in t:
            genre_sequence.append(t[i])

        if n == 1:
            genre_sequence = [genre_sequence[n]]

        logger.info("viterbi_algorithm <-------------- ")
        return genre_sequence