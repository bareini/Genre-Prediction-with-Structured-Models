import os
import logging
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import accuracy_score, confusion_matrix as conf_matrix



def viterbi_algorithm(self, sequence, label_in_train, ):
    """
    :type sentence: list of words
    """
    pie = {}
    bp = {}

    # logger.info("viterbi_algorithm --------------> ")
    # Base Case
    pie[(0, "*", "*")] = 1.0

    for k in range(1, len(sequence) + 1):

        Sk = tags_seen_in_train
        Sk_1 = tags_seen_in_train
        Sk_2 = tags_seen_in_train

        # if the word appeared in the training data with tags we assign this tags to S
        word = sentence[k - 1].lower()
        if word in self.seen_words_with_tags:
            Sk = self.seen_words_with_tags[word]

        if k == 1:
            Sk_2, Sk_1 = ["*"], ["*"]
        elif sentence[k - 2].lower() in self.seen_words_with_tags:
            Sk_1 = self.seen_words_with_tags[sentence[k - 2].lower()]

        if k == 2:
            Sk_2 = ["*"]
        elif k > 2 and sentence[k - 3].lower() in self.seen_words_with_tags:
            Sk_2 = self.seen_words_with_tags[sentence[k - 3].lower()]

        probability_table = self.probability(Sk_2, Sk_1, Sk, sentence, k)

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
    n = len(sentence)
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

    tag_sequence = []
    for i in t:
        tag_sequence.append(t[i])

    if n == 1:
        tag_sequence = [tag_sequence[n]]

    logger.info("viterbi_algorithm <-------------- ")
    return tag_sequence