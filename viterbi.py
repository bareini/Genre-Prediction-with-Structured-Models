import os
import logging
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import accuracy_score, confusion_matrix as conf_matrix


def find_feature_vector_test_set(self, sentence, k, t, u, v):
    """
    Find all the places in feature vector where the entries are 1
    :param sentence: list of word
    :param k: current position
    :param t: tag in position k-2
    :param u: tag in position k-1
    :param v: tag in position k
    :return:
    """

    position_in_vector = []
    origin_word = sentence[k - 1]
    word = origin_word.lower()

    # word x tag
    pair = word + "_" + v + "_100"
    if pair in self.features_position:
        position_in_vector.append(self.features_position[pair])
    # tag trigram
    trigram = t + "_" + u + "_" + v + "_103"
    if trigram in self.features_position:
        position_in_vector.append(self.features_position[trigram])
    # tag bigram
    bigram = u + "_" + v + "_104"
    if bigram in self.features_position:
        position_in_vector.append(self.features_position[bigram])

    if self.model_type == "advanced":
        # tag unigram
        unigram = v + "_105"
        if unigram in self.features_position:
            position_in_vector.append(self.features_position[unigram])

        # prefix <=4 x tag
        prefix_list = [word[:i] + '_' + v + "_101" for i in range(1, 5)]
        prefix_list = list(set(prefix_list))
        for item in prefix_list:
            if item in self.features_position:
                position_in_vector.append(self.features_position[item])

        # suffix <=4 x tag
        suffix_list = [word[i:] + '_' + v + "_102" for i in range(-4, 0)]
        suffix_list = list(set(suffix_list))
        for item in suffix_list:
            if item in self.features_position:
                position_in_vector.append(self.features_position[item])

        # current tag with previous word
        if u != "*":
            previous_word_current_tag = sentence[k - 2] + "_" + v + "_106"
            if previous_word_current_tag in self.features_position:
                position_in_vector.append(self.features_position[previous_word_current_tag])
        # current tag with next word
        if k != (len(sentence)):
            next_word_current_tag = sentence[k] + "_" + v + "_107"
            if next_word_current_tag in self.features_position:
                position_in_vector.append(self.features_position[next_word_current_tag])
        # is the string a number
        if str.isdigit(word):
            position_in_vector.append(self.features_position["is_numeric"])
        # is the string start with upper letter
        if str.isupper(origin_word[0]):
            position_in_vector.append(self.features_position["is_capitalized"])

    return position_in_vector



def probability(self, Sk_2, Sk_1, Sk, sentence, k):
    """
    :param Sk_2:
    :param Sk_1:
    :param Sk:
    :param sentence:
    :param k: the current position
    :return:
    """

    probability_table = defaultdict(tuple)
    weights = self.v

    for t in Sk_2:
        for u in Sk_1:
            for v in Sk:
                probability_table[(t, u, v)] = np.exp(
                    sum(weights[self.find_feature_vector_test_set(sentence, k, t, u, v)]))

            # Constant Denominator
            denominator = np.sum(probability_table[(t, u, v)] for v in Sk)
            for v in Sk:
                probability_table[(t, u, v)] /= denominator

    return probability_table

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