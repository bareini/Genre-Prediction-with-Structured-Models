import logging
import os
import numpy as np
import time
from reference.chu_liu import Digraph
import pickle
from scipy.sparse import csr_matrix
from collections import defaultdict
from copy import copy
import math
from scipy.sparse import csr_matrix
from collections import namedtuple, defaultdict, Counter


def STRUCTUREDperceptron(self, num_of_iter):
    """
    this method implements the pseudo-code of the perceptron

    :param num_of_iter: N from the pseudo-code
    :return: the final weight vector
    :rtype: csr_matrix[int]
    """
    for i in range(num_of_iter):
        print('{}: Starting Iteration #{}'.format(time.asctime(time.localtime(time.time())), i + 1))
        logging.info('{}: Starting Iteration #{}'.format(time.asctime(time.localtime(time.time())), i + 1))
        for t in self.gold_tree.keys():
            self.calculate_new_scores(t)
            if t % 100 == 0:
                print('{}: Working on sentence #{}'.format(time.asctime(time.localtime(time.time())), t + 1))
                logging.info('{}: Working on sentence #{}'.format(time.asctime(time.localtime(time.time())), t + 1))
            self.current_sentence = t
            pred_tree = self.calculate_mst(t)
            if not GraphUtil.identical_dependency_tree(pred_tree, self.gold_tree[t]):
                curr_feature_vec = self.features_vector_train[t]
                new_feature_vec = self.model.create_global_feature_vector(pred_tree, t, mode=self._mode)
                new_weight_vec = self.current_weight_vec + curr_feature_vec - new_feature_vec
                self.current_weight_vec_iter += 1
                self.current_weight_vec = new_weight_vec
        if i + 1 in [20, 50, 80, 100]:
            with open(os.path.join(self._directory, 'final_weight_vec_{}.pkl'.format(i + 1)), 'wb') as f:
                pickle.dump(self.current_weight_vec, f)
    print("{}: the number of weight updates in this training:{}".format(time.asctime(time.localtime(time.time()))
                                                                        , self.current_weight_vec_iter))
    logging.info("{}: the number of weight updates in this training:{}"
                 .format(time.asctime(time.localtime(time.time())), self.current_weight_vec_iter))
    with open(os.path.join(self._directory, 'final_weight_vec_{}.pkl'.format(num_of_iter)), 'wb') as f:
        pickle.dump(self.current_weight_vec, f)
    return self.current_weight_vec


def error_factor(x,y):
    # how wrong were you?

    pred = MulticlasslabelPerceptron.labels_mapping[x]
    true = MulticlasslabelPerceptron.labels_mapping[y]

    pred_label_list = pred.split('_')
    true_label_list = true.split('_')

    denominator = pred_label_list + list(set(true_label_list) - set(pred_label_list))

    count_correct = 0.0
    for label in denominator:
        if label in pred_label_list and label in true_label_list:
            count_correct += 1

    # error %
    factor = len(denominator) - count_correct / len(denominator)

    return factor


# should get the possiable label list - ordered
# examples shouls be splitted

class MulticlasslabelPerceptron():
    """
        Multi class Perceptron classifier which, given a tuple of features and label,
        training a model based on them; i.e. learn a k weight vectors (where k==number of possible labels),
        that a linear combination of it with the features, produces a classification.
        then, it returns the predicted label which yields the maximum prediction for the label`s weight vector
    """

    def __init__(self, examples, true_labels, labels, original_labels, iterations):
        """

        :param examples: the object to predict containing sample's features, true_labels: sample's true labels, labels: all exsiting labels
        :type examples: csr matrix , true_labels: marix of strings,
        :param original_labels: bag of words - all the possible unigrams
        """
        self.__ZERO = 0
        self.labels_mapping = {}
        self.reverse_label_mapping = {}
        self.num_of_labels = 0
        self.original_labels = original_labels

        # map labels
        for i in range(len(labels)):
            self.labels_mapping[i] = labels[i]
            self.num_of_labels += 1

        # init weight matrix
        self.weight = np.zeros(shape=( self.num_of_labels, examples.shape[1]), dtype=np.float16)

        #self.fit(examples, iterations)

    def predict(self, x):
        """
        predict the label of the object

        :param x: the features of the object we which to predict
        :type x: dict[Union[str, int], Union[int, float]]
        :return: the predicted label, which yields the argmax label_k over the weight_k*x
        :rtype: Union[str,int,bool]

        """
        # predict_list = [self.dot_product(x, i) for i in range(self.num_of_labels)]
        # label_index = predict_list.index(max(predict_list))
        # return self.reverse_label_mapping[label_index]

        predict_mat = np.dot( self.weight, x)
        max_perdition = predict_mat.max(0)  # hopefully max per row

        return max_perdition

    def fit(self, examples, true_labels, iterations):
        """
        fit the model using the training data, by updating the weights (per class), using the
        :method:`BinaryPerceptron.update_weight()`)
        if the predicted != correct_label, then updates the correct_label weight vector,
        and for the predicted label weight vector

        :param int iterations: number of iteration to run the training
        :param examples: the object to predict, with its` correct label
        :type examples: list[tuple[dict[Union[str, int], Union[int,float]], Union[int, bool, str]]]
        """
        # TODO: select an updating policy - naive (normal multiclass update), uniform (all get the same due to the severity), update only the wrong ones

        for i in range(iterations):

            res = self.predict(examples)

            for j in res:
                if res[j] != true_labels[j]:

                    factor = 1
                    # reduce the weight of predicted class
                    self.update_weight(examples[j], -factor, self.labels_mapping[res[j]])
                    # increase the score of the correct class
                    self.update_weight(examples[j], factor, self.labels_mapping[true_labels[j]])

    def update_weight(self, x, factor, i):
        """
        update the i-th weight vector in *self.weight` by adding the feature (or subtracting, depends on the *sign*)

        :param int i: the index of the weight vector
        :param dict[Union[str, int], Union[int,float]] x: features dict of a specific example
        :param int sign: 1 if y_i is positive, otherwise -1

        """

        # for key, val in x.items():
        #     self.weight[i][key] += sign * val

        self.weight[i] += x * factor



    # def dot_product(self, x, i):
    #     """
    #     calculates the dot product between the i-th weight vector and the feautres
    # 
    #     :param dict[Union[str, int], Union[int,float]] x: features dict of a specific example
    #     :param int i: the index of the weight vector
    #     :return: the dot product between the existing features of x to the i-th weight vector
    #     :rtype: int
    #     """
    #     dot_product_res = 0
    #     for key, val in x.items():
    #         dot_product_res += val * self.weight[i][key]
    #     return dot_product_res
