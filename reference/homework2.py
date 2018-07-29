############################################################
# Homework 2
############################################################

student_name = "066434952_305200545_303013981"
# using python 3

############################################################
# Imports
############################################################

import homework2_data as data
import math

# Include your imports here, if any are used.


############################################################
# Section 1: Perceptrons
############################################################

class BinaryPerceptron(object):
    """
    Binary Perceptron classifier which, given a tuple of features and label,
    training the model based on them; i.e. learn a weight vector, that a linear combination of it with
    the features, produces a classification.
    """
    def __init__(self, examples, iterations):
        """

        :param examples: the object to predict, with its` correct prediction
        :type examples: list[tuple[dict[Union[str,int], Union[int,float]], Union[int, bool, str]]]
        :param int iterations: the number of iterations to run the perceptron fit
        """

        self.__ZERO = 0
        self.weight = dict()
        self.init_weight(examples)
        self.fit(iterations, examples)

    def predict(self, x):
        """
        predict the binary classification of the object - i.e. return if the dot product
        between the weight to the features > 0

        :param x: the features of the object we which to predict
        :type x: dict[Union[str, int], Union[int, float]]
        :return: a binary prediction on the given features
        :rtype: bool
        """
        return self.dot_product(x) > 0

    def dot_product(self, x):
        """
        calculates a dot product between the weight vectors and the sparse features dict

        :param x: the features of the object we which to predict
        :type x: dict[Union[str, int], Union[int, float]]
        :return: the results of the dot product
        :rtype: int
        """
        dot_product_res = 0
        for key, val in x.items():
            dot_product_res += val * self.weight[key]
        return dot_product_res

    def fit(self, iterations, examples):
        """
        fit the model using the training data, by updating the *self.weight* attribute, using the
        :method:`BinaryPerceptron.update_weight()`)

        :param int iterations: number of iteration to run the training
        :param examples: the object to predict, with its` correct prediction
        :type examples: list[tuple[dict[Union[str, int], Union[int,float]], Union[int, bool, str]]]
        """
        for i in range(iterations):
            for (x, y) in examples:
                res = self.predict(x)
                if res != y:
                    sign = 1 if y else -1
                    self.update_weight(x, sign)

    def update_weight(self, x, sign):
        """
        update the *self.weight` by adding the feature (or subtracting, depends on the *sign*)

        :param dict[Union[str, int], Union[int,float]] x: features dict of a specific example
        :param int sign: 1 if y_i is positive, otherwise -1
        """
        for key, val in x.items():
            self.weight[key] += sign * val

    def init_weight(self, examples):
        """
        init the weight of the class to 0 (in the size of the features space)

        :param examples: the object to predict, with its` correct prediction
        :type examples: list[tuple[dict[Union[str, int], Union[int,float]], Union[int, bool, str]]]
        """
        features = set()
        for x, y in examples:
            features.update(x.keys())
        self.weight = {key: self.__ZERO for key in features}


class MulticlassPerceptron(object):
    """
        Multi class Perceptron classifier which, given a tuple of features and label,
        training a model based on them; i.e. learn a k weight vectors (where k==number of possible labels),
        that a linear combination of it with the features, produces a classification.
        then, it returns the predicted label which yields the maximum prediction for the label`s weight vector
    """

    def __init__(self, examples, iterations):
        """

        :param examples: the object to predict, with its` correct prediction
        :type examples: list[tuple[dict[Union[str, int], Union[int,float]], Union[int, bool, str]]]
        :param iterations: number of iteration to run the training
        """
        self.__ZERO = 0
        self.weight = []
        self.labels_mapping = {}
        self.reverse_label_mapping = {}
        self.num_of_labels = 0
        self.init_weight(examples)
        self.fit(examples, iterations)

    def predict(self, x):
        """
        predict the label of the object

        :param x: the features of the object we which to predict
        :type x: dict[Union[str, int], Union[int, float]]
        :return: the predicted label, which yields the argmax label_k over the weight_k*x
        :rtype: Union[str,int,bool]

        """
        predict_list = [self.dot_product(x, i) for i in range(self.num_of_labels)]
        label_index = predict_list.index(max(predict_list))
        return self.reverse_label_mapping[label_index]

    def fit(self, examples, iterations):
        """
        fit the model using the training data, by updating the weights (per class), using the
        :method:`BinaryPerceptron.update_weight()`)
        if the predicted != correct_label, then updates the correct_label weight vector,
        and for the predicted label weight vector

        :param int iterations: number of iteration to run the training
        :param examples: the object to predict, with its` correct label
        :type examples: list[tuple[dict[Union[str, int], Union[int,float]], Union[int, bool, str]]]
        """
        for i in range(iterations):
            for (x, y) in examples:
                res = self.predict(x)
                if res != y:
                    # reduce the weight of predicted class
                    self.update_weight(x, -1, self.labels_mapping[res])
                    # increase the score of the correct class
                    self.update_weight(x, 1, self.labels_mapping[y])

    def init_weight(self, examples):
        """
        initialize k weight vectors (where k = |# of labels|) in *self.weight*, in the size of the features space.

        :param examples: the object to predict, with its` correct label
        :type examples: list[tuple[dict[Union[str, int], Union[int,float]], Union[int, bool, str]]]
        """
        features = set()
        labels_set = set()
        for x, y in examples:
            features.update(x.keys())
            labels_set.update([y])
        self.num_of_labels = len(labels_set)
        # map the possible labels to their respected place in *self.weight* list
        self.labels_mapping = {key: i for i, key in enumerate(labels_set)}
        # map the indexes of the label's weight vector to their actual label
        self.reverse_label_mapping = {i: key for i, key in enumerate(labels_set)}
        self.weight = [{key: self.__ZERO for key in features} for _ in range(self.num_of_labels)]

    def update_weight(self, x, sign, i):
        """
        update the i-th weight vector in *self.weight` by adding the feature (or subtracting, depends on the *sign*)

        :param int i: the index of the weight vector
        :param dict[Union[str, int], Union[int,float]] x: features dict of a specific example
        :param int sign: 1 if y_i is positive, otherwise -1

        """
        for key, val in x.items():
            self.weight[i][key] += sign * val

    def dot_product(self, x, i):
        """
        calculates the dot product between the i-th weight vector and the feautres

        :param dict[Union[str, int], Union[int,float]] x: features dict of a specific example
        :param int i: the index of the weight vector
        :return: the dot product between the existing features of x to the i-th weight vector
        :rtype: int
        """
        dot_product_res = 0
        for key, val in x.items():
            dot_product_res += val * self.weight[i][key]
        return dot_product_res


############################################################
# Section 2: Applications
############################################################

class IrisClassifier(object):
    """
    the train accuracy is : 0.9666666666666667
    """
    def __init__(self, data):
        """
        this class classifies the Iris data, using :class:`<MulticlassPerceptron>`
        (As there are multiple possible labels)

        :param data: the provided iris data
        :type data: list[tuple[float, float, float, float], str]
        """
        # feature extraction from the data in the instructions
        self.features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        example = [(self.create_dict(x), y) for (x, y) in data]
        iterations = 15
        self.predictor = MulticlassPerceptron(example, iterations)

    def classify(self, instance):
        """
        classify using the multi-class perceptron

        :param instance: the sample we want to classify
        :type instance: tuple[float, float, float, float], Union[int, bool, str]]
        :return: the predicted label
        """
        return self.predictor.predict(self.create_dict(instance))

    def create_dict(self, x):
        """
        creation of the sparse dict, where the key is the feature name, and value is the value of the feature

        :param tuple[float, float, float, float] x: the feature 4-tuple
        :return: a dict
        :rtype: dict[str, Union[int,float]]
        """
        return {key: val for key, val in zip(self.features, x)}


class DigitClassifier(object):
    """
    the train accuracy is: 0.985090243264452
    """
    def __init__(self, data):
        """
        this class classifies the MNIST data, using :class:`<MulticlassPerceptron>`
        (As there are multiple possible labels (digits))

        :param list[tuple[int*16], int] data: the 8*8 representation of the digit
        """
        # the features names are index
        self.features = list(range(64))
        example = [(self.create_dict(x), y) for (x, y) in data]
        iterations = 27
        self.predictor = MulticlassPerceptron(example, iterations)

    def classify(self, instance):
        """
        classify using the multi-class perceptron

        :param instance: the sample we want to classify
        :type instance: tuple[int*16], Union[int, bool, str]]
        :return: the predicted label
        """
        return self.predictor.predict(self.create_dict(instance))

    def create_dict(self, x):
        """
        creation of the sparse dict, where the key is the feature name, and value is the value of the feature

        :param tuple[int*16] x: the sample we want to classify
        :return: a feature dict
        :rtype: dict[int, int]
        """
        return {key: val for key, val in zip(self.features, x)}


class BiasClassifier(object):
    """
    the train accuracy is: 1.0
    """
    def __init__(self, data):
        """
        this class classifies a 1D data, using :class:`<BinaryPerceptron>`, whether the x values is >1 or not

        :param list[tuple[float, bool]] data: 1d point
        """
        # added a feature which serves as an indicator whether x > 1
        self.features = ['x', 'larger_than_one']
        example = [(self.create_dict(x), y) for (x, y) in data]
        iterations = 2
        self.predictor = BinaryPerceptron(example, iterations)

    def classify(self, instance):
        """
        classify with binary perceptron

        :param int instance:
        :return: the prediction
        :rtype: bool
        """
        return self.predictor.predict(self.create_dict(instance))

    def create_dict(self, x):
        """
        creates a sparse dict for the example x (with the added feature, as described in __init__)

        :param int x: the x point
        :return: a features dict
        :rtype: dict[str, int]
        """
        bias_dict = dict({self.features[0]: x, self.features[1]: 1 if x >= 1 else -1})
        return bias_dict


class MysteryClassifier1(object):
    """
    the train accuracy is: 1.0
    """
    def __init__(self, data):
        """
        classify whether a point is out of a circle with radius of 2, and center in (0,0) or not
        using Binary Perceptron

        in this class we added two features:
        * if the distance from (0,0) to the point is >= 2, then returns the distance, else the distance with minus sign
        * the distance from (0,0) multiplied by the delta (max(x1,x2)-min(x1,x2)) of the axes :
            * this feature was added to deal with extreme values

        :param list[tuple[tuple[int, int], bool]] data: the training data
        """
        self.features = ['x', 'y']  # original points
        example = [(self.create_dict(x), y) for (x, y) in data]
        iterations = 1
        self.predictor = BinaryPerceptron(example, iterations)

    def classify(self, instance):
        """
        classify using binary perceptron

        :param Tuple[float, float] instance: the sample to classify
        :return: binary prediction whether the point is in the circle or not
        :rtype: bool
        """
        return self.predictor.predict(self.create_dict(instance))

    @staticmethod
    def radius(x):
        """
        calculates the distance of a point from (0,0)

        :param tuple[int, int] x: the point
        :return: the distance from (0,0)
        :rtype: float
        """
        return math.sqrt(sum((0-x_i) ** 2 for x_i in x))

    def create_dict(self, x):
        """
        creates the features dict.
        the added features were described in the "__init__"

        :param tuple[int, int] x: a point
        :return: a features dict
        :rtype: dict[str, float]
        """
        mist_dict = {key: val for key, val in zip(self.features, x)}
        res_radius = MysteryClassifier1.radius(x)
        max_min = max(x) - min(x)
        mist_dict.update({'radius_scale': res_radius if res_radius >= 2 else -res_radius})
        mist_dict.update({'radius': res_radius*max_min})
        return mist_dict


class MysteryClassifier2(object):
    """
    the train accuracy is: 1.0
    """
    def __init__(self, data):
        """
        classify whether we have an even number of minus sign (where zero is defined to be even) in the 3-points tuple
        or not, using Binary Perceptron.

        in this class we added two features:

        * if the multiplication of the points >= 0, then returns the delta (max(x1,x2,x3)-min(x1,x2,x3)) else the delta with minus sign
            * this feature was added to deal with extreme values
        * the multiplication of the points multiplied by the delta (max(x1,x2,x3)-min(x1,x2,x3)) of the axes

        :param list[tuple[float, float, float], bool] data: the training data
        """
        self.features = ['x', 'y', 'z']
        example = [(self.create_dict(x), y) for (x, y) in data]
        iterations = 10
        self.predictor = BinaryPerceptron(example, iterations)

    def classify(self, instance):
        """
        classify using binary perceptron

        :param tuple[float, float, float] instance: the point to classify
        :return: return the prediction (if the number of minuses are even)
        :rtype: bool
        """
        return self.predictor.predict(self.create_dict(instance))

    @staticmethod
    def mul(x):
        """
        return the multiplication of the points

        :param tuple(float, float, float) x: the point to predict
        :return: the multiplication of the points
        :rtype: float
        """
        x1, x2, x3 = x
        return x1*x2*x3

    def create_dict(self, x):
        """
        creates the features dict.
        the added features were described in the "__init__"

        :param tuple[float, float, float] x: a point
        :return: a features dict
        :rtype: dict[str, float]
        """
        mist_dict = {key: val for key, val in zip(self.features, x)}
        res_mul = MysteryClassifier2.mul(x)
        max_min = max(x) - min(x)
        mist_dict.update({'mul': res_mul*max_min})
        mist_dict.update({'mul_sign': max_min if res_mul >= 0 else -max_min})
        return mist_dict


# if __name__ == '__main__':
    # train = [({"x1": 1}, True), ({"x2": 1}, True), ({"x1": -1},False), ({"x2": -1}, False)]
    # test = [{"x1": 1}, {"x1": 1, "x2": 1}, {"x1": -1, "x2": 1.5}, {"x1": -0.5, "x2": -2}]
    #
    # p = BinaryPerceptron(train, 1)
    # print([p.predict(x) for x in test])
    # train = [({"x1": 1}, 1), ({"x1": 1, "x2": 1}, 2), ({"x2": 1}, 3),
    #          ({"x1": -1, "x2": 1}, 4), ({"x1": -1}, 5), ({"x1": -1, "x2": -1}, 6),
    #          ({"x2": -1}, 7), ({"x1": 1, "x2": -1}, 8)]
    #
    # p = MulticlassPerceptron(train, 10)
    #
    # # Test whether the classifier correctly learned the training data
    # print([p.predict(x) for x, y in train])
    # c = IrisClassifier(data.iris)
    # c_test_classify = [c.classify(x[0]) for x in data.iris]
    # acc = [1 for (y, y_prime) in zip(data.iris, c_test_classify) if y[1] == y_prime]
    # print(sum(acc)/len(data.iris))
    # print(c.classify((5.1, 3.5, 1.4, 0.2)))
    # print(c.classify((7.0, 3.2, 4.7, 1.4)))
    # c = DigitClassifier(data.digits)
    # c_test_classify = [c.classify(x[0]) for x in data.digits]
    # acc = [1 for (y, y_prime) in zip(data.digits, c_test_classify) if y[1] == y_prime]
    # print(sum(acc) / len(data.digits))

    # print(c.classify((0, 0, 5, 13, 9, 1, 0, 0, 0, 0, 13, 15, 10, 15, 5, 0, 0, 3, 15, 2, 0, 11, 8, 0, 0
    #                   , 4, 12, 0, 0, 8, 8, 0, 0, 5, 8, 0, 0, 9, 8, 0, 0, 4, 11, 0, 1, 12, 7, 0, 0, 2,
    #                   14, 5, 10, 12, 0, 0, 0, 0, 6, 13, 10, 0, 0, 0)))
    # c = BiasClassifier(data.bias)
    # c_test_classify = [c.classify(x[0]) for x in data.bias]
    # acc = [1 for (y, y_prime) in zip(data.bias, c_test_classify) if y[1] == y_prime]
    # print(sum(acc) / len(data.bias))


    # print([c.classify(x) for x in (-1, 0, 0.5, 1.5, 2, 0.99, 1.000000001)])
    # c = MysteryClassifier1(data.mystery1)
    # c = BiasClassifier(data.bias)
    # c_test_classify = [c.classify(x[0]) for x in data.mystery1]
    # acc = [1 for (y, y_prime) in zip(data.mystery1, c_test_classify) if y[1] == y_prime]
    # print(sum(acc) / len(data.mystery1))

    # print([c.classify(x) for x in ((0, 0), (0, 1), (-1, 0), (1, 2), (-3, -4), (991000, 98851000), (0, 2))])
    # print(c.predictor.weight)

    # c = MysteryClassifier2(data.mystery2)
    # c_test_classify = [c.classify(x[0]) for x in data.mystery2]
    # acc = [1 for (y, y_prime) in zip(data.mystery2, c_test_classify) if y[1] == y_prime]
    # print(sum(acc) / len(data.mystery2))
    #
    # print([c.classify(x) for x in ((1, 1, 1), (-1000, -10000, -10000), (1, 2,-3), (-1, -2, 3), (60000, -0.053, 0.1), (-1231230, -1231230,-1231231), (-5000, -5000, -5000))])
    # print(c.predictor.weight)


