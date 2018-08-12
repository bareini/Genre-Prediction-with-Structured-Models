import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize

from model import Model


class ParametersMEMM:
    """
    This class implements parameters learning, assuming MEMM model, using gradient decent (implemented using l-bgfs)
    """

    def __init__(self, model, lambda_):
        """


        :param float lambda_: the regularization term
        :param Model model: instance of the model class
        """
        self.lambda_ = lambda_
        self.model = model
        self.w = np.ones(shape=self.model.feature_vector_len, dtype=np.float16)

    def obj_function(self, w):
        """


        :param np.ndarray w: the weight vector
        :return: the value of the current loss function
        :rtype: float
        """
        linear_term = np.sum(self.model.train_feature_matrix.dot(w.transpose()))

        normalized_partial_sums = []
        for node_idx, node_matrix in self.model.possible_genres_per_node_matrix.items():
            normalized_partial_sums.append(np.sum(np.log(np.sum(np.exp(node_matrix.dot(w))))))

        normalized_sum = np.sum(normalized_partial_sums)

        regularization_part = self.lambda_ * 0.5 * np.dot(w, w)

        return -(linear_term - normalized_sum - regularization_part)

    def gradient(self, w):
        """
        this method calculates the gradient of the weight vector

        :param w:
        :return:
        """

        # empirical counts that are converted from matrix to vector
        empirical_counts = np.squeeze(np.asarray(csr_matrix.sum(self.model.train_feature_matrix, axis=0)))

        # expected counts
        expected_counts = np.zeros(self.model.feature_vector_len)
        for node_idx, node_matrix in self.model.possible_genres_per_node_matrix.items():
            nominators = np.exp(node_matrix.dot(w))
            denominator = np.sum(nominators)
            prob = nominators / denominator
            expected_counts += node_matrix.transpose().dot(prob)

        return -(empirical_counts - expected_counts - self.lambda_ * w)

    def gradient_decent(self, vec_file_name, res_file_name):
        """

        :param f_name:
        :return:
        """
        results = minimize(
            fun=self.obj_function,
            x0=self.w ,
            method='L-BFGS-B',
            jac=self.gradient,
            options={'disp': True, 'maxiter': 40, 'ftol': 1e2 * np.finfo(float).eps}
        )
        self.w = results.x
        with open(res_file_name, "w") as file_object:
            # file_object.write("{}".format(results[1]))
            # file_object.write("{}".format(results[2]))
            file_object.write("{}".format(results))
        np.savetxt(fname=vec_file_name, X=results.x, delimiter=',')