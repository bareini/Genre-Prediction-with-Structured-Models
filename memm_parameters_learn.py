import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize


class ParametersMEMM:
    """

    """

    def __init__(self, model):
        """

        :param model:
        """
        self.model = model

    def obj_function(self, w):
        """

        :param np.ndarray w:
        :return:
        """
        linear_term = np.sum(self.model.dot(w.transpose()))
        np.sum(np.log(np.sum(np.exp())))