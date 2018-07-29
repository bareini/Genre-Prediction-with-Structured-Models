import unittest
import numpy as np
from struct_perceptron import StructPerceptron, GraphUtil
from chu_liu import Digraph
import os
from scipy.sparse import csr_matrix


class TestStructPerceptron(unittest.TestCase):
    class Model:
        def __init__(self, gold_tree):
            self.feature_vec_len = 5
            self.features_vector_train = csr_matrix(np.random.randint(2, size=5))
            self.create_global_feature_vector = lambda x, y, mode: csr_matrix(np.random.randint(2, size=5))
            self.train_gold_tree = gold_tree
            self.test_gold_tree = gold_tree
            self.comp_gold_tree = gold_tree

        def get_local_feature_vec(self, x , y, z, mode):
            return csr_matrix(np.random.randint(2, size=5))

    def test_create_full_graph(self):
        a = {0: [1, 2, 3], 1: [2, 3], 2: [1, 3], 3: [1, 2]}
        graph = Digraph(a)
        new_graph = graph.mst()
        suc = new_graph.successors
        self.make_vaild(suc)
        suc = {0: suc}
        m = self.Model(suc)
        perc = StructPerceptron(m, directory=directory)
        self.assertRaises(Exception, GraphUtil.create_full_graph(suc))
        # print(perc.full_graph)

    def test_identical_dependency_tree(self):
        a = {0: [1, 2, 3], 1: [2, 3], 2: [1, 3], 3: [1, 2]}
        graph = Digraph(a)
        new_graph = graph.mst()
        suc = new_graph.successors
        self.make_vaild(suc)
        suc = {0: suc}
        m = self.Model(suc)
        perc = StructPerceptron(m, directory=directory)
        GraphUtil.create_full_graph(suc)
        full_graph = perc.full_graph[0]
        self.assertEqual(True, GraphUtil.identical_dependency_tree(full_graph, a))
        # self.assertEqual(full_graph, a)

    def test_perceptron(self):
        n = 10
        a = {0: [1, 2, 3], 1: [2, 3], 2: [1, 3], 3: [1, 2]}
        graph = Digraph(a)
        new_graph = graph.mst()
        suc = new_graph.successors
        suc = self.make_vaild(suc)
        suc = {0: suc}
        m = self.Model(suc)
        perc = StructPerceptron(m, directory=directory)
        self.assertRaises(Exception, perc.perceptron(n))
        print(perc.perceptron(n).A)

    def make_vaild(self, suc):
        """
        :type suc: dict[int,List[int]]
        :param suc:
        :return:
        """
        remove_set = set()
        for source, targets in suc.items():
            if len(targets) < 1:
                remove_set.add(source)
        for node in remove_set:
            suc.pop(node)
        return suc


directory = os.path.abspath("output\\temp\\")
if __name__ == '__main__':

    unittest.main()
