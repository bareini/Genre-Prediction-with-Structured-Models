import time
import logging
from reference.chu_liu import Digraph
import numpy as np
from scipy.sparse import csr_matrix
from collections import namedtuple, defaultdict, Counter

# Constants
TRAIN_FILE_NAME = "train.labeled"
TEST_FILE_NAME = "test.labeled"
COMPETITION_FILE_NAME = "comp.unlabeled"

# Subclass of tuple from the form of Node as given in the exercise
Node = namedtuple('Node', ('counter', 'word', 'pos', 'head'))


class Utils:
    def __init__(self, model_type):

        self.model_type = model_type

    def parse_data(self, file_name):
        """
        :param file_name: train, test ot compeition file
        :return:parse_data
        """

        logger.debug("Utils: parse_data -------------> ")
        trees_list = []
        root_node = Node(0, "Root", "*", 0)
        with open(file_name, "r") as data:
            tree_nodes = [root_node]
            for row in data:
                # Add the tree and start a new tree
                if row == "\n":
                    trees_list.append(tree_nodes)
                    tree_nodes = [root_node]
                    continue

                tokens = row.split("\t")
                if file_name == COMPETITION_FILE_NAME:
                    tokens[6] = -1
                node = Node(int(tokens[0]), tokens[1], tokens[3], int(tokens[6]))
                tree_nodes.append(node)

        logger.debug("Utils: parse_data <------------- ")
        return trees_list

    def save_weights_vector(self, w, num_iter, model_type):

        logger.debug("Utils: save_weights_vector -------------> ")

        file_name = str(num_iter) + '_weigths_vector_' + model_type
        # Save the weigths vector to file
        np.savetxt(fname=file_name, X=w, delimiter=",")

        logger.debug("Utils: save_weights_vector <------------- ")

    def calculate_accuracy(self, true_head_list, pred_head_list):

        logger.debug("Utils: calculate_accuracy -------------> ")
        count_correct = 0
        total = len(true_head_list)
        for i in range(0, total):
            if true_head_list[i] == pred_head_list[i]:
                count_correct += 1

        logger.debug("Utils: calculate_accuracy <------------- ")
        return float(count_correct) / total

    def create_comp_file(self, head_prediction_list, trees):

        logger.debug("Utils: create_comp_file -------------> ")

        try:

            if self.model_type == 'basic':
                file_name = 'comp_m1_305286528.wtag'
            else:
                file_name = 'comp_m2_305286528.wtag'

            with open(file_name, "a") as new_file:
                # Hold the accumulate length of trees
                tree_len = 0
                for tree in trees:
                    for node in tree:
                        if node.counter == 0:
                            continue
                        text_to_write = str(node.counter) + "\t" + \
                                        node.word + "\t" + \
                                        "_" + "\t" + \
                                        node.pos + "\t" + \
                                        "_" + "\t" + \
                                        "_" + "\t" + \
                                        str(head_prediction_list[tree_len + node.counter - 1]) + "\t" + \
                                        "_" + "\t" + \
                                        "_" + "\t" + \
                                        "_"
                        new_file.write(text_to_write)
                        new_file.write("\n")

                    new_file.write("\n")
                    tree_len += len(tree) - 1

            logger.debug("Utils: create_comp_file <------------- ")

        except Exception as e:
            print(e)


class Features:
    def __init__(self, model_type):

        self.model_type = model_type

    def create_features(self, modifier, head, tree):

        # logger.debug("Features: create_features -------------> ")
        directed_distance = np.sign(modifier.counter - head.counter)
        distance = abs(modifier.counter - head.counter)
        cword = modifier.word.lower()
        cpos = modifier.pos
        crpos = "STOP"
        crrpos = "STOP"
        clpos = "**"
        cllpos = "**"
        # left node of modifier in the sentence
        if modifier.counter > 1:
            clpos = tree[modifier.counter - 1].pos
            if modifier.counter > 2:
                cllpos = tree[modifier.counter - 2].pos
        # right node of modifier in the sentence
        if modifier.counter < len(tree) - 1:
            crpos = tree[modifier.counter + 1].pos
            if modifier.counter < len(tree) - 2:
                crrpos = tree[modifier.counter + 2].pos

        pword = head.word.lower()
        ppos = head.pos
        prpos = "STOP"
        prrpos = "STOP"
        plpos = "**"
        pllpos = "**"
        # left node of head in the sentence
        if head.counter > 1:
            plpos = tree[head.counter - 1].pos
            if head.counter > 2:
                pllpos = tree[head.counter - 2].pos
        # right node of head in the sentence
        if head.counter < len(tree) - 1:
            prpos = tree[head.counter + 1].pos
            if head.counter < len(tree) - 2:
                prrpos = tree[head.counter + 2].pos

        between = ""
        if directed_distance == 1:
            for i in range(1, distance - 1):
                between += "_" + tree[head.counter + i].pos
        else:
            for i in range(1, distance - 1):
                between += "_" + tree[modifier.counter + i].pos

        # logger.debug("Features: create_features <------------- ")
        if self.model_type == "basic":
            return [pword + "_" + ppos + "_01",
                    pword + "_02",
                    ppos + "_03",
                    cword + "_" + cpos + "_04",
                    cword + "_05",
                    cpos + "_06",
                    ppos + "_" + cword + "_" + cpos + "_08",
                    pword + "_" + ppos + "_" + cpos + "_10",
                    ppos + "_" + cpos + "_13"]

        return [
            # pword + "_" + ppos + "_01",
            # pword + "_02",
            # ppos + "_03",
            # cword + "_" + cpos + "_04",
            # cword + "_05",
            # cpos + "_06",
            pword + "_" + ppos + "_" + cword + "_" + cpos + "_07",
            ppos + "_" + cword + "_" + cpos + "_08",
            pword + "_" + cword + "_" + cpos + "_09",
            pword + "_" + ppos + "_" + cpos + "_10",
            # pword + "_" + ppos + "_" + cword + "_11",
            pword + "_" + cword + "_12",
            ppos + "_" + cpos + "_13",
            pword + "_" + ppos + "_" + prpos + "_14",
            pword + "_" + ppos + "_" + plpos + "_15",
            ppos + "_" + prpos + "_" + plpos + "_16",
            cword + "_" + cpos + "_" + crpos + "_17",
            # cword + "_" + cpos + "_" + clpos + "_18",
            cpos + "_" + crpos + "_" + clpos + "_19",
            cpos + "_" + ppos + "_" + prpos + "_20",
            cpos + "_" + ppos + "_" + plpos + "_21",
            cpos + "_" + ppos + "_" + clpos + "_22",
            cpos + "_" + ppos + "_" + crpos + "_23",
            ppos + "_" + crpos + "_" + clpos + "_24",
            cpos + "_" + prpos + "_" + plpos + "_25",
            cpos + "_" + ppos + "_" + crpos + "_" + clpos + "_26",
            cpos + "_" + ppos + "_" + prpos + "_" + plpos + "_27",
            cpos + "_" + ppos + "_" + crpos + "_" + prpos + "_28",
            cpos + "_" + ppos + "_" + clpos + "_" + plpos + "_29",
            cpos + "_" + ppos + "_" + clpos + "_" + prpos + "_30",
            cpos + "_" + ppos + "_" + crpos + "_" + plpos + "_31",
            cpos + "_" + ppos + "_" + str(distance) + "_32",
            cpos + "_" + str(distance) + "_33",
            ppos + "_" + str(distance) + "_34",
            cpos + "_" + ppos + "_" + str(directed_distance) + "_35",
            cpos + clpos + cllpos + ppos + "_36",
            cpos + crpos + crrpos + ppos + "_37",
            ppos + plpos + pllpos + cpos + "_38",
            ppos + prpos + prrpos + cpos + "_39",
            between + "_40",
            cpos + clpos + cllpos + crrpos + crrpos + ppos + "_41",
            str(directed_distance) + "_42"
        ]


class DependencyParser:
    def __init__(self, model_type):

        self.features = Features(model_type)
        self.model_type = model_type
        self.features_position = defaultdict(str)
        self.feature_vector_len = 0
        self.words_features_matrix = defaultdict(tuple)
        self.training_features_matrix = defaultdict(tuple)

    def initilaze_training_features(self, trees):
        """
        1. p - word, p - pos
        2. p - word
        3. p - pos
        4. c - word, c - pos
        5. c - word
        6. c - pos
        8. p - pos, c - word, c - pos
        10. p - word, p - pos, c - pos
        13. p - pos, c - pos

        :param model_type:
        :param tree:
        :return:
        """

        logger.debug("DependencyParsing: initilaze_training_features --------------> ")

        features_list = []
        # Go over the trees and create the features
        for tree in trees:
            for modifier in tree:

                if modifier.counter == 0:
                    continue
                head = tree[modifier.head]

                features = self.features.create_features(modifier, head, tree)
                features_list += features

        features_counter = Counter(features_list)

        # Go over the features and decide which one will remain in the model
        # Save the position of each feature in the vector
        position = 0

        # Initialize counters to see how many features we have from each type
        count = defaultdict(int)
        for i in range(1, 43):
            count[i] = 0

        for feature in features_counter:
            # p - word, p - pos
            if feature[-3:] == "_01" and features_counter[feature] > 1:
                self.features_position[feature] = position
                position += 1
                count[1] += 1

            # p - word
            elif feature[-3:] == "_02" and features_counter[feature] > 1:
                self.features_position[feature] = position
                position += 1
                count[2] += 1

            # p - pos
            elif feature[-3:] == "_03" and features_counter[feature] > 1:
                self.features_position[feature] = position
                position += 1
                count[3] += 1

            # c - word, c - pos
            elif feature[-3:] == "_04" and features_counter[feature] > 1:
                self.features_position[feature] = position
                position += 1
                count[4] += 1

            # c - word
            elif feature[-3:] == "_05" and features_counter[feature] > 1:
                self.features_position[feature] = position
                position += 1
                count[5] += 1

            # c - pos
            elif feature[-3:] == "_06" and features_counter[feature] > 1:
                self.features_position[feature] = position
                position += 1
                count[6] += 1

            # p - pos, c - word, c - pos
            elif feature[-3:] == "_08" and features_counter[feature] > 1:
                self.features_position[feature] = position
                position += 1
                count[8] += 1

            # p - word, p - pos, c - pos
            elif feature[-3:] == "_10" and features_counter[feature] > 1:
                self.features_position[feature] = position
                position += 1
                count[10] += 1

            # p - pos, c - pos
            elif feature[-3:] == "_13" and features_counter[feature] > 1:
                self.features_position[feature] = position
                position += 1
                count[13] += 1

            # Advanced:
            if self.model_type == "advanced":
                if feature[-3:] == "_07" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[7] += 1

                elif feature[-3:] == "_09" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[9] += 1

                elif feature[-3:] == "_11" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[11] += 1

                elif feature[-3:] == "_12" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[12] += 1

                elif feature[-3:] == "_14" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[14] += 1

                elif feature[-3:] == "_15" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[15] += 1

                elif feature[-3:] == "_16" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[16] += 1

                elif feature[-3:] == "_17" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[17] += 1

                elif feature[-3:] == "_18" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[18] += 1

                elif feature[-3:] == "_19" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[19] += 1

                elif feature[-3:] == "_20" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[20] += 1

                elif feature[-3:] == "_21" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[21] += 1

                elif feature[-3:] == "_22" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[22] += 1

                elif feature[-3:] == "_23" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[23] += 1

                elif feature[-3:] == "_24" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[24] += 1

                elif feature[-3:] == "_25" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[25] += 1

                elif feature[-3:] == "_26" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[26] += 1

                elif feature[-3:] == "_27" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[27] += 1

                elif feature[-3:] == "_28" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[28] += 1

                elif feature[-3:] == "_29" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[29] += 1

                elif feature[-3:] == "_30" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[30] += 1

                elif feature[-3:] == "_31" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[31] += 1

                elif feature[-3:] == "_32" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[32] += 1

                elif feature[-3:] == "_33" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[33] += 1

                elif feature[-3:] == "_34" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[34] += 1

                elif feature[-3:] == "_35" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[35] += 1

                elif feature[-3:] == "_36" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[36] += 1

                elif feature[-3:] == "_37" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[37] += 1

                elif feature[-3:] == "_38" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[38] += 1

                elif feature[-3:] == "_39" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[39] += 1

                elif feature[-3:] == "_40" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[40] += 1

                elif feature[-3:] == "_41" and features_counter[feature] > 1:
                    self.features_position[feature] = position
                    position += 1
                    count[41] += 1

                elif feature[-3:] == "_42" and features_counter[feature] > 0:
                    self.features_position[feature] = position
                    position += 1
                    count[42] += 1

        self.feature_vector_len = position + 1

        for i in range(1, 43):
            logger.debug("Number of feature " + str(i) + ": " + str(count[i]))

        logger.debug("Total features: " + str(self.feature_vector_len))
        logger.debug("DependencyParsing: initilaze_training_features <-------------- ")

        self.build_features_head_modifier(trees)

    def build_features_head_modifier(self, trees):

        logger.debug("DependencyParsing: build_features_head_modifier -------------->")

        sentence_num = 0
        # Do for all sentences
        for tree in trees:
            sentence_num += 1
            # For each word
            for modifier in tree:
                # If root
                if modifier.counter == 0:
                    continue

                word_matrix_rows_index = []
                word_matrix_columns_index = []

                for possible_head in tree:
                    # node cannot be head of itself
                    if possible_head.counter == modifier.counter:
                        continue

                    columns_index = []

                    current_features = self.features.create_features(modifier, possible_head, tree)
                    for feature in current_features:
                        if feature in self.features_position:
                            word_matrix_rows_index.append(possible_head.counter)
                            columns_index.append(self.features_position[feature])

                    word_matrix_columns_index += columns_index

                    if possible_head.counter == modifier.head:
                        rows_index = np.zeros(shape=len(columns_index), dtype=np.int32)
                        cols_index = np.asarray(a=columns_index, dtype=np.int32)
                        data_to_insert = np.ones(len(columns_index), dtype=np.int8)
                        train_word_vector = csr_matrix((data_to_insert, (rows_index, cols_index)),
                                                       shape=(1, self.feature_vector_len))
                        self.training_features_matrix[(sentence_num, modifier.counter)] = train_word_vector

                rows_index = np.asarray(a=word_matrix_rows_index, dtype=np.int32)
                cols_index = np.asarray(a=word_matrix_columns_index, dtype=np.int32)
                data_to_insert = np.ones(len(word_matrix_rows_index), dtype=np.int8)
                word_matrix = csr_matrix((data_to_insert, (rows_index, cols_index)),
                                         shape=(len(tree), self.feature_vector_len))

                self.words_features_matrix[(sentence_num, modifier.counter)] = word_matrix

        logger.debug("DependencyParsing: build_features_head_modifier <--------------")

    def percpetron(self, trees, num_iteration):

        logger.debug("DependencyParsing: percpetron -------------->")
        start = time.time()
        total_trees = len(trees)

        try:
            # Initialize weight vector
            w = np.zeros(shape=self.feature_vector_len, dtype=np.float16)
            # Initialize average weight vector
            a = np.zeros(shape=self.feature_vector_len, dtype=np.float16)
            for n in range(0, num_iteration):
                num_tree = 1
                for tree in trees:
                    logger.debug("percpetron - number of iterations: " + str(n) + " number tree: " + str(num_tree))
                    for node in tree:
                        # If root - don't need to check
                        if node.counter == 0:
                            continue
                        gold_head = node.head
                        argmax_head = np.argmax(self.words_features_matrix[num_tree, node.counter].dot(w))

                        # Check if the true head of the node equal to the head we got with the max score
                        if gold_head != argmax_head:
                            w = w + self.training_features_matrix[num_tree, node.counter].toarray()[0] - \
                                self.words_features_matrix[num_tree, node.counter][argmax_head, :].toarray()[0]
                            a = a + ((num_iteration-n) * (self.training_features_matrix[num_tree, node.counter].toarray()[0] -
                                                          self.words_features_matrix[num_tree, node.counter][argmax_head, :].toarray()[0]))

                    num_tree += 1

            a = a / (num_iteration * total_trees)
            end = time.time()
            logger.debug("DependencyParsing: percpetron <--------------")
            logger.info("DependencyParsing: time for percpetron - " + str(end - start))
            utils.save_weights_vector(a, num_iteration, self.model_type)
            return a + w

        except Exception as e:
            print(e)
            logger.error(e)


class Inference:
    def __init__(self, model_type, file_type, features_len, features_position, weight_vector=None):

        logger.debug("Inference: init")
        # Init classes
        self.utils = Utils(model_type)
        self.features = Features(model_type)

        if weight_vector is None:
            file_name = '20_weigths_vector_' + model_type
            # Load the weigths vector from file
            self.weight_vector = np.loadtxt(file_name, delimiter=",")
        else:
            self.weight_vector = weight_vector

        if file_type == "test":
            self.trees = utils.parse_data(TEST_FILE_NAME)
        elif file_type == "comp":
            self.trees = utils.parse_data(COMPETITION_FILE_NAME)
        else:
            self.trees = utils.parse_data(TRAIN_FILE_NAME)

        self.features_len = features_len
        self.model_type = model_type
        self.features_position = features_position
        self.mstrees = defaultdict(int)

    def build_successors_graph(self, tree):

        logger.debug("Inference: build_successors_graph -------------->")
        successors = defaultdict(tuple)
        for target in tree:
            # The root can't be modifier only head
            if target.counter == 0:
                continue
            for source in tree:
                # Node can't be a head of itself
                if source.counter == target.counter:
                    continue
                if source.counter not in successors:
                    successors[source.counter] = [target.counter]
                else:
                    successors[source.counter].append(target.counter)

        logger.debug("Inference: build_successors_graph <--------------")
        return successors

    def get_weights(self, tree):

        logger.debug("Inference: get_weights -------------->")
        weights = defaultdict(tuple)
        # For each word
        for modifier in tree:
            # If root
            if modifier.counter == 0:
                continue

            word_matrix_rows_index = []
            word_matrix_columns_index = []

            # Every possible head
            for possible_head in tree:
                # node cannot be head of itself
                if possible_head.counter == modifier.counter:
                    continue

                columns_index = []

                current_features = self.features.create_features(modifier, possible_head, tree)
                for feature in current_features:
                    if feature in self.features_position:
                        word_matrix_rows_index.append(possible_head.counter)
                        columns_index.append(self.features_position[feature])

                word_matrix_columns_index += columns_index

            rows_index = np.asarray(a=word_matrix_rows_index, dtype=np.int32)
            cols_index = np.asarray(a=word_matrix_columns_index, dtype=np.int32)
            data_to_insert = np.ones(len(word_matrix_rows_index), dtype=np.int8)
            word_matrix = csr_matrix((data_to_insert, (rows_index, cols_index)),
                                     shape=(len(tree), self.features_len))

            weights[modifier.counter] = word_matrix.dot(self.weight_vector)

        logger.debug("Inference: get_weights <--------------")
        return weights

    def convert_succ_to_list(self):

        pred_head_list = []
        for succ in self.mstrees:
            dict = {}
            for key in self.mstrees[succ]:
                for node in self.mstrees[succ][key]:
                    dict[node] = key

            for item in sorted(dict.items()):
                pred_head_list.append(item[1])

        return pred_head_list

    def convert_tree_to_gold_list(self):

        gold_head_list = []
        for tree in self.trees:
            dict = {}
            for node in tree:
                if node.counter == 0:
                    continue
                dict[node.counter] = node.head

            for item in sorted(dict.items()):
                gold_head_list.append(item[1])

        return gold_head_list

    def make_inference(self):

        logger.debug("Inference: make_inference -------------->")
        start = time.time()
        num_tree = 1
        for tree in self.trees:
            weights = self.get_weights(tree)
            digraph = Digraph(self.build_successors_graph(tree), lambda source, target: weights[target][source])
            mst = digraph.mst()
            self.mstrees[num_tree] = mst.successors
            num_tree += 1

        end = time.time()
        logger.debug("Inference: time - " + str(end - start))

        prediction_list = self.convert_succ_to_list()
        gold_list = self.convert_tree_to_gold_list()
        accuracy = utils.calculate_accuracy(prediction_list, gold_list)
        logger.info("Inference: Accuracy: " + str(accuracy * 100) + " %")

        logger.debug("Inference: make_inference - Finish")


if __name__ == '__main__':

    try:
        # create logger with 'spam_application'
        logger = logging.getLogger("dependency_parsing")
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler('dependency_parsing.log')
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        dp = DependencyParser("advanced")
        utils = Utils("advanced")

        trees = utils.parse_data(TRAIN_FILE_NAME)
        dp.initilaze_training_features(trees)
        w = dp.percpetron(trees, 20)
        inf = Inference("advanced", "test", dp.feature_vector_len, dp.features_position)
        inf.make_inference()

        # for num in [20,50,80,100]:
        #     w = dp.percpetron(trees, 20)

    except Exception as e:
        print(e)