import os
import logging
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import accuracy_score, confusion_matrix as conf_matrix


# create logger with 'spam_application'
logger = logging.getLogger("memm")

RUN_LOSS_FUNCTION = True


class MEMM:
    def __init__(self, train_data, test_data, model_type):

        logger.info("MEMM: init class")
        self.train_data = train_data
        self.test_data = test_data
        self.model_type = model_type

        # Lambda regularizer
        if model_type == 'basic':
            self.lambda_reg = 0.1
        if model_type == 'advanced':
            self.lambda_reg = 1.5

        # All the features that passed the conditions we decided on will be save in this list and we will use them in the model
        self.features_position = {}
        self.features_list = []

        # List of words with all the seen tags for her in the training data
        self.seen_words_with_tags = {}

        # List of all tags seen in the training data
        self.tags_seen_in_train = []

        # For each sentence we save in the list there are 2 sub-lists : 1. sentences list 2. tags list
        self.sentences_in_test_data = [[], []]

        # Save the length of the feature vector
        self.feature_vector_len = 0
        
        # Weights vector V
        if RUN_LOSS_FUNCTION == False:
            if model_type == 'basic':
                file_name = 'weigths_vector_basic'
            else:
                file_name = 'weigths_vector_advanced'
            if os.path.isfile(file_name):
                self.v = np.loadtxt(file_name, delimiter=",")
        else:
            self.v = np.ones(self.feature_vector_len, dtype=np.float16)

        # Initialize trainunf matrix
        self.training_matrix = csr_matrix([0])

        # All positions of 1
        self.all_ones_positions_in_the_feature_vector = {}

    def initialize_features_vector(self):
        """
        Base Features:
            100 - word x tag
            103 - tag trigram
            104 - tag bigram
        Advanced Features:
            101 - prefix <=4 X tag
            102 - suffix <=4 X tag
            105 - tag unigram
            106 - current tag with previous word
            107 - current tag with next word
            captalization
            numeric 
        :return:
        """

        logger.info("initialize_features_vector --------------> ")

        # For each sentence we save in the list there are 2 sub-lists : 1. sentences list 2. tags list
        sentences_in_train_data = []
        # Keep the number of appearences of each feature in the data
        features_counter = {}

        # Initial tags in the position i-1, i-2
        tag1 = tag2 = "*"

        words_list = []
        tags_list = []

        for line in self.train_data:

            pairs = line.split()
            for pair in pairs:

                origin_word = pair.split("_")[0]
                word = origin_word.lower()
                tag = pair.split("_")[1]

                words_list.append(origin_word)
                tags_list.append(tag)

                # Check if we saw the tag or the word and if not add to the list
                if tag not in self.tags_seen_in_train:
                    self.tags_seen_in_train.append(tag)
                if word not in self.seen_words_with_tags:
                    self.seen_words_with_tags[word] = []
                    self.seen_words_with_tags[word].append(tag)
                elif tag not in self.seen_words_with_tags[word]:
                    self.seen_words_with_tags[word].append(tag)

                # word x tag
                word_tag = word + "_" + tag + "_100"
                if word_tag not in features_counter:
                    features_counter[word_tag] = 1
                else:
                    features_counter[word_tag] += 1

                # tag trigram
                trigram = tag2 + "_" + tag1 + "_" + tag + "_103"
                if trigram not in features_counter:
                    features_counter[trigram] = 1
                else:
                    features_counter[trigram] += 1

                # tag bigram
                bigram = tag1 + "_" + tag + "_104"
                if bigram not in features_counter:
                    features_counter[bigram] = 1
                else:
                    features_counter[bigram] += 1

                if self.model_type == "advanced":
                    # prefix <=4 x tag
                    prefix_list = [word[:i] + '_' + tag + "_101" for i in range(1, 5)]
                    prefix_list = list(set(prefix_list))
                    for item in prefix_list:
                        if item not in features_counter:
                            features_counter[item] = 1
                        else:
                            features_counter[item] += 1

                    # suffix <=4 x tag
                    suffix_list = [word[i:] + '_' + tag + "_102" for i in range(-4, 0)]
                    suffix_list = list(set(suffix_list))
                    for item in suffix_list:
                        if item not in features_counter:
                            features_counter[item] = 1
                        else:
                            features_counter[item] += 1

                    # tag unigram
                    unigram = tag + "_105"
                    if unigram not in features_counter:
                        features_counter[unigram] = 1
                    else:
                        features_counter[unigram] += 1

                    # current tag with previous word
                    if tag1 != "*":
                        previous_word_current_tag = words_list[len(words_list)-2].lower() + "_" + tag + "_106"
                        if previous_word_current_tag not in features_counter:
                            features_counter[previous_word_current_tag] = 1
                        else:
                            features_counter[previous_word_current_tag] += 1

                    # current tag with next word
                    if tag1 != "*":
                        # for the convenience - taking the current word with the previous tag
                        next_word_current_tag = word + "_" + tag1 + "_107"
                        if next_word_current_tag not in features_counter:
                            features_counter[next_word_current_tag] = 1
                        else:
                            features_counter[next_word_current_tag] += 1

                # Check if the tag represent end of a sentence
                if tag == ".":
                    assert len(words_list) == len(tags_list)
                    sentences_in_train_data.append([words_list, tags_list])
                    words_list = []
                    tags_list = []
                    tag1 = tag2 = "*"

                else:
                    tag2 = tag1
                    tag1 = tag

        position_counter = 0
        if self.model_type == "advanced":
            position_counter = 2
            # Add to features_list numeric and capital for the right length
            self.features_list.append("is_numeric")
            self.features_position["is_numeric"] = 0

            self.features_list.append("is_capitalized")
            self.features_position["is_capitalized"] = 1
            
        # Initialize counters to see how many features we have from each type
        count_100 = 0
        count_101 = 0
        count_102 = 0
        count_103 = 0
        count_104 = 0
        count_105 = 0
        count_106 = 0
        count_107 = 0
        if self.model_type == 'basic':
            threshold_bigram = 1
        else:
            threshold_bigram = 3
            
        for features_type in features_counter:
            # word x tag
            if features_type[-4:] == "_100" and features_counter[features_type] > 1:
                self.features_list.append(features_type)
                self.features_position[features_type] = position_counter
                position_counter += 1
                count_100 += 1
            # tag trigram
            elif features_type[-4:] == "_103" and features_counter[features_type] > 1:
                self.features_list.append(features_type)
                self.features_position[features_type] = position_counter
                position_counter += 1
                count_103 += 1
            # tag bigram
            elif features_type[-4:] == "_104" and features_counter[features_type] > threshold_bigram:
                self.features_list.append(features_type)
                self.features_position[features_type] = position_counter
                position_counter += 1
                count_104 += 1
            if self.model_type == "advanced":
                # prefix <=4 X tag
                if features_type[-4:] == "_101" and features_counter[features_type] > 5:
                    self.features_list.append(features_type)
                    self.features_position[features_type] = position_counter
                    position_counter += 1
                    count_101 += 1
                # suffix <=4 X tag
                elif features_type[-4:] == "_102" and features_counter[features_type] > 5:
                    self.features_list.append(features_type)
                    self.features_position[features_type] = position_counter
                    position_counter += 1
                    count_102 += 1
                # tag unigram
                elif features_type[-4:] == "_105" and features_counter[features_type] > 3:
                    self.features_list.append(features_type)
                    self.features_position[features_type] = position_counter
                    position_counter += 1
                    count_105 += 1
                # current tag with previous word
                elif features_type[-4:] == "_106" and features_counter[features_type] > 3:
                    self.features_list.append(features_type)
                    self.features_position[features_type] = position_counter
                    position_counter += 1
                    count_106 += 1
                # current tag with next word
                elif features_type[-4:] == "_107" and features_counter[features_type] > 3:
                    self.features_list.append(features_type)
                    self.features_position[features_type] = position_counter
                    position_counter += 1
                    count_107 += 1
                    
        logger.debug("Number of feature 100: " + str(count_100))
        logger.debug("Number of feature 101: " + str(count_101))
        logger.debug("Number of feature 102: " + str(count_102))
        logger.debug("Number of feature 103: " + str(count_103))
        logger.debug("Number of feature 104: " + str(count_104))
        logger.debug("Number of feature 105: " + str(count_105))
        logger.debug("Number of feature 106: " + str(count_106))
        logger.debug("Number of feature 107: " + str(count_107))
        logger.debug("initialize_features_vector <-------------- ")

        self.feature_vector_len = len(self.features_list)
        self.create_features_vector_for_train(sentences_in_train_data)

    def create_features_vector_for_train(self, sentences_in_train_data):
        """
        Create the vector with the chosen features in initialize_featurs_vecor
        Build Train Matrix with only seen words and tags
        """

        logger.info("create_features_vector_for_train --------------> ")

        try:
            # Feature vector of all seen tags and words - unite to one matrix
            training_matrix_rows_index_counter = 0
            training_matrix_rows_index = []
            training_matrix_columns_index = []

            sentence_index = -1
            for item in sentences_in_train_data:
                logger.info("create_features_vector_for_train : go over sentence: " + str(sentence_index))
                sentence_index += 1
                words_sequence = item[0]
                tags_sequence = item[1]

                for position in range(0, len(words_sequence)):
                    # Sub matrix for each word in the sentence with all the possible tags
                    word_matrix_rows_index_counter = 0
                    word_matrix_rows_index = []
                    word_matrix_columns_index = []

                    origin_tag = tags_sequence[position]
                    origin_word = words_sequence[position]
                    word = origin_word.lower()

                    for tag in self.tags_seen_in_train:

                        columns_index = []
                        if position == 0:
                            tag_2 = tag_1 = "*"

                        # Check if this word was seen before
                        if word in self.seen_words_with_tags:
                            pair = word + "_" + tag + "_100"
                            if pair in self.features_position:
                                word_matrix_rows_index.append(word_matrix_rows_index_counter)
                                columns_index.append(self.features_position[pair])
                        # tag trigram
                        trigram = tag_2 + "_" + tag_1 + "_" + tag + "_103"
                        if trigram in self.features_position:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns_index.append(self.features_position[trigram])

                        # tag bigram
                        bigram = tag_1 + "_" + tag + "_104"
                        if bigram in self.features_position:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns_index.append(self.features_position[bigram])

                        if self.model_type == "advanced":
                            # tag unigram
                            unigram = tag + "_105"
                            if unigram in self.features_position:
                                word_matrix_rows_index.append(word_matrix_rows_index_counter)
                                columns_index.append(self.features_position[unigram])

                            # prefix <=4 x tag
                            prefix_list = [word[:i] + '_' + tag + "_101" for i in range(1, 5)]
                            prefix_list = list(set(prefix_list))
                            for prefix in prefix_list:
                                if prefix in self.features_position:
                                    word_matrix_rows_index.append(word_matrix_rows_index_counter)
                                    columns_index.append(self.features_position[prefix])

                            # suffix <=4 x tag
                            suffix_list = [word[i:] + '_' + tag + "_102" for i in range(-4, 0)]
                            suffix_list = list(set(suffix_list))
                            for suffix in suffix_list:
                                if suffix in self.features_position:
                                    word_matrix_rows_index.append(word_matrix_rows_index_counter)
                                    columns_index.append(self.features_position[suffix])

                            # current tag with previous word
                            if tag_1 != "*":
                                previous_word_current_tag = words_sequence[position - 1] + "_" + tag + "_106"
                                if previous_word_current_tag in self.features_position:
                                    word_matrix_rows_index.append(word_matrix_rows_index_counter)
                                    columns_index.append(self.features_position[previous_word_current_tag])

                            # current tag with next word
                            if position != (len(words_sequence) - 1):
                                next_word_current_tag = words_sequence[position + 1] + "_" + tag + "_107"
                                if next_word_current_tag in self.features_position:
                                    word_matrix_rows_index.append(word_matrix_rows_index_counter)
                                    columns_index.append(self.features_position[next_word_current_tag])

                            # is the string a number
                            if str.isdigit(word):
                                word_matrix_rows_index.append(word_matrix_rows_index_counter)
                                columns_index.append(self.features_position["is_numeric"])

                            # is the string start with upper letter
                            if str.isupper(origin_word[0]):
                                word_matrix_rows_index.append(word_matrix_rows_index_counter)
                                columns_index.append(self.features_position["is_capitalized"])

                        word_matrix_columns_index += columns_index
                        word_matrix_rows_index_counter += 1
                        if tag == origin_tag:
                            temp = list(np.ones(len(columns_index), dtype=np.int32) * training_matrix_rows_index_counter)
                            training_matrix_rows_index += temp
                            training_matrix_columns_index += columns_index
                            training_matrix_rows_index_counter += 1

                    rows_index = np.asarray(a=word_matrix_rows_index, dtype=np.int32)
                    cols_index = np.asarray(a=word_matrix_columns_index, dtype=np.int32)
                    data_to_insert = np.ones(len(word_matrix_rows_index), dtype=np.int8)
                    word_matrix = csr_matrix((data_to_insert, (rows_index, cols_index)),
                                             shape=(len(self.tags_seen_in_train), self.feature_vector_len))

                    # Initialize the dict where key : (sentence,word), value : word_matrix
                    self.all_ones_positions_in_the_feature_vector[(sentence_index, position)] = word_matrix
                    tag_2 = tag_1
                    tag_1 = origin_tag

            rows_index = np.asarray(a=training_matrix_rows_index, dtype=np.int32)
            cols_index = np.asarray(a=training_matrix_columns_index, dtype=np.int32)
            data_to_insert = np.ones(len(training_matrix_rows_index), dtype=np.int8)
            self.training_matrix = csr_matrix((data_to_insert, (rows_index, cols_index)),
                                              shape=(training_matrix_rows_index_counter, self.feature_vector_len))

            logger.info("create_features_vector_for_train <-------------- ")

            self.find_maximum_likelihood_estimation()

        except Exception as e:
            print str(e)

    def build_function_lv(self, v):
        """
        Build the function L(v)
        :return:
        """

        logger.info("bulid_function_lv --------------> ")

        try:

            l_func_part_1 = np.sum(self.training_matrix.dot(v.transpose()))

            sum_log_vector = []
            for (i, j) in self.all_ones_positions_in_the_feature_vector:
                mat = self.all_ones_positions_in_the_feature_vector[(i, j)]
                sum_log_vector.append(np.log(np.sum(np.exp(mat.dot(v)))))

            l_func_part_2 = np.sum(sum_log_vector)

            regularization_part = self.lambda_reg * 0.5 * np.dot(v, v)
            # summary L(v)

            logger.info("bulid_function_lv <-------------- ")

            return -(l_func_part_1 - l_func_part_2 - regularization_part)

        except Exception as e:
            print e

    def build_grad(self, v):

        try:

            logger.info("build_grad --------------> ")
            # Empirical counts, convert matrix to vector
            empirical_counts = np.squeeze(np.asarray(csr_matrix.sum(self.training_matrix, axis=0)))

            # Expected counts
            expected_counts = np.zeros(self.feature_vector_len)
            for (i, j) in self.all_ones_positions_in_the_feature_vector:
                mat = self.all_ones_positions_in_the_feature_vector[(i, j)]
                # Calculating all probabilities at once per each (i,j) --> x(i)
                nominators = np.exp(mat.dot(v))
                denominator = np.sum(nominators)
                prob = nominators / denominator
                expected_counts += mat.transpose().dot(prob)

            logger.info("build_grad <-------------- ")

            return -(empirical_counts - expected_counts - self.lambda_reg * v)

        except Exception as e:
            print e

    def find_maximum_likelihood_estimation(self):
        """
        find maximum of L(v) - find argmax v
        return vector v
        :return:
        """

        logger.info("find_maximum_likelihood_estimation --------------> ")
        
        if RUN_LOSS_FUNCTION:
            result = fmin_l_bfgs_b( func   = self.build_function_lv, 
                                    x0     = np.ones(shape=self.feature_vector_len, dtype=np.float16) * 0.1, 
                                    fprime = self.build_grad, 
                                    factr  = 1e12,
                                    pgtol  = 1e-3)
          
            logger.debug("result is: " + str(result))
            self.v = result[0]
      
            if self.model_type == "basic":
                file_name   = "result_basic"
                vector_name = 'weigths_vector_basic'
            else:
                file_name   = "result_advanced"
                vector_name = 'weigths_vector_advanced'
            with open(file_name, "w") as file_object:
                file_object.write(str(result[1]))
                file_object.write(str(result[2]))
            
            # Save the weigths vector to file
            np.savetxt(fname = vector_name, X = result[0], delimiter = ",")
         
        logger.debug("v is: " + str(self.v))
        logger.debug("len(v) is: " + str(len(self.v)))

        logger.info("find_maximum_likelihood_estimation <-------------- ")

        self.make_inference()

    def split_sentences_from_test_data(self, data):

        logger.info("split_sentences_from_test_data --------------> ")
        sentences_in_data = []
        words_list = []

        for line in open(data, "r").readlines():
            words = line.split(" ")
            for word in words:
                words_list.append(word.replace('\r\n',''))
            sentences_in_data.append(words_list)
            words_list = []

        logger.info("split_sentences_from_test_data <-------------- ")
        return sentences_in_data

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
                    probability_table[(t, u, v)] = np.exp(sum(weights[self.find_feature_vector_test_set(sentence, k, t, u, v)]))

                # Constant Denominator
                denominator = np.sum(probability_table[(t, u, v)] for v in Sk)
                for v in Sk:
                    probability_table[(t, u, v)] /= denominator

        return probability_table

    def viterbi_algorithm(self, sentence):

        """
        :type sentence: list of words
        """
        pie = {}
        bp = {}

        logger.info("viterbi_algorithm --------------> ")
        # Base Case
        pie[(0, "*", "*")] = 1.0

        for k in range(1, len(sentence) + 1):

            Sk   = self.tags_seen_in_train
            Sk_1 = self.tags_seen_in_train
            Sk_2 = self.tags_seen_in_train
            
            # if the word appeared in the training data with tags we assign this tags to S
            word = sentence[k-1].lower()
            if word in self.seen_words_with_tags:
                Sk = self.seen_words_with_tags[word]
                
            if k == 1:
                Sk_2, Sk_1 = ["*"], ["*"]
            elif sentence[k-2].lower() in self.seen_words_with_tags:
                Sk_1 = self.seen_words_with_tags[sentence[k-2].lower()]
                
            if k == 2:
                Sk_2 = ["*"]
            elif k > 2 and sentence[k-3].lower() in self.seen_words_with_tags:
                Sk_2 = self.seen_words_with_tags[sentence[k-3].lower()]

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
                    t[n-1] = u

        for k in range(n - 2, 0, -1):
            t[k] = bp[k + 2, t[k + 1], t[k + 2]]

        tag_sequence = []
        for i in t:
            tag_sequence.append(t[i])
            
        if n == 1:
            tag_sequence = [tag_sequence[n]]

        logger.info("viterbi_algorithm <-------------- ")
        return tag_sequence

    def confusion_matrix(self, true_tags_list, pred_tags_list):

        true_tags = []
        pred_tags = []

        for tag_list in true_tags_list:
            true_tags += tag_list

        for tag_list in pred_tags_list:
            pred_tags += tag_list

        assert len(true_tags) == len(pred_tags)
        confusion_matrix = conf_matrix(true_tags, pred_tags, self.tags_seen_in_train)

        return confusion_matrix

    def pretty_print_confusion_matrix(self, matrix, tag_labels):
        """
        Pretty print for confusion matrix
        """
 
        width = max([len(x) for x in tag_labels]+[5]) # 5 digit numbers
        # Print header: the labels
        print "    " + " " * width,
        for label in tag_labels:
            print "%{0}s,".format(width) % label,
        print
        # Print rows
        for i, label1 in enumerate(tag_labels):
            print "    %{0}s,".format(width) % label1,
            for j in range(len(tag_labels)):
                cell = "%{0}d,".format(width) % matrix[i, j]
                print cell,
            print
        
    def calculate_accuracy(self, true_tags_list, pred_tags_list):

        true_tags = []
        pred_tags = []

        for tag_list in true_tags_list:
            true_tags += tag_list

        for tag_list in pred_tags_list:
            pred_tags += tag_list

        assert len(true_tags) == len(pred_tags)
        accuracy = accuracy_score(true_tags, pred_tags)

        return accuracy

    def create_file_with_predictive_tags(self):
        
        logger.info("create_file_with_predictive_tags --------------> ")
        
        try:
            if self.model_type == 'basic':
                file_name = 'comp_m1_305286528.wtag'
            else:
                file_name = 'comp_m2_305286528.wtag'

            with open(file_name, "a") as new_file:
                for sentence_index in range(0, len(self.sentences_in_test_data[0])):
                    text_to_write = ""
                    for word_index in range(0, len(self.sentences_in_test_data[0][sentence_index])):
                        # pair word with its tag
                        pair = self.sentences_in_test_data[0][sentence_index][word_index] + "_" + self.sentences_in_test_data[1][sentence_index][word_index]
                        if word_index != len(self.sentences_in_test_data[0][sentence_index])-1:
                            text_to_write += pair + " "
                        else:
                            text_to_write += pair + "\n"
                            new_file.write(text_to_write)
                            
            logger.info("create_file_with_predictive_tags <-------------- ")
        
        except Exception as e:
            print e

    def make_inference(self):

        logger.info("make_inference --------------> ")
        
        if 'test' in self.test_data:
            data = open(self.test_data, "r")
            self.sentences_in_test_data.append([])
            words_list = []
            tags_list = []

            for line in data:
                pairs = line.split()
                for pair in pairs:
                    word = pair.split("_")[0]
                    tag = pair.split("_")[1]

                    words_list.append(word)
                    tags_list.append(tag)

                    if tag == ".":
                        assert len(words_list) == len(tags_list)
                        self.sentences_in_test_data[0].append(words_list)
                        self.sentences_in_test_data[2].append(tags_list)
                        words_list = []
                        tags_list = []

        else:
            self.sentences_in_test_data[0] = self.split_sentences_from_test_data(self.test_data)

        # Run Viterbi Algorithem for each sentence in the test data and get the tag sequence
        tag_sequence_list = []
        for sentence in self.sentences_in_test_data[0]:
            tag_sequence = self.viterbi_algorithm(sentence)
            tag_sequence_list.append(tag_sequence)
            logger.info("tag sequence is: " + str(tag_sequence))

        self.sentences_in_test_data[1] = tag_sequence_list
        self.create_file_with_predictive_tags()
        # 0 : word sequence
        # 1 : tag prediction sequence
        # 2 : tag gold sequence

#         accuracy = self.calculate_accuracy(self.sentences_in_test_data[2], self.sentences_in_test_data[1])
#         logger.info("accuracy score : " + str(accuracy * 100) + "%")
#         confusion_matrix = self.confusion_matrix(self.sentences_in_test_data[2], self.sentences_in_test_data[1])
#         self.pretty_print_confusion_matrix(confusion_matrix, self.tags_seen_in_train)
        
        logger.info("make_inference <-------------- ")