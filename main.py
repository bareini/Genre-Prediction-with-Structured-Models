import os
import logging
import time
from datetime import datetime
import pickle

from sklearn.model_selection import KFold


import config
from model import Model
import MLpreceptron
# from viterbi_ML import viterbi
from evaluate import Evaluate
from memm_parameters_learn import ParametersMEMM
# from Viterbi_NLP import viterbi
import pandas as pd
from collections import Counter
from viterbi import Viterbi

# # open log connection
# directory = 'C:\\Users\\RomG\\PycharmProjects\\NLP_HW1\\'
# LOG_FILENAME = datetime.now().strftime(directory + 'logs_MEMM/LogFileMEMM_%d_%m_%Y_%H_%M.log')
# logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)


# def cross_validation(train_file_for_cv):
#     # Cross validation part 1: split the data to folds
#     text_file = open(train_file_for_cv, 'r')
#     train_data = text_file.read().split('\n')
#     kf = KFold(n_splits=5, shuffle=True)
#
#     lambda_list = [10.0, 15.0]
#     for lamda in lambda_list:
#         CV_start_time = time.time()
#         logging.info('{}: Start running 5-fold CV for lambda: {}'.format(time.asctime(time.localtime(time.time())),
#                                                                           lamda))
#         print('{}: Start running 5-fold CV for lambda: {}'.format(time.asctime(time.localtime(time.time())), lamda))
#         k = 0
#
#         for train_index, test_index in kf.split(train_data):
#             # Create the train and test data according to the folds, and save the new data
#             train_k_fold = list(train_data[i] for i in train_index)
#             test_k_fold = list(train_data[i] for i in test_index)
#             train_file_cv = datetime.now().strftime(directory + 'data/train_cv_file_%d_%m_%Y_%H_%M.wtag')
#             test_file_cv = datetime.now().strftime(directory + 'data/test_cv_file_%d_%m_%Y_%H_%M.wtag')
#             with open(train_file_cv, 'w', newline='\n') as file:
#                 for sentence in train_k_fold:
#                     file.write(str(sentence) + '\n')
#             with open(test_file_cv, 'w', newline='\n') as file:
#                 for sentence in test_k_fold:
#                     file.write(str(sentence) + '\n')
#
#             feature_type_dict_cv = {
#                 # 'all_features': [['feature_100', 'feature_101', 'feature_102', 'feature_103', 'feature_104',
#                 #                   'feature_105', 'feature_106', 'feature_107', 'feature_108', 'feature_109',
#                 #                   'feature_110', 'feature_111'],
#                 #                  ['feature_100', 'feature_101', 'feature_102', 'feature_103', 'feature_104',
#                 #                   'feature_105', 'feature_106', 'feature_107', 'feature_110', 'feature_111',
#                 #                   'feature_108']],
#                 'basic_model': [['feature_100', 'feature_103', 'feature_104']]}
#
#             for feature_type_name_cv, feature_type_list_cv in feature_type_dict_cv.items():
#                 logging.info('{}: Start running fold number {} for lambda: {}'.
#                              format(time.asctime(time.localtime(time.time())), k, lamda))
#                 print('{}: Start running fold number {} for lambda: {}'
#                       .format(time.asctime(time.localtime(time.time())), k, lamda))
#                 main(train_file_cv, test_file_cv, 'test_cv_fold_' + str(k), feature_type_list_cv, lamda, comp=False)
#
#             run_time_cv = (time.time() - CV_start_time) / 60.0
#             print("{}: Finish running iteration {} of 10-fold CV for lambda: {}. Run time is: {} minutes".
#                   format(time.asctime(time.localtime(time.time())), k, lamda, run_time_cv))
#             logging.info('{}: Finish running iteration {} 10-fold CV for lambda:{} . Run time is: {} minutes'.
#                          format(time.asctime(time.localtime(time.time())), k, lamda, run_time_cv))
#             k += 1


# def main(train_file_to_use, test_file_to_use, test_type, features_combination_list, lamda, comp):
# for perm in itertools.combinations(features_combination_list_sub, 4):
#    features_combination_list.append(list(perm))
# def main():
#     # start all combination of features
#     features_combination_list = []
#     for features_combination in features_combination_list:
#
#         print('{}: Start creating MEMM for features : {}'.format(time.asctime(time.localtime(time.time())),
#                                                                  features_combination))
#         logging.info('{}: Start creating MEMM for features : {}'.format(time.asctime(time.localtime(time.time())),
#                                                                         features_combination))
#         train_start_time = time.time()
#         memm_class = MEMM(directory, train_file_to_use, features_combination)
#
#         logging.info('{}: Finish MEMM for features : {}'.format(time.asctime(time.localtime(time.time())),
#                                                                 features_combination))
#         print('{}: Finish MEMM for features : {}'.format(time.asctime(time.localtime(time.time())),
#                                                          features_combination))
#
#         print('{}: Start gradient for features : {} and lambda: {}'.
#               format(time.asctime(time.localtime(time.time())), features_combination, lamda))
#         logging.info('{}: Start gradient for features : {} and lambda: {}'.
#                      format(time.asctime(time.localtime(time.time())), features_combination, lamda))
#         gradient_class = Gradient(model=memm_class, lambda_value=lamda)
#         gradient_result = gradient_class.gradient_descent()
#
#         train_run_time = (time.time() - train_start_time) / 60.0
#         print('{}: Finish gradient for features : {} and lambda: {}. run time: {}'.
#               format(time.asctime(time.localtime(time.time())), features_combination, lamda, train_run_time))
#         logging.info('{}: Finish gradient for features : {} and lambda: {}. run time: {}'.
#                      format(time.asctime(time.localtime(time.time())), features_combination, lamda, train_run_time))
#
#         weights = gradient_result.x
#         #   np.savetxt(gradient_file, weights, delimiter=",")
#
#         viterbi_start_time = time.time()
#         print('{}: Start viterbi'.format((time.asctime(time.localtime(time.time())))))
#         viterbi_class = viterbi(memm_class, data_file=test_file_to_use, w=weights)
#         viterbi_result = viterbi_class.viterbi_all_data
#         viterbi_run_time = (time.time() - viterbi_start_time) / 60.0
#         print('{}: Finish viterbi. run time: {}'.format((time.asctime(time.localtime(time.time()))), viterbi_run_time))
#         logging.info('{}: Finish viterbi. run time: {}'.format((time.asctime(time.localtime(time.time()))),
#                                                                viterbi_run_time))
#
#         write_file_name = datetime.now().strftime(directory + 'file_results/result_MEMM_most_common_tags_' + test_type +
#                                                   '%d_%m_%Y_%H_%M.wtag')
#         confusion_file_name = datetime.now().strftime(directory + 'confusion_files/CM_MEMM_most_common_tags_' + test_type +
#                                                       '%d_%m_%Y_%H_%M.xls')
#         evaluate_class = Evaluate(memm_class, test_file_to_use, viterbi_result, write_file_name,
#                                   confusion_file_name, comp=comp)
#         if not comp:
#             word_results_dictionary = evaluate_class.run()
#         if comp:
#             evaluate_class.write_result_doc()
#         logging.info('{}: The model hyper parameters: \n lambda:{} \n test file: {} \n train file: {}'
#                      .format(time.asctime(time.localtime(time.time())), lamda, test_file_to_use, train_file_to_use))
#         logging.info('{}: Related results files are: \n {} \n {}'.
#                      format(time.asctime(time.localtime(time.time())), write_file_name, confusion_file_name))
#
#         # print(word_results_dictionary)
#         summary_file_name = '{0}analysis/summary_{1}_{2.day}_{2.month}_{2.year}_{2.hour}_{2.minute}.csv' \
#             .format(directory, test_type, datetime.now())
#         evaluate_class.create_summary_file(lamda, features_combination, test_file_to_use, train_file_to_use,
#                                            summary_file_name, gradient_class.file_name, comp)
#
#         logging.info('{}: Following Evaluation results for features {}'.
#                      format(time.asctime(time.localtime(time.time())), features_combination))
#         if not comp:
#             logging.info('{}: Evaluation results are: \n {} \n'.format(time.asctime(time.localtime(time.time())),
#                                                                        word_results_dictionary))
#         logging.info('-----------------------------------------------------------------------------------')


if __name__ == "__main__":
    # open log connection
    sub_dirs = config.sub_dirs
    base_directory = os.path.abspath(os.curdir)
    data_dir = os.path.join(base_directory, config.data_dir)
    run_name = config.run_name
    run_dir = datetime.now().strftime("{}_%d_%m_%Y_%H_%M_%S".format(run_name))
    directory = os.path.join(base_directory, config.output_dir, run_dir)
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(directory, sub_dir))
    LOG_FILENAME = datetime.now().strftime(os.path.join(directory, config.log_dir, 'LogFile_%d_%m_%Y_%H_%M.log'))
    logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)

    logging.info('{}: Start running'.format(time.asctime(time.localtime(time.time()))))
    print('{}: Start running'.format(time.asctime(time.localtime(time.time()))))
    df_x = pd.read_pickle(os.path.join(data_dir, config.viewing_data_name))
    df_demo = pd.read_csv(os.path.join(data_dir, config.demo_file_name), index_col=0)
    device_house_dict = pickle.load(open(os.path.join(data_dir, config.device_house_dict), 'rb'))
    house_device_dict = pickle.load(open(os.path.join(data_dir, config.house_device_dict), 'rb'))
    # todo: fix in the code!!!!!!@!#!@#@$#
    device_house_dict = device_house_dict[config.household_id]

    house_device_dict = dict(list(house_device_dict.items())[41209:41210])
    df_x_temp = df_x.loc[df_x[config.x_device_id] == '0000000050f3']

    model = Model(df_demo=df_demo.loc[df_demo[config.household_id] == 1471346],
                  df_x=df_x_temp,
                  house_device=house_device_dict,
                  device_house=device_house_dict,
                  test_df=df_x_temp,
                  test_demo_df=df_demo.loc[df_demo[config.household_id] == 1471346])
    # Baselines - baseline predictions
    # most_common = MLpreceptron.return_common_stupid(df_x['Program Genre'])
    most_common_value = Counter(model.true_genres).most_common()[0][0]
    most_common = [most_common_value] * len(model.true_genres)
    print(most_common)

    preceptron_clf =  MLpreceptron.MulticlasslabelPerceptron(model.train_feature_matrix, model.true_genres,
                                                             list(set(model.true_genres)), model.atomic_tags, 10)

    preceptron_pred = preceptron_clf.predict_genere(model.train_feature_matrix)
    print(preceptron_pred)

    memm = ParametersMEMM(model, 0.1)

    weights_filename = os.path.join(directory, config.weights_file_name)
    results_filename = os.path.join(directory, config.results_file_name)

    memm.gradient_decent(weights_filename, results_filename)
    viterbi = Viterbi(model, memm.w)
    memm_pred = []
    # todo; make avilalble when the sequences dict is merged
    # for seq in model.devices:
    #     pred = viterbi.viterbi_algorithm(seq)
    #     memm_pred.extend(pred)
    seq = list(df_x_temp.df_id)
    pred = viterbi.viterbi_algorithm(seq)
    print(pred)
    accuracy_most_common, recall_most_common, precision_most_common = Evaluate.calc_acc_recall_precision(pred_labels=most_common, true_labels=list(set(model.true_genres)))
    print(accuracy_most_common, recall_most_common, precision_most_common)

    # memm_pred = memm.gradient_decent(weights_filename, results_filename)

