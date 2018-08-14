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
    model = Model(df_demo=df_demo.loc[df_demo[config.household_id] == 1471346],
                  df_x=df_x.loc[df_x[config.x_device_id] == '0000000050f3'],
                  house_device=house_device_dict,
                  device_house=device_house_dict)
    # Baselines - baseline predictions
    most_common = MLpreceptron.return_common_stupid(df_x['Program Genre'])
    print(most_common)

    # preceptron_clf =  MLpreceptron.MulticlasslabelPerceptron(model.train_feature_matrix, model.true_genres, list(set(model.true_genres)) model.atomic_tags, 10)



    # baseline1_perd - simply most common


    # train_file = directory + 'data/train_small.wtag'
    # test_file = directory + 'data/test_small.wtag'
    # comp_file = directory + 'data/comp_small.words'
    # cv = False
    # comp = False
    # if cv:
    #     cross_validation(train_file)
    # else:
    #     feature_type_dict = {
    #     #                   'all_features': [['feature_100', 'feature_101', 'feature_102', 'feature_103', 'feature_104',
    #     #                                       'feature_105', 'feature_106', 'feature_107', 'feature_108', 'feature_109',
    #     #                                       'feature_110','feature_111']],
    #     #                                       #  ['feature_100', 'feature_101', 'feature_102', 'feature_103', 'feature_104',
    #     #                                       # 'feature_105', 'feature_106', 'feature_107'],
    #     #                                       #  ['feature_100', 'feature_101','feature_102', 'feature_103','feature_104',
    #     #                                       #   'feature_105', 'feature_106', 'feature_107','feature_108', 'feature_109'],
    #     #                                       #  ['feature_100', 'feature_101', 'feature_102', 'feature_103', 'feature_104',
    #     #                                       # 'feature_105', 'feature_106', 'feature_107','feature_110','feature_111']]}
    #                          'basic_model': [['feature_100', 'feature_103', 'feature_104']]}
    #     feature_type_dict = {
    #         'all_features': [['feature_100', 'feature_101', 'feature_102', 'feature_103', 'feature_104',
    #                           'feature_105', 'feature_106', 'feature_107', 'feature_108', 'feature_109',
    #                           'feature_110', 'feature_111','feature_112','feature_113','feature_114']]}
    #         #'basic_model': [['feature_100', 'feature_103', 'feature_104']]}
    #
    # start_time = time.time()
    # main()
    # # lambda_list = [10.0]
    # # for lambda_ in lambda_list:
    # #     if not comp:
    # #         for feature_type_name, feature_type_list in feature_type_dict.items():
    # #             main(train_file, test_file, 'test', feature_type_list, lambda_, comp)
    # #     else:
    # #         for feature_type_name, feature_type_list in feature_type_dict.items():
    # #             main(comp_file, test_file, 'test', feature_type_list, lambda_, comp)
    # run_time = (time.time() - start_time) / 60.0
    # print("{}: Finish running with lamda: {}. Run time is: {} minutes".
    #       format(time.asctime(time.localtime(time.time())), lambda_, run_time))
    # logging.info('{}: Finish running with lambda:{} . Run time is: {} minutes'.
    #              format(time.asctime(time.localtime(time.time())), lambda_, run_time))
