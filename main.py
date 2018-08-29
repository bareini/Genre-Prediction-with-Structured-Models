import logging
import os
import pickle
import time
from collections import Counter
from datetime import datetime

# from Viterbi_NLP import viterbi
import pandas as pd

import MLpreceptron
import config
# from viterbi_ML import viterbi
from evaluate import Evaluate
from memm_parameters_learn import ParametersMEMM
from model import Model
from viterbi import Viterbi

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
    # df_demo = pd.read_csv(os.path.join(data_dir, config.demo_file_name), index_col=0)
    df_demo = pd.read_pickle(os.path.join(data_dir, config.demo_file_name))
    df_demo = df_demo.drop([config.voter], axis=1)
    device_house_dict = pickle.load(open(os.path.join(data_dir, config.device_house_dict), 'rb'))
    house_device_dict = pickle.load(open(os.path.join(data_dir, config.house_device_dict), 'rb'))
    hh_ids = pickle.load(open(os.path.join(base_directory, config.models_folder, config.household_list), 'rb'))

    house_device_dict = dict(list(house_device_dict.items()))

    for type_ in config.model_types_to_run:
        if type_ in [config.basic, config.advanced, config.advanced_2, config.super_advanced]:
            df_x = pd.read_pickle(os.path.join(data_dir, config.viewing_data_name))

            df_x = df_x.loc[df_x[config.x_household_id].isin(hh_ids)]
            threshold = round(len(df_x.groupby(['Device ID'])) * config.train_threshold)
            g = df_x.groupby(['Device ID']).groups
            dev_threshold = list(g)[threshold - 1]
            idx = g[dev_threshold][-1]
            dfx_train = df_x.loc[:idx, ]
            dfx_test = df_x.loc[idx + 1:, ]

            logging.info('{}: before create model: {}'.format(time.asctime(time.localtime(time.time())), type_))
            model = Model(df_demo=df_demo,
                          df_x=dfx_train,
                          house_device=house_device_dict,
                          device_house=device_house_dict,
                          model_type=type_,
                          test_df=dfx_test)
            model_name = '{}_model.pkl'.format(type_)
            pickle.dump(model, open(os.path.join(directory, config.dict_folder, model_name), 'wb'))
            # model = pickle.load(open(os.path.join(base_directory, config.models_folder, 'advanced2_model.pkl'), 'rb'))

        else:
            df_x = None
            dfx_train = None
            dfx_test = None
            # load clustering dataframe
            df_full = pd.read_pickle(os.path.join(data_dir, config.full_viewing))
            df_full = df_full.loc[df_full[config.x_household_id].isin(hh_ids)]

            df_clusters = pd.read_pickle(os.path.join(data_dir, config.clusters_df))
            df_clusters = df_clusters.loc[df_clusters[config.x_household_id].isin(hh_ids)]

            threshold = round(len(df_clusters.groupby(['Device ID'])) * config.train_threshold)
            g = df_clusters.groupby(['Device ID']).groups
            dev_threshold = list(g)[threshold - 1]
            idx = g[dev_threshold][-1]
            df_clusters_train = df_clusters.loc[:idx, ]
            df_clusters_test = df_clusters.loc[idx + 1:, ]

            inner_type = config.inner_clustered_type

            logging.info('{}: before create clusters model'.format(time.asctime(time.localtime(time.time()))))
            model = Model(df_demo=df_demo,
                          df_x=df_clusters_train,
                          house_device=house_device_dict,
                          device_house=device_house_dict,
                          model_type=inner_type,
                          test_df=df_clusters)
            model_name = '{}_cluster_model.pkl'.format(inner_type)
            pickle.dump(model, open(os.path.join(directory, config.dict_folder, model_name), 'wb'))
            # model = pickle.load(open(os.path.join(base_directory, config.models_folder, model_name), 'rb'))

            logging.info('{}: before clusters_memm'.format(time.asctime(time.localtime(time.time()))))
            memm = ParametersMEMM(model, 0.1)

            weights_filename = os.path.join(directory, "{}_{}.txt".format(config.weights_file_name, inner_type))
            results_filename = os.path.join(directory, "{}_{}.txt".format(config.results_file_name, inner_type))

            memm.gradient_decent(weights_filename, results_filename)

            logging.info('{}: before clusters viterbi'.format(time.asctime(time.localtime(time.time()))))
            # todo: change not from pickle
            clusters_viterbi = Viterbi(model, memm.w)
            memm_pred = []
            memm_pred_nodes =[]
            for seq in model.dict_nodes_per_device.values():
                pred = clusters_viterbi.viterbi_algorithm(seq)
                memm_pred.extend(pred)
                memm_pred_nodes.extend(seq)

            pickle.dump((memm_pred, memm_pred_nodes), open(os.path.join(directory, config.dict_folder, "viterbi.pkl"),
                                                           'wb'))
            # memm_pred, memm_pred_nodes = pickle.load(open(
            #     os.path.join(base_directory, config.models_folder, 'viterbi.pkl'), 'rb'))
            df_full.loc[memm_pred_nodes, config.x_clustered_genre] = memm_pred
            df_full[config.x_clustered_prev_1] = df_full.groupby(config.x_device_id)[
                config.x_clustered_genre].shift(1).fillna('*')
            df_full[config.x_clustered_prev_2] = df_full.groupby(config.x_device_id)[
                 config.x_clustered_genre].shift(2).fillna('**')
            df_full.loc[(df_full[config.x_clustered_prev_2] == '**')
                                   & (df_full[config.x_clustered_prev_1] != '*'), config.x_clustered_prev_2] = '*'
            # df_full.loc[:, config.x_clustered_advanced_1] = df_full.loc[
            #     df_full[config.x_clustered_advanced_1_loc],
            #     config.x_clustered_genre
            # ]
            # df_full.loc[:, config.x_clustered_advanced_2] = df_full.loc[
            #     df_full[config.x_clustered_advanced_2_loc],
            #     config.x_clustered_genre
            # ]

            df_full_train = df_full.loc[:idx, ]
            df_full_test = df_full.loc[idx + 1:, ]

            # create full model
            logging.info('{}: before create model'.format(time.asctime(time.localtime(time.time()))))
            df_full[config.x_clustered_genre] = memm_pred
            model = Model(df_demo=df_demo,
                          df_x=df_full_train,
                          house_device=house_device_dict,
                          device_house=device_house_dict,
                          model_type=type_,
                          test_df=df_full_test)
            model_name = '{}_model.pkl'.format(type_)
            pickle.dump(model, open(os.path.join(directory, config.dict_folder, model_name), 'wb'))

        # model = pickle.load(open(os.path.join(base_directory, config.models_folder, 'basic_model.pkl'), 'rb'))

    # Baselines - baseline predictions
    # most_common = MLpreceptron.return_common_stupid(df_x['Program Genre'])
    # num_of_iter = config.num_of_iters
    # logging.info('{}: before preceptron'.format(time.asctime(time.localtime(time.time()))))
    # preceptron_input = model.train_feature_matrix.tocsc()[:, model.prec_positions]
    # preceptron_clf = MLpreceptron.MulticlasslabelPerceptron(preceptron_input, model.true_genres,
    #                                                         list(set(model.true_genres)), model.atomic_tags, num_of_iter)
    # model.create_test_matrix()
    # preceptron_input = model.test_feature_matrix
    # preceptron_pred = preceptron_clf.predict_genere(preceptron_input)  # todo: change to actual test set
    # print('preceptron_pred:{}'.format(preceptron_pred))
    #
    # # evaluate preceptron
    # logging.info('{}: before preceptron evaluate_{}'.format(time.asctime(time.localtime(time.time())), num_of_iter))
    # evaluate = Evaluate(model)
    # accuracy, recall, precision, f1 = evaluate.calc_acc_recall_precision(pred_labels=preceptron_pred)
    # bin_accuracy = evaluate.bin_acc_recall_precision(pred_labels=preceptron_pred)
    # evaluate.evaluate_per_dev()
    # logging.info('{}: evaluate preceptron_{}: bin_accuracy: {}, accuracy:{}, recall: {}, precision: {}, f1: {}'.format(
    #     time.asctime(time.localtime(time.time())), num_of_iter,
    #     bin_accuracy, accuracy, recall, precision, f1))
    # pickle.dump(evaluate, open(
    #     os.path.join(directory, config.results_folder, 'perceptron_evaluate_{}.pkl'.format(num_of_iter)), 'wb'))
    #
    # logging.info('{}: before_most_common'.format(time.asctime(time.localtime(time.time()))))
    # most_common_value = Counter(model.true_genres).most_common()[0][0]
    # most_common = [most_common_value] * len(model.test_true_genres)
    # print(most_common)

        # top_10 = Counter(model.true_genres).most_common()[0][0]

        # # evaluate most common
        # evaluate = Evaluate(model)
        # accuracy, recall, precision, f1 = evaluate.calc_acc_recall_precision(pred_labels=most_common)
        # bin_accuracy = evaluate.bin_acc_recall_precision(pred_labels=most_common)
        # evaluate.evaluate_per_dev()
        # logging.info('{}: evaluate most_common: bin_accuracy:{}, accuracy:{}, recall:{}, precision:{}, f1: {}'.format(
        #     time.asctime(time.localtime(time.time())), bin_accuracy, accuracy, recall, precision, f1))
        # pickle.dump(evaluate, open(os.path.join(directory, config.results_folder, 'most_common.pkl'), 'wb'))

        logging.info('{}: before memm'.format(time.asctime(time.localtime(time.time()))))
        memm = ParametersMEMM(model, 0.1)

        weights_filename = os.path.join(directory, "{}_{}.txt".format(config.weights_file_name, type_))
        results_filename = os.path.join(directory, "{}_{}.txt".format(config.results_file_name, type_))

        memm.gradient_decent(weights_filename, results_filename)

        logging.info('{}: before viterbi'.format(time.asctime(time.localtime(time.time()))))
        # todo: change not from pickle
        viterbi = Viterbi(model, memm.w)
        # viterbi = pickle.load(open(os.path.join(base_directory, config.models_folder, 'viterbi.pkl'), 'rb'))
        memm_pred = []
        memm_pred_nodes = []

        # todo; make avilalble when the sequences dict is merged
        for seq in model.dict_nodes_per_device.values():
            # if len(seq) < 3:
            #     continue
            pred = viterbi.viterbi_algorithm(seq)
            memm_pred.extend(pred)
            memm_pred_nodes.extend(seq)

        pickle.dump((memm_pred, memm_pred_nodes), open(os.path.join(directory, config.dict_folder, "viterbi.pkl"),
                                                           'wb'))


        # seq = model.dict_notes_per_device['00000047d22b']
        # pred = viterbi.viterbi_algorithm(seq)

        pickle.dump(viterbi, open(os.path.join(directory, config.dict_folder, "viterbi.pkl"), 'wb'))
        pickle.dump(memm_pred, open(os.path.join(directory, config.dict_folder, "memm_pred_{}.pkl".format(type_)), 'wb'))
        # evaluate model - memm
        logging.info('{}: before evaluate'.format(time.asctime(time.localtime(time.time()))))
        evaluate = Evaluate(model)
        accuracy, recall, precision, f1 = evaluate.calc_acc_recall_precision(pred_labels=memm_pred)
        bin_accuracy = evaluate.bin_acc_recall_precision(pred_labels=memm_pred)
        evaluate.evaluate_per_dev()
        print(accuracy, recall, precision, bin_accuracy)
        logging.info('{}: evaluate MEMM: bin_accuracy: {}, accuracy:{}, recall: {}, precision: {}, f1: {}'.format(
            time.asctime(time.localtime(time.time())), bin_accuracy, accuracy, recall, precision, f1))

        pickle.dump(evaluate,
                    open(os.path.join(directory, config.results_folder, 'memm_evaluate_{}.pkl'.format(type_)), 'wb'))
