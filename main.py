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
    df_x = pd.read_pickle(os.path.join(data_dir, config.viewing_data_name))
    # df_demo = pd.read_csv(os.path.join(data_dir, config.demo_file_name), index_col=0)
    df_demo = pd.read_pickle(os.path.join(data_dir, config.demo_file_name))
    device_house_dict = pickle.load(open(os.path.join(data_dir, config.device_house_dict), 'rb'))
    house_device_dict = pickle.load(open(os.path.join(data_dir, config.house_device_dict), 'rb'))
    # # todo: fix in the code!!!!!!@!#!@#@$#
    # device_house_dict = device_house_dict[config.household_id]
    hh_ids = pickle.load(open(os.path.join(base_directory, config.models_folder, config.household_list), 'rb'))

    df_x = df_x.loc[df_x[config.x_household_id].isin(hh_ids)]

    threshold = round(len(df_x.groupby(['Device ID'])) * config.train_threshold)
    g = df_x.groupby(['Device ID']).groups
    df_demo = df_demo.drop([config.voter], axis=1)
    dev_threshold = list(g)[threshold-1]
    idx = g[dev_threshold][-1]
    dfx_train = df_x.loc[:idx, ]
    dfx_test = df_x.loc[idx + 1:, ]

    # dfx_train = df_x.loc[:10, ]
    # dfx_test = df_x.loc[11:15, ]

    house_device_dict = dict(list(house_device_dict.items()))
    df_x_temp = df_x

    logging.info('{}: before create model'.format(time.asctime(time.localtime(time.time()))))
    model = Model(df_demo=df_demo,
                  df_x=dfx_train,
                  house_device=house_device_dict,
                  device_house=device_house_dict,
                  model_type='basic',
                  test_df=dfx_test)
    pickle.dump(model, open(os.path.join(directory, config.dict_folder, 'basic_model.pkl'), 'wb'))
    # model = pickle.load(open(os.path.join(base_directory, config.models_folder, 'basic_model.pkl'), 'rb'))

    # Baselines - baseline predictions
    # most_common = MLpreceptron.return_common_stupid(df_x['Program Genre'])

    logging.info('{}: before preceptron'.format(time.asctime(time.localtime(time.time()))))
    preceptron_input = model.train_feature_matrix.tocsc()[:, model.prec_positions]
    preceptron_clf = MLpreceptron.MulticlasslabelPerceptron(preceptron_input, model.true_genres,
                                                            list(set(model.true_genres)), model.atomic_tags, 20)
    model.create_test_matrix()
    preceptron_input = model.test_feature_matrix
    preceptron_pred = preceptron_clf.predict_genere(preceptron_input)  # todo: change to actual test set
    print('preceptron_pred:{}'.format(preceptron_pred))

    # evaluate preceptron
    logging.info('{}: before preceptron evaluate'.format(time.asctime(time.localtime(time.time()))))
    evaluate = Evaluate(model)
    accuracy, recall, precision, f1 = evaluate.calc_acc_recall_precision(pred_labels=preceptron_pred)
    bin_accuracy = evaluate.bin_acc_recall_precision(pred_labels=preceptron_pred)
    evaluate.evaluate_per_dev()
    logging.info('{}: evaluate preceptron: bin_accuracy: {}, accuracy:{}, recall: {}, precision: {}, f1: {}'.format(
        time.asctime(time.localtime(time.time())),
        bin_accuracy, accuracy, recall, precision, f1))
    pickle.dump(evaluate, open(os.path.join(directory, config.results_folder, 'perceptron_evaluate.pkl'), 'wb'))

    logging.info('{}: before_most_common'.format(time.asctime(time.localtime(time.time()))))
    most_common_value = Counter(model.true_genres).most_common()[0][0]
    most_common = [most_common_value] * len(model.test_true_genres)
    print(most_common)

    # top_10 = Counter(model.true_genres).most_common()[0][0]

    # evaluate most common
    evaluate = Evaluate(model)
    accuracy, recall, precision, f1 = evaluate.calc_acc_recall_precision(pred_labels=most_common)
    bin_accuracy = evaluate.bin_acc_recall_precision(pred_labels=most_common)
    evaluate.evaluate_per_dev()
    logging.info('{}: evaluate most_common: bin_accuracy:{}, accuracy:{}, recall:{}, precision:{}, f1: {}'.format(
        time.asctime(time.localtime(time.time())), bin_accuracy, accuracy, recall, precision, f1))
    pickle.dump(evaluate, open(os.path.join(directory, config.results_folder, 'most_common.pkl'), 'wb'))

    logging.info('{}: before memm'.format(time.asctime(time.localtime(time.time()))))
    memm = ParametersMEMM(model, 0.1)

    weights_filename = os.path.join(directory, config.weights_file_name)
    results_filename = os.path.join(directory, config.results_file_name)

    memm.gradient_decent(weights_filename, results_filename)

    logging.info('{}: before viterbi'.format(time.asctime(time.localtime(time.time()))))
    # todo: change not from pickle
    viterbi = Viterbi(model, memm.w, preceptron_clf, most_common_value)
    # viterbi = pickle.load(open(os.path.join(base_directory, config.models_folder, 'viterbi.pkl'), 'rb'))
    memm_pred = []
    # todo; make avilalble when the sequences dict is merged
    for seq in model.dict_nodes_per_device.values():
        if len(seq) < 3:
            continue
        pred = viterbi.viterbi_algorithm(seq)
        memm_pred.extend(pred)

    # seq = model.dict_notes_per_device['00000047d22b']
    # pred = viterbi.viterbi_algorithm(seq)

    pickle.dump(viterbi, open(os.path.join(directory, config.dict_folder, "viterbi.pkl"), 'wb'))
    pickle.dump(memm_pred, open(os.path.join(directory, config.dict_folder, "memm_pred.pkl"), 'wb'))
    # evaluate model - memm
    logging.info('{}: before evaluate'.format(time.asctime(time.localtime(time.time()))))
    evaluate = Evaluate(model)
    accuracy, recall, precision, f1 = evaluate.calc_acc_recall_precision(pred_labels=memm_pred)
    bin_accuracy = evaluate.bin_acc_recall_precision(pred_labels=memm_pred)
    evaluate.evaluate_per_dev()
    print(accuracy, recall, precision, bin_accuracy)
    logging.info('{}: evaluate MEMM: bin_accuracy: {}, accuracy:{}, recall: {}, precision: {}, f1: {}'.format(
        time.asctime(time.localtime(time.time())), bin_accuracy, accuracy, recall, precision, f1))
    pickle.dump(evaluate, open(os.path.join(directory, config.results_folder, 'memm_evaluate.pkl'), 'wb'))
