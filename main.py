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

    threshold = round(len(df_x.groupby(['Device ID'])) * config.train_threshold)
    g = df_x.groupby(['Device ID']).groups
    df_demo = df_demo.drop([config.voter], axis=1)
    dev_threshold = list(g)[threshold]
    idx = g[dev_threshold][-1]
    dfx_train = df_x.loc[:idx, ]
    dfx_test = df_x.loc[idx + 1:, ]

    house_device_dict = dict(list(house_device_dict.items()))
    df_x_temp = df_x

    logging.info('{}: befor_create_model'.format(time.asctime(time.localtime(time.time()))))
    model = Model(df_demo=df_demo,
                  df_x=dfx_train,
                  house_device=house_device_dict,
                  device_house=device_house_dict,
                  test_df=dfx_test)
    pickle.dump(model, open(os.path.join(directory, config.dict_folder, 'model.pkl'),'wb'))
    # model = pickle.load(open(os.path.join(base_directory, config.models_folder, 'model.pkl'),'rb'))

    # Baselines - baseline predictions
    # most_common = MLpreceptron.return_common_stupid(df_x['Program Genre'])
    logging.info('{}: before_most_common'.format(time.asctime(time.localtime(time.time()))))
    most_common_value = Counter(model.true_genres).most_common()[0][0]
    most_common = [most_common_value] * len(model.true_genres)
    print(most_common)

    logging.info('{}: before_preceptron'.format(time.asctime(time.localtime(time.time()))))
    preceptron_clf =  MLpreceptron.MulticlasslabelPerceptron(model.train_feature_matrix, model.true_genres,
                                                             list(set(model.true_genres)), model.atomic_tags, 10)

    preceptron_pred = preceptron_clf.predict_genere(model.train_feature_matrix)
    print(preceptron_pred)

    logging.info('{}: before_memm'.format(time.asctime(time.localtime(time.time()))))
    memm = ParametersMEMM(model, 0.1)

    weights_filename = os.path.join(directory, config.weights_file_name)
    results_filename = os.path.join(directory, config.results_file_name)

    memm.gradient_decent(weights_filename, results_filename)

    logging.info('{}: before_viterbi'.format(time.asctime(time.localtime(time.time()))))
    viterbi = Viterbi(model, memm.w)
    memm_pred = []
    # todo; make avilalble when the sequences dict is merged
    for seq in model.devices_gen:
        pred = viterbi.viterbi_algorithm(seq)
        memm_pred.extend(pred)
    # seq = list(df_x_temp.df_id)
    # pred = viterbi.viterbi_algorithm(seq)

    logging.info('{}: before_evaluate'.format(time.asctime(time.localtime(time.time()))))
    evaluate = Evaluate(model)
    accuracy, recall, precision = evaluate.calc_acc_recall_precision(pred_labels=most_common)
    bin_accuracy = evaluate.bin_acc_recall_precision(pred_labels=most_common)
    evaluate.evaluate_per_dev()
    print(accuracy, recall, precision, bin_accuracy)

