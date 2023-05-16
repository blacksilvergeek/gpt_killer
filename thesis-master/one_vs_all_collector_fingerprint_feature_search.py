import datetime
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_curve, auc, average_precision_score

import linear_class
import signal_similarity
import train_helper
from Utils import dataTools, ClusterUtils

# set training params
nClasses = 1  # Either authorName or not
ratioTrain = 0.6  # Ratio of training samples
ratioValid = 0.2  # Ratio of validation samples (out of the total training)

all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']

# the results from each run
dataDir = 'authorData'  # Data directory
dataFilename = 'authorshipData.mat'  # Data filename
dataPath = os.path.join(dataDir, dataFilename)  # Data path

NO_OF_SPLITS = 10
ONE_VS_ALL_CRASH_FILE = "results/one_vs_all/One_vs_all_fingerprint_feature_search_crash_dump_{0}_{1}.json"
ONE_VS_ONE_RESULTS = "results/one_vs_all/One_vs_all_fingerprint_feature_search_results_{0}.json"


def main():
    Path("results/one_vs_all").mkdir(parents=True, exist_ok=True)

    logging.info('Loading data for One-vs-One.')
    data_general = dataTools.Authorship('poe', 1, 0, dataPath)

    logging.info('Computing Dissimilarty matrix')
    dis_m = signal_similarity.get_dissimilarity_matrix(data=data_general)

    logging.info('Starting training.')

    GCNN_results, svc_results = load_prev_results()

    for idx, name in enumerate(all_author_names):
        most_similar_name = get_most_similar_name(dis_m, idx)
        logging.info('Training GCNN and SVM on {0} v.s. {1}'.format(name, most_similar_name))

        if name not in svc_results.keys():
            svc_results[name] = {}
        if name not in GCNN_results.keys():
            GCNN_results[name] = {}

        try:
            percentages = [0.01, 0.02, 0.05, .1, .15, .20, .25, .30, 0.35, 0.4, 0.5]
            # percentages = [0.05, .1, .15, .20, .25, .30]

            for perc in percentages:
                logging.info('Starting training with feature count: {0}'.format(perc))

                if str(perc) in svc_results[name].keys() and len(svc_results[name][str(perc)]) == 10:
                    logging.info('Skipping SVC for {1} with perc: {0}'.format(perc, name))
                else:
                    svc_results[name][str(perc)] = collect_svc(name, most_similar_name, perc)
                    logging.info('SVM results successfully collected: {0}'.format(svc_results[name][str(perc)]))

                if str(perc) in GCNN_results[name].keys() and len(GCNN_results[name][str(perc)]) == 10:
                    logging.info('Skipping GCNN for {1} with perc: {0}'.format(perc, name))
                else:
                    GCNN_results[name][str(perc)] = collect_gcnn(name, most_similar_name, perc)
                    logging.info('GCNN results successfully collected: {0}'.format(GCNN_results[name][str(perc)]))

                dump_results(GCNN_results, svc_results)

        except BaseException as e:
            dump_crash(GCNN_results, svc_results)
            logging.error("Exception during collecting one-vs-one with feature search", exc_info=True)

            raise e


def load_prev_results(path=ONE_VS_ONE_RESULTS):
    logging.info("Loading previous results")

    gcnn_path = path.format("GCNN")

    gcnn_res = {}
    svc_res = {}

    if Path(gcnn_path).exists():
        with open(gcnn_path, 'r') as f:
            gcnn_res = json.load(f)
        with open(path.format("SVC"), 'r') as f:
            svc_res = json.load(f)

    return gcnn_res, svc_res


def collect_gcnn(name, most_similar_name, perc):
    gcnn_results = []

    data = dataTools.Authorship(name, ratioTrain, ratioValid, dataPath)
    logging.info('Training GCNN on {0}'.format(name))
    h_params = ClusterUtils.load_best_hyperparams(name)

    phi, indices = load_phi(name, data, percentage=perc)

    if indices.shape[0] < 2:
        return [{'acc': 0, 'f1': 0, 'auc': 0, 'prec': 0, 'time': 0}]

    for split_n in range(NO_OF_SPLITS):
        data.get_split(name, ratioTrain, ratioValid)

        data.reduce_dim(indices)

        start = time.perf_counter()

        gcnn = train_helper.train_net(data, h_params, phi=phi)

        end = time.perf_counter()

        gcnn_eval = evaluate_gcnn(gcnn, data)
        gcnn_eval['time'] = start - end
        gcnn_results.append(gcnn_eval)

        logging.info('SPLIT {0}: GCNN results successfully collected: {1}'.format(split_n, gcnn_results[split_n]))

    return gcnn_results


def collect_svc(name, most_similar_name, perc):
    svc_results = []

    data = dataTools.Authorship(name, ratioTrain, ratioValid, dataPath)
    logging.info('Training SVM on {0}'.format(name))

    phi, indices = load_phi(name, data, percentage=perc)

    if indices.shape[0] < 2:
        return [{'acc': 0, 'f1': 0, 'auc': 0, 'prec': 0, 'time': 0}]

    for split_n in range(NO_OF_SPLITS):
        data.get_split(name, ratioTrain, ratioValid)

        data.reduce_dim(indices)

        start = time.perf_counter()

        svc = linear_class.train_svc(data)
        end = time.perf_counter()

        svc_result = evaluate_svc(svc, data)
        svc_result['time'] = start - end

        svc_results.append(svc_result)

        logging.info('SPLIT {0}: SVM results successfully collected: {1}'.format(split_n, svc_result))

    return svc_results


def get_most_similar_name(dis_m, idx):
    np.fill_diagonal(dis_m, 100000)
    most_similar = np.argmin(dis_m[idx])
    most_similar_name = all_author_names[most_similar]
    return most_similar_name


def dump_crash(GCNN_results, svc_results):
    today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    with open(ONE_VS_ALL_CRASH_FILE.format(today, "GCNN"), 'w+') as f:
        json.dump(GCNN_results, f)
    with open(ONE_VS_ALL_CRASH_FILE.format(today, "SVM"), 'w+') as f:
        json.dump(svc_results, f)


def dump_results(GCNN_results, svc_results):
    logging.info("Dumping one-vs-one results to files")

    today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    with open(ONE_VS_ONE_RESULTS.format("GCNN"), 'w+') as f:
        json.dump(GCNN_results, f)
    with open(ONE_VS_ONE_RESULTS.format("SVC"), 'w+') as f:
        json.dump(svc_results, f)

    logging.info("Dumping one-vs-one results SUCCESSFUL!")


def evaluate_svc(arch, data):
    X_valid, y_val = data.getSamples('valid')
    X_valid = preprocessing.scale(X_valid)

    y_hat = arch.predict(X_valid)

    return get_results(y_hat, y_val)


def evaluate_gcnn(arch, data):
    X_valid, y_val = data.getSamples('valid')
    X_valid = X_valid[:, arch.order].unsqueeze(1)

    with torch.no_grad():
        y_hat = arch.archit(X_valid)
        acc = data.evaluate(y_hat, y_val)

        y_hat = np.round(y_hat)
        y_hat = y_hat.squeeze(1).numpy()

        res = get_results(y_hat, y_val.numpy())
        res['acc'] = acc

    return res


def get_results(y_hat, y_val):
    totalErrors = np.sum(np.abs(y_hat - y_val) > 1e-9)
    accuracy = 1 - totalErrors.item() / len(y_val)

    f1 = f1_score(y_val, y_hat)
    fpr, tpr, _ = roc_curve(y_val, y_hat)
    roc_auc = auc(fpr, tpr)
    average_precision = average_precision_score(y_val, y_hat)

    result = {'acc': accuracy, 'f1': f1, 'auc': roc_auc, 'prec': average_precision}

    return result


def load_phi(name, data, phi_matrix_path='EdgeVariGNN_important_words_phi.txt', percentage=0.01, eps=0.0001):
    with open(phi_matrix_path, 'r') as f:
        file = json.load(f)

        phi_whole = np.array(file[name]['phi'])
        data.reduce_dim(file[name]['nodes'])

        indices_to_zero = np.array([x for x in
                                    np.argwhere(
                                        np.abs(phi_whole) - np.max(np.abs(phi_whole)) + percentage * np.max(
                                            np.abs(phi_whole)) < eps)])

        # indices_to_zero = np.array([x for x in
        #                             np.argwhere(
        #                                 np.logical_not(np.isclose(np.abs(phi_whole),
        #                                                           np.max(np.abs(phi_whole)) - percentage * np.max(
        #                                                               np.abs(phi_whole)))))])

        for x, y in indices_to_zero:
            phi_whole[x, y] = 0

        ind_X = []

        for i in range(phi_whole.shape[0]):
            if np.any(phi_whole[i, :]) or np.any(phi_whole[:, i]):
                ind_X.append(i)

        phi = phi_whole[ind_X, :][:, ind_X]

        logging.info(
            "PHI matrix for {1} loaded. Number of dimensions: {0}".format(len(ind_X), name))

        return phi, np.array(ind_X)


if __name__ == '__main__':
    logging.basicConfig(filename='one-vs-all-fingerprint_feature_search.log',
                        format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    main()
