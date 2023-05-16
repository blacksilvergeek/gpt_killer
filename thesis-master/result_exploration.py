import json
import os
from os import path
from pathlib import Path

import matplotlib
import torch

import train_helper as th
from Utils import graphTools

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})
RS = 123

# %%##################################################################
# Function for summarizing training results.
all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']


def summarize_training_results(train_result):
    author_best_acc = {}
    for author_name in train_result.keys():
        author_results = train_result[author_name]

        means = []
        stds = []

        for comb in author_results.keys():
            mean = np.mean(author_results[comb])
            std = np.std(author_results[comb])

            means.append(mean)
            stds.append(std)

        # best_comb = max(author_results.items(), key=operator.itemgetter(1))[0]
        # best_acc = max(author_results.items(), key=operator.itemgetter(1))[1]

        best_acc = max(means)
        index_of_best = means.index(best_acc)

        std = stds[index_of_best]
        best_comb = list(author_results.keys())[index_of_best]

        # best_combs.append(best_comb)
        author_best_acc[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

    return author_best_acc


def analyse_GNN_results(train_result):
    means = []
    stds = []

    for comb in train_result.keys():
        mean = np.mean(train_result[comb])
        std = np.std(train_result[comb])

        means.append(mean)
        stds.append(std)

    best_acc = max(means)
    index_of_best = means.index(best_acc)
    std = stds[index_of_best]
    best_comb = list(train_result.keys())[index_of_best]

    return best_acc, best_comb, std


def analyse_GNN_results_extra(train_result):
    means_acc = []
    means_f1 = []
    means_auc = []
    stds = []

    for comb in train_result.keys():
        if not train_result[comb]:
            continue

        acc_array = np.array(train_result[comb]['acc'], dtype=np.float)
        f1_arr = np.array(train_result[comb]['f1'], dtype=np.float)

        mean_acc = np.mean(acc_array)
        mean_f1 = np.mean(f1_arr)

        if 'auc' in train_result[comb]:
            mean_auc = np.mean(train_result[comb]['auc'])
        else:
            mean_auc = 0

        std = np.std(acc_array)

        means_acc.append(mean_acc)
        means_f1.append(mean_f1)
        means_auc.append(mean_auc)
        stds.append(std)

    best_acc = max(means_acc)
    best_f1 = max(means_f1)
    best_auc = max(means_auc)

    index_of_best = means_acc.index(best_acc)
    std = stds[index_of_best]
    best_comb = list(train_result.keys())[index_of_best]

    return best_acc, best_comb, std, best_f1, best_auc


def analyse_result_object(train_result):
    means_acc = []
    means_f1 = []
    means_auc = []
    stds = []

    for comb in train_result.keys():
        if not train_result[comb]:
            continue

        mean_acc = np.mean([x['acc'] for x in train_result[comb]])
        mean_f1 = np.mean([x['f1'] for x in train_result[comb]])
        mean_auc = np.mean([x['auc'] for x in train_result[comb]])
        std = np.std([x['acc'] for x in train_result[comb]])

        means_acc.append(mean_acc)
        means_f1.append(mean_f1)
        means_auc.append(mean_auc)
        stds.append(std)

        best_acc = max(means_acc)
        best_f1 = max(means_f1)
        best_auc = max(means_auc)

        index_of_best = means_acc.index(best_acc)
        std = stds[index_of_best]
        best_comb = list(train_result.keys())[index_of_best]

    return best_acc, best_comb, std, best_f1, best_auc


def prepare_for_training(data, order):
    data = torch.from_numpy(data).double()

    data_ordered = data[:, order].unsqueeze(1)

    return data_ordered


def fashion_scatter(x, colors, name=""):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    plt.title(name)
    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def analyze_linear_results(json):
    results = {}
    results_std = {}

    for author in all_author_names:
        temp = json[author]

        if isinstance(temp, dict):
            means_knn = [np.mean(v) for k, v in temp.items()]
            max_acc = np.max(means_knn)
            std = [np.std(v) for k, v in temp.items()][0]
        else:
            max_acc = np.mean(temp)
            std = np.std(temp)

        results[author] = max_acc
        results_std[author] = std

    return results, results_std


# %%##################################################################
# compare the results of 1 and 2 layer feedforward networks

with open('1_feedforward_results_BCLoss.txt', 'r') as f:
    train_result = json.load(f)
    feedforward_results = summarize_training_results(train_result)

with open('2_feedforward_results_BCLoss.txt', 'r') as f:
    train_result = json.load(f)
    two_feedforward_results = summarize_training_results(train_result)

best_count = []

for author_name in feedforward_results.keys():
    if feedforward_results[author_name]['best_acc'] >= two_feedforward_results[author_name]['best_acc']:
        best_count.append(1)
    else:
        best_count.append(2)

# %%##################################################################
# group authors into bins
first = []
second = []
third = []

for author_name in two_feedforward_results.keys():
    acc = two_feedforward_results[author_name]['best_acc']

    if acc < 0.9:
        first.append(author_name)
    elif 0.9 <= acc < 0.95:
        second.append(author_name)
    elif 0.95 <= acc:
        third.append(author_name)

# %%##################################################################
# Anaylse linear model results. Compare to GCNN resutls.
all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']

knn_file = open('results/knn_results.txt', 'r')
svm_file = open('results/svm_results.txt', 'r')

knn_search_results = json.load(knn_file)
svm_search_results = json.load(svm_file)

knn_results, knn_std = analyze_linear_results(knn_search_results)
svm_results, svm_std = analyze_linear_results(svm_search_results)

knn_file.close()
svm_file.close()

# %%##################################################################
# Anaylse GCNN results on random SO

all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']

BASE_FILE_NAME_RANDOM_SO = 'results/random_so/GNN_Polynomial_random_so_results_'
GCNN_random_so_results = dict.fromkeys(all_author_names, {'best_acc': None, 'std': None, 'best_comb': None})

for author_name in all_author_names:
    if not path.exists("{0}{1}.txt".format(BASE_FILE_NAME_RANDOM_SO, author_name)):
        continue

    with open('{0}{1}.txt'.format(BASE_FILE_NAME_RANDOM_SO, author_name), 'r') as f:
        train_result = json.load(f)
        GCNN_random_so_results[author_name] = {'best_acc': np.mean(train_result[list(train_result.keys())[0]]),
                                               "best_comb": list(train_result.keys())[0],
                                               "std": np.std(train_result[list(train_result.keys())[0]])}

# %%##################################################################
# Anaylse GCNN results on classification by Nationality and Gender

all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']

FILE_NAME_GENDER = 'results/gender/GNN_Polynomial_gender_results_20200312161323.txt'

with open(FILE_NAME_GENDER, 'r') as f:
    train_result = json.load(f)

    means_acc = []
    means_f1 = []
    means_auc = []
    stds = []

    for comb in train_result.keys():
        mean_acc = np.mean(train_result[comb]['acc'])
        mean_f1 = np.mean(train_result[comb]['f1'])
        mean_auc = np.mean(train_result[comb]['auc'])
        std = np.std(train_result[comb]['acc'])

        means_acc.append(mean_acc)
        means_f1.append(mean_f1)
        means_auc.append(mean_auc)
        stds.append(std)

    best_acc = max(means_acc)
    best_f1 = max(means_f1)
    best_auc = max(means_auc)

    index_of_best = means_acc.index(best_acc)
    std = stds[index_of_best]
    best_comb = list(train_result.keys())[index_of_best]

    gender_results = {'best_acc': best_acc,
                      'best_f1': best_f1,
                      'best_auc': best_auc,
                      'best_comb': best_comb,
                      'std': std}

    # %%##################################################################
    # Nationality classification on sparse fp (parameter search)

    with open('results/gender/gender_phi_results_GCNN_0.4_search.txt', 'r') as f:
        train_result = json.load(f)
        result = analyse_result_object(train_result)


    # %%##################################################################
    # Export nationality results
    def get_author_avg(data):
        acc = np.average([item['acc'] for item in data])
        f1 = np.average([item['f1'] for item in data])
        auc = np.average([item['auc'] for item in data])
        prec = np.average([item['prec'] for item in data])
        std = np.std([item['acc'] for item in data])

        return acc, f1, auc, prec, std


    NATIONALITY_GNN = 'results/nationality/GCNN_nationality_results_20200310112333.txt'
    NATIONALITY_EDG = 'results/nationality/EdgeNet_nationality_search_results_.txt'
    NATIONALITY_SVM = 'results/nationality/svm_results_nationality.txt'
    NATIONALITY_HYB = 'results/nationality/Edge_GCNN_nationality_results_.txt'
    NATIONALITY_PHI = 'results/nationality/GCNN_nationality_phi_results_03.txt'
    NATIONALITY_PHI_SVM = 'results/nationality/SVC_nationality_phi_results_03.txt'

    files = {'Edgenet': NATIONALITY_EDG, 'GCNN': NATIONALITY_GNN, 'Edge-GCNN': NATIONALITY_HYB, 'SVM': NATIONALITY_SVM,
             'GCNN PHI': NATIONALITY_PHI, 'SVM PHI': NATIONALITY_PHI_SVM}

    nationality_results = {}

    nationality_box_plot_data = []
    nationality_box_plot_data_fp = []

    for k, v in files.items():
        with open(v, 'r') as f:
            train_result = json.load(f)

            if k == 'SVM':
                nationality_results[k] = np.mean(train_result)
                nationality_box_plot_data.append(train_result)

            elif k == 'GCNN PHI' or k == 'SVM PHI':
                best_acc, best_comb, std, best_f1, best_auc = get_author_avg(train_result)
                nationality_results[k] = {'acc': best_acc, "f1": best_f1, 'auc': best_auc, "std": std}
                nationality_box_plot_data_fp.append([item['acc'] for item in train_result])

            else:
                all_acc = train_result[list(train_result.keys())[0]]['acc']
                nationality_results[k] = {'acc': np.mean(all_acc),
                                          'f1': np.mean(train_result[list(train_result.keys())[0]]['f1']),
                                          'auc': np.mean(train_result[list(train_result.keys())[0]]['auc']),
                                          "std": np.std(all_acc)}
                nationality_box_plot_data.append(all_acc)

    df_nationality = pd.DataFrame(nationality_results)
    import plot_tools

    labels = (list(files.keys())[-2:])
    # plot_tools.box_plot_with_dist(labels, nationality_box_plot_data_fp, 'Classification by Nationality (FP)',
    #                               'Accuracy')

    df_nationality = df_nationality.round(2)
    # df_nationality.to_latex('results/latex_tables/nationality_comparison.tex')
    # %%##################################################################
    # Analyse GAT and GCAT results on AA

    all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                        'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving',
                        'poe',
                        'jewett', 'wharton']

    FILE_NAME_GAT = 'results/AA_GAT/Autorship_attribution_GAT_results_'
    FILE_NAME_GCAT = 'results/AA_GCAT/Autorship_attribution_GCAT_results_'

    GAT_RES = {}
    GCAT_RES = {}
    EDGE_RES = {}

    for author_name in all_author_names:
        if not (path.exists("{0}{1}.txt".format(FILE_NAME_GAT, author_name)) and path.exists(
                "{0}{1}.txt".format(FILE_NAME_GCAT, author_name))):
            GAT_RES[author_name] = {'best_acc': None, "best_comb": None, "std": None}
            GCAT_RES[author_name] = {'best_acc': None, "best_comb": None, "std": None}
            EDGE_RES[author_name] = {'best_acc': None, "best_comb": None, "std": None}

            continue

        with open('{0}{1}.txt'.format(FILE_NAME_GAT, author_name), 'r') as f:
            train_result = json.load(f)
            GAT_RES[author_name] = train_result
            best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(GAT_RES[author_name])
            GAT_RES[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

        with open('{0}{1}.txt'.format(FILE_NAME_GCAT, author_name), 'r') as f:
            train_result = json.load(f)
            GCAT_RES[author_name] = train_result
            best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(GCAT_RES[author_name])
            GCAT_RES[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

    # %%##################################################################
    # Analyse Edgenet results

    all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                        'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving',
                        'poe',
                        'jewett', 'wharton']

    FILE_NAME_EDGE_GCNN_BEST = 'results/edgenet_authorship_attribution/best_GCNN/Autorship_attribution_edgenet_results_'
    FILE_NAME_EDGE_search = 'results/edgenet_authorship_attribution/edgnet_grid_search/Autorship_attribution_edgenet_search_results_'
    EDGE_RES = {}
    EDGE_GRID_RES = {}

    for author_name in all_author_names:
        if not path.exists("{0}{1}.txt".format(FILE_NAME_EDGE_GCNN_BEST, author_name)):
            EDGE_RES[author_name] = {'best_acc': 0, "best_comb": 'None', "std": 0}

            continue

        with open('{0}{1}.txt'.format(FILE_NAME_EDGE_GCNN_BEST, author_name), 'r') as f:
            train_result = json.load(f)
            for key, v in train_result.items():
                cur = train_result[key]
                for k2, v2 in cur.items():
                    res = []
                    for v3 in v2:
                        if v3 is not None:
                            res.append(v3)
                    # train_result[key][k2] = [0 if v is None else v for v in v2]
                    train_result[key][k2] = res

            EDGE_RES[author_name] = train_result
            best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(EDGE_RES[author_name])
            EDGE_RES[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

    for author_name in all_author_names:
        if not path.exists("{0}{1}.txt".format(FILE_NAME_EDGE_search, author_name)):
            EDGE_GRID_RES[author_name] = {'best_acc': 0, "best_comb": 'None', "std": 0}

            continue
        with open('{0}{1}.txt'.format(FILE_NAME_EDGE_search, author_name), 'r') as f:
            train_result = json.load(f)
            EDGE_GRID_RES[author_name] = train_result
            best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(EDGE_GRID_RES[author_name])
            EDGE_GRID_RES[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

# %%##################################################################
# Anaylse results from using Phi matrix as a Shift operator

all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']

BASE_FILE_NAME_PHI = 'results/phi/GNN_Polynomial_phi_results_'
BASE_FILE_NAME_PHI_perc = 'results/phi_perc/GNN_Polynomial_phi_perc_results_'

GCNN_PHI_acc_results_all = {}
GCNN_PHI_perc_results_all = {}

GCNN_PHI_results = {}
GCNN_PHI_perc_results = {}

phi_acc = open('EdgeVariGNN_important_words_phi_accuracy3.txt', 'r')
phi_perc = open('EdgeVariGNN_important_words_phi.txt', 'r')

train_result_phi_acc = json.load(phi_acc)
train_result_phi_perc = json.load(phi_perc)

for author_name in all_author_names:
    if not path.exists("{0}{1}.txt".format(BASE_FILE_NAME_PHI, author_name)):
        continue

    with open('{0}{1}.txt'.format(BASE_FILE_NAME_PHI, author_name), 'r') as f:
        train_result = json.load(f)
        GCNN_PHI_acc_results_all[author_name] = train_result
        best_acc, best_comb, std = analyse_GNN_results(GCNN_PHI_acc_results_all[author_name])
        GCNN_PHI_results[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

    GCNN_PHI_results[author_name]['no_of_words'] = np.count_nonzero(train_result_phi_acc[author_name]['phi'])

    if not path.exists("{0}{1}.txt".format(BASE_FILE_NAME_PHI_perc, author_name)):
        continue

    with open('{0}{1}.txt'.format(BASE_FILE_NAME_PHI_perc, author_name), 'r') as f:
        train_result = json.load(f)
        GCNN_PHI_perc_results_all[author_name] = train_result
        best_acc, best_comb, std = analyse_GNN_results(GCNN_PHI_perc_results_all[author_name])
        GCNN_PHI_perc_results[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

    GCNN_PHI_perc_results[author_name]['no_of_words'] = len(train_result_phi_perc[author_name]['indices'])

phi_acc.close()
phi_perc.close()

# %%##################################################################
# Compare GCNN results with FF results

with open('results/2_feedforward_results_BCLoss.txt', 'r') as f:
    train_result = json.load(f)
    two_feedforward_results = summarize_training_results(train_result)

BASE_FILE_NAME_1L = 'results/bc_loss_gnn/GNN_Polynomial_results_'
BASE_FILE_NAME_2L = 'results/Gnn_2_layers/GNN_Polynomial_2layers_results_'

GCNN_results = {}
GCNN_2layer_results = {}
GCNN_2layer_results_all = {}
GCNN_results_all = {}

for author_name in two_feedforward_results.keys():
    if not path.exists("{0}{1}.txt".format(BASE_FILE_NAME_1L, author_name)):
        continue

    with open('results/bc_loss_gnn/GNN_Polynomial_results_{0}.txt'.format(author_name), 'r') as f:
        train_result = json.load(f)
        GCNN_results_all[author_name] = train_result

    with open('{0}{1}.txt'.format(BASE_FILE_NAME_2L, author_name), 'r') as f:
        train_result = json.load(f)
        GCNN_2layer_results_all[author_name] = train_result

for author_name in GCNN_results_all.keys():
    best_acc, best_comb, std = analyse_GNN_results(GCNN_results_all[author_name])
    GCNN_results[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

    best_acc, best_comb, std = analyse_GNN_results(GCNN_2layer_results_all[author_name])
    GCNN_2layer_results[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

best_count_acc = []
best_count_std = []

for author_name in GCNN_results.keys():
    # compare accuracies
    if GCNN_results[author_name]['best_acc'] >= two_feedforward_results[author_name]['best_acc']:
        best_count_acc.append(1)
    else:
        best_count_acc.append(2)

        # compare stds
    if GCNN_results[author_name]['std'] >= two_feedforward_results[author_name]['std']:
        best_count_std.append(1)
    else:
        best_count_std.append(2)

# create a dataframe with info from GCNN and FF2
model_comparison_df = pd.DataFrame.from_dict(GCNN_results)
model_comparison_df = model_comparison_df.T

model_comparison_df['Feed forward'] = [v['best_acc'] for k, v in two_feedforward_results.items()]
model_comparison_df['best_comb_ff'] = [v['best_comb'] for k, v in two_feedforward_results.items()]

# add linear model results
model_comparison_df['SVM'] = [v for k, v in svm_results.items()]
model_comparison_df['SVM std'] = [v for k, v in svm_std.items()]
model_comparison_df['KNN'] = [v for k, v in knn_results.items()]
model_comparison_df['KNN std'] = [v for k, v in knn_std.items()]

model_comparison_df['GCNN with random SO'] = [v['best_acc'] for v in GCNN_random_so_results.values()]

# add 2 layer GCNN results
model_comparison_df['2l GCNN'] = [v['best_acc'] for k, v in GCNN_2layer_results.items()]
model_comparison_df['2l GCNN std'] = [v['std'] for k, v in GCNN_2layer_results.items()]
model_comparison_df['best_combination_2layer'] = [v['best_comb'] for k, v in GCNN_2layer_results.items()]

# add GCNN results using PHI as SO
model_comparison_df['GCNN with extracted SO (Acc)'] = [v['best_acc'] for k, v in GCNN_PHI_results.items()]
model_comparison_df['GCNN with extracted SO (Perc)'] = [v['best_acc'] for k, v in GCNN_PHI_perc_results.items()]

model_comparison_df['non_zero_el_acc_phi'] = [v['no_of_words'] for k, v in GCNN_PHI_results.items()]
model_comparison_df['non_zero_el_perc_phi'] = [v['no_of_words'] for k, v in GCNN_PHI_perc_results.items()]

model_comparison_df['GAT'] = [v['best_acc'] for k, v in GAT_RES.items()]
model_comparison_df['GCAT'] = [v['best_acc'] for k, v in GCAT_RES.items()]
model_comparison_df['EdgeNets (best GCNN)'] = [v['best_acc'] for k, v in EDGE_RES.items()]
model_comparison_df['EdgeNets (best GCNN) std'] = [v['std'] for k, v in EDGE_RES.items()]
model_comparison_df['EdgeNets (HP search)'] = [v['best_acc'] for k, v in EDGE_GRID_RES.items()]
model_comparison_df['EdgeNets (HP search) std'] = [v['std'] for k, v in EDGE_GRID_RES.items()]
model_comparison_df['Edge_search_best_comb'] = [v['best_comb'] for k, v in EDGE_GRID_RES.items()]
model_comparison_df['Edge_best_gcnn_std'] = [v['std'] for k, v in EDGE_RES.items()]

# model_comparison_df = model_comparison_df[
#     ['best_acc', 'GAT', 'GCAT', 'EdgeNets (HP search)', 'Edge_search_best_comb', 'EdgeNets (best GCNN)',
#      'Edge_best_gcnn_std', 'GCNN with random SO', 'GCNN with extracted SO (Acc)',
#      'GCNN with extracted SO (Perc)',
#      'non_zero_el_acc_phi',
#      'non_zero_el_perc_phi',
#      '2l GCNN',
#      'Feed forward', 'SVM', 'KNN',
#      'best_comb',
#      'best_combination_2layer', 'best_comb_ff', 'std']]

# %%##################################################################
# Load data

# \\\ Own libraries:
import Utils.dataTools

# \\\ Separate functions:

dataDir = 'authorData'  # Data directory
dataFilename = 'authorshipData.mat'  # Data filename
dataPath = os.path.join(dataDir, dataFilename)  # Data path

# data = Utils.dataTools.Authorship('poe', 0.6, 0.2, dataPath)
data = Utils.dataTools.AutorshipGenderNationality(0.6, 0.2, dataPath)

# %%##################################################################
# run TSNE on the raw signals


X = np.concatenate((data.selectedAuthor['all']['wordFreq'], data.authorData['abbott']['wordFreq'],
                    ), axis=0)
# y = np.concatenate((np.array(['poe' for _ in range(data.selectedAuthor['all']['wordFreq'].shape[0])]),
#                     np.array(['abt' for _ in range(data.authorData['abbott']['wordFreq'].shape[0])])), axis=0)
y = np.concatenate((np.array([1 for _ in range(data.selectedAuthor['all']['wordFreq'].shape[0])]),
                    np.array([0 for _ in range(data.authorData['abbott']['wordFreq'].shape[0])]),
                    ), axis=0)

pca = PCA(n_components=4)
pca_result = pca.fit_transform(X)

pca_df = pd.DataFrame(columns=['pca1', 'pca2', 'pca3', 'pca4'])
pca_df['pca1'] = pca_result[:, 0]
pca_df['pca2'] = pca_result[:, 1]
pca_df['pca3'] = pca_result[:, 2]
pca_df['pca4'] = pca_result[:, 3]

print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))

top_two_comp = pca_df[['pca1', 'pca2']]
fashion_scatter(top_two_comp.values, y, "PCA")

fashion_tsne = TSNE(random_state=RS, perplexity=20).fit_transform(X)
fashion_scatter(fashion_tsne, y, "TSNE")
plt.show()
plt.title("Raw signal")

# X_embedded = TSNE(n_components=2).fit_transform(X)
# X_embedded.shape

# %%##################################################################
# train model, with different parameters, use TSNE for dim renderExternalDocumentation

data.get_split('poe', 0.6, 0.2)

comb = ([1, 32], [2])

archit = th.train_net(data, comb)
net = archit.archit

###################################################################
#                                                                   #
#                    EMBEDDING HOOK                                 #
#                                                                   #
#####################################################################
import collections
from functools import partial

activations = collections.defaultdict(list)


def save_activation(name, mod, inp, out):
    activations[name].append(out.cpu())


# Registering hooks for all the Conv2d layers
# Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
# called repeatedly at different stages of the forward pass (like RELUs), this will save different
# activations. Editing the forward pass code to save activations is the way to go for these cases.
for name, m in net.named_modules():
    if name.strip() == "GFL.2":
        # partial to assign the layer name to each hook
        m.register_forward_hook(partial(save_activation, name))

X = np.concatenate((data.authorData['abbott']['wordFreq'],
                    data.authorData['poe']['wordFreq']), axis=0)

X = prepare_for_training(X, archit.order)

with torch.no_grad():
    net(X)

activations = activations['GFL.2'][0]
activations = activations.detach().numpy()
activations = activations.reshape((activations.shape[0], activations.shape[1] * activations.shape[2]))

y = np.concatenate((
    np.array([0 for _ in range(data.authorData['abbott']['wordFreq'].shape[0])]),
    np.array([1 for _ in range(data.authorData['poe']['wordFreq'].shape[0])])), axis=0)

fashion_tsne = TSNE(random_state=RS, perplexity=20).fit_transform(activations)
fashion_scatter(fashion_tsne, y)
plt.title(str(comb))
plt.show()

# %%##################################################################
# use this code to analysi PHI of edge variant GNNs

test = modelsGNN['EdgeVariGNN'].archit.EVGFL[0].Phi

phi = test[0, 0, 0, 0, :, :]
phi = phi.detach().numpy()

function_words = np.array(data.functionWords)
function_words = function_words[nodesToKeep]  # we get order from taining NN

important_pairs = [(function_words[x[0]] + " - " + function_words[x[1]]) for x in
                   np.argwhere(np.abs(phi) > np.max(phi) - 0.05 * np.max(phi))]

indices = [x for x in
           np.argwhere(np.abs(phi) > np.max(phi) - 0.05 * np.max(phi))]

result = {'indices': indices, 'nodes': nodesToKeep, 'order': order}

# %%##################################################################
# Plot scatter plot for GCNN combinations and dataset expolration results

plt.style.use('fivethirtyeight')

means = [it['meanDeg'] for it in author_info.values()]
F = [eval(it['best_comb'])[0][1] for it in GCNN_results.values()]

df_mean_degree = pd.DataFrame(columns=['mean degree', 'combination'], data={'mean degree': means, 'combination': F})
ax = sns.scatterplot(x="mean degree", y="combination", data=df_mean_degree)
plt.title("Mean degree vs number of Features")
plt.show()

df_best_acc_mean_degree = pd.DataFrame(columns=['Accuracy', 'Mean degree'],
                                       data={'Mean degree': means, 'Accuracy': model_comparison_df['best_acc']})
bx = sns.relplot(x="Mean degree", y="Accuracy", data=df_best_acc_mean_degree)
plt.title("Mean degree vs number of Accuracy")

plt.show()

# %%##################################################################
# Build subgraphs for each author
import networkx as nx
import pandas as pd

plt.style.use('fivethirtyeight')

function_words = np.array(data.functionWords)

with open('EdgeVariGNN_important_words_phi_accuracy3.txt', 'r') as f:
    training_results = json.load(f)

    for author_name in data.authorData.keys():
        current = training_results[author_name]
        current_words = function_words[current['nodes']]  # we get order from taining NN
        df = pd.DataFrame({'source': [current_words[x[0]] for x in current['indices']],
                           'target': [current_words[x[1]] for x in current['indices']]})

        G = nx.from_pandas_edgelist(df, 'source', 'target')
        fig = plt.figure()
        nx.draw(G, with_labels=True, alpha=0.6, node_size=2000, width=6.0, node_color="#008fd5", node_shape="o",
                font_size=25, linewidths=10)
        plt.show()
        fig.savefig("f_word_subgraph{0}.png".format(author_name))

# %%##################################################################
# Visualize subgraphs for each author

import networkx as nx
import pandas as pd
import igraph as i

print(i.__version__)
plt.style.use('fivethirtyeight')

function_words = np.array(data.functionWords)

with open('EdgeVariGNN_important_words_phi.txt', 'r') as f:
    training_results = json.load(f)

    for author_name in data.authorData.keys():
        current = training_results[author_name]
        current_words = function_words[current['nodes']]  # we get order from taining NN
        df = pd.DataFrame({'source': [current_words[x[0]] for x in current['indices']],
                           'target': [current_words[x[1]] for x in current['indices']],
                           'weight': [abs(current['phi'][x[0]][x[1]]) for x in current['indices']]})
        df.to_csv('results/important_words_subgraphs_percentage/' + author_name + '_subgraph.csv', index=False)

# %%##################################################################
# Build word map for each author.
edg_var_word_map = {}
edge_var_important_words = {}


def unique_author_words(name, all):
    other_words = []

    for k, v in all.items():
        if k == name:
            continue

        other_words.extend(all[k])

    return [x for x in all[name] if x not in other_words]


with open('EdgeVariGNN_important_words_phi.txt', 'r') as f:
    edge_var_important_words = json.load(f)
    function_words = np.array(data.functionWords)

    for author_name in data.authorData.keys():
        print(author_name)
        current = edge_var_important_words[author_name]

        indices = np.array(edge_var_important_words[author_name]['indices']).flatten()
        indices = np.unique(indices)

        current_function_words = function_words[current['nodes']]  # we get order from taining NN

        words = current_function_words[indices]

        edg_var_word_map[author_name] = words

# %%##################################################################
# plot correlation between unique words and Accuracy
from sklearn.preprocessing import normalize


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


unique_words = [len(unique_author_words(k, edg_var_word_map)) for k, v in edg_var_word_map.items()]
acc = model_comparison_df['best_acc'].tolist()

plt.style.use('fivethirtyeight')

df_unique_words_acc = pd.DataFrame(columns=['Accuracy', 'Unique words'],
                                   data={'Unique words': unique_words, 'Accuracy': acc})
bx = sns.relplot(x="Accuracy", y="Unique words", data=df_unique_words_acc)
plt.title("Unique words vs number of Accuracy")
plt.show()

# ########################################################
# try jaccard similarity

jaccard_distances = {}

for k, v in edg_var_word_map.items():
    distances = []

    for k2, v2 in edg_var_word_map.items():
        if k2 == k:
            continue

        distances.append(jaccard_similarity(v2, v))

    jaccard_distances[k] = np.mean(distances)

df_jaccard_acc = pd.DataFrame(columns=['Accuracy', 'Jaccard'],
                              data={'Jaccard': list(jaccard_distances.values()), 'Accuracy': acc})
bx = sns.lmplot(x="Accuracy", y="Jaccard", data=df_jaccard_acc)
plt.title("Jaccard vs number of Accuracy")
plt.show()

########################################################
# try degree and accuracy
acc = model_comparison_df['best_acc'].tolist()
df_degree_acc = pd.DataFrame(columns=['Accuracy', 'Degree'],
                             data={'Degree': [x.shape[0] for x in edg_var_word_map.values()], 'Accuracy': acc})
bx = sns.relplot(x="Accuracy", y="Degree", data=df_degree_acc)
plt.title("Degree vs Accuracy")
plt.show()

# %%##################################################################
# histogram of the most popular word pairs

from scipy.stats import norm
import collections

no_of_words_to_select = 20

all_selected_pairs = np.concatenate(list([v['pairs'] for v in edge_var_important_words.values()]))

pair_count = dict(collections.Counter(all_selected_pairs).most_common(no_of_words_to_select))
labels, values = zip(*pair_count.items())

indexes = np.arange(len(labels))
bar_width = 2.35

# Plot the histogram.
plt.bar(indexes, values, )
plt.xticks(indexes, labels, rotation='vertical')

values = list(pair_count.values())
mu, std = norm.fit(values)
plt.style.use('fivethirtyeight')
plt.show()

# df_pop_words_hist = pd.DataFrame(columns=['Word'],
#                                  data={'Word': all_selected_pairs})

# %%##################################################################
# histogram of the most popular words

from scipy.stats import norm
import collections

no_of_words_to_select = 50

all_selected_words = np.concatenate(list(edg_var_word_map.values()))

word_count = dict(collections.Counter(all_selected_words).most_common(no_of_words_to_select))
labels, values = zip(*word_count.items())

# not_used_words = set(function_words).difference(set(word_count.keys()))
#
# for w in not_used_words:
#     word_count[w] = 0

indexes = np.arange(len(labels))
bar_width = 2.35

# Plot the histogram.
plt.bar(indexes, values)
plt.xticks(indexes, labels, rotation='vertical')

# values = list(word_count.values())
# mu, std = norm.fit(values)
plt.style.use('fivethirtyeight')
#
# xnew = np.linspace(0, no_of_words_to_select, 300)  # 300 represents number of points to make between T.min and T.max
# power_smooth = spline(indexes, values, xnew)
# plt.plot(xnew, power_smooth)

# # Plot the PDF.
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# title = "The most popular words"
# plt.title(title)

plt.show()

df_pop_words_hist = pd.DataFrame(columns=['Word'],
                                 data={'Word': all_selected_words})

# %%##################################################################
# Correlation between signal and PHI

graphNormalizationType = 'rows'  # or 'cols' - Makes all rows add up to 1.
keepIsolatedNodes = False  # If True keeps isolated nodes
forceUndirected = True  # If True forces the graph to be undirected (symmetrizes)
forceConnected = True  # If True removes nodes (from lowest to highest degree)

correlation = []
for author_name in edge_var_important_words.keys():
    phi = np.array(edge_var_important_words[author_name]['phi'])
    # phi = phi.flatten()
    data.get_split(author_name, 0.6, 0.2)
    nNodes = data.selectedAuthor['all']['wordFreq'].shape[1]

    signal = graphTools.Graph('fuseEdges', nNodes,
                              data.selectedAuthor['train']['WAN'],
                              'sum', graphNormalizationType, keepIsolatedNodes,
                              forceUndirected, forceConnected, [])

    corr = np.corrcoef(phi, signal.S[edge_var_important_words[author_name]['nodes']])
    correlation.append(corr)

# %%##################################################################
# Correlation between no of non zero elements in phi and accuracy

plt.style.use('fivethirtyeight')

df_acc_non_zero_phi_perc = pd.DataFrame()
df_acc_non_zero_phi_perc['Non zero elements'] = model_comparison_df['non_zero_el_perc_phi'].tolist()
df_acc_non_zero_phi_perc['Accuracy'] = model_comparison_df['best_acc_perc_phi'].tolist()

bx = sns.relplot(x="Accuracy", y="Non zero elements", data=df_acc_non_zero_phi_perc)
plt.show()

# %%##################################################################
# unique vs popular words for learning on GCNN

BASE_FILE_NAME_POPULAR = 'results/popular words GCNN/GNN_Polynomial_popular_words_results_'
BASE_FILE_NAME_UNIQUE = 'results/unique words GCNN/GNN_Polynomial_unique_words_results_'

GCNN_PUPULAR_results = {}
GCNN_UNIQUE_results = {}

for author_name in all_author_names:
    if not path.exists("{0}{1}.txt".format(BASE_FILE_NAME_POPULAR, author_name)):
        continue

    with open('{0}{1}.txt'.format(BASE_FILE_NAME_POPULAR, author_name), 'r') as f:
        train_result = json.load(f)
        best_acc, best_comb, std = analyse_GNN_results(train_result)
        GCNN_PUPULAR_results[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

    with open('{0}{1}.txt'.format(BASE_FILE_NAME_UNIQUE, author_name), 'r') as f:
        train_result = json.load(f)
        best_acc, best_comb, std = analyse_GNN_results(train_result)
        GCNN_UNIQUE_results[author_name] = {'best_acc': best_acc, "best_comb": best_comb, "std": std}

df_unique_popular = pd.DataFrame(GCNN_PUPULAR_results)
df_unique_popular = df_unique_popular.T

df_unique_popular['Unique'] = [v['best_acc'] for k, v in GCNN_UNIQUE_results.items()]

# %%##################################################################
# Correlation between 2nd eigenvalue and accuracy
plt.style.use('fivethirtyeight')

with open('author_info.json', 'r') as f:
    author_info = json.load(f)

    # Best acc and Fiedler eigenvalue
    df_best_acc_fiedler_eigenvalue = pd.DataFrame(columns=['Accuracy', 'Algebraic connectivity'],
                                                  data={'Algebraic connectivity': [x['eigenvalues_avg'][1] for x in
                                                                                   author_info.values()],
                                                        'Accuracy': model_comparison_df['best_acc'].tolist()})
    bx = sns.relplot(x="Algebraic connectivity", y="Accuracy", data=df_best_acc_fiedler_eigenvalue)
    plt.plot(np.unique(df_best_acc_fiedler_eigenvalue['Algebraic connectivity']), np.poly1d(
        np.polyfit(df_best_acc_fiedler_eigenvalue['Algebraic connectivity'], df_best_acc_fiedler_eigenvalue['Accuracy'],
                   1))(np.unique(df_best_acc_fiedler_eigenvalue['Algebraic connectivity'])))

    plt.show()

    # df_best_acc_fiedler_eigenvalue['Filter taps'] = [eval(it['best_comb'])[1][0] for it in GCNN_results.values()]
    # bx = sns.lmplot(x="Algebraic connectivity", y="Filter taps", data=df_best_acc_fiedler_eigenvalue)
    # plt.show()
    #
    # df_best_acc_fiedler_eigenvalue['Diameter'] = [x['diameter'] for x in author_info.values()]
    # bx = sns.relplot(x="Diameter", y="Filter taps", data=df_best_acc_fiedler_eigenvalue)
    # plt.show()
    #
    # bx = sns.relplot(x="Diameter", y="Accuracy", data=df_best_acc_fiedler_eigenvalue)
    # plt.show()

# %%##################################################################
# correlation between accuracy and non-zero els of matrix

with open('GNN_Polynomial_phi_non_zero_results_poe.txt', 'r') as f:
    phi_non_zero = json.load(f)

    means = []
    stds = []
    non_zero_count = []

    for k, v in phi_non_zero.items():
        means.append(np.mean(v['acc']))
        stds.append(np.std(v['acc']))
        non_zero_count.append(v['non_zero'])

    df_acc_non_zero = pd.DataFrame()

    df_acc_non_zero['acc'] = means
    df_acc_non_zero['stds'] = stds
    df_acc_non_zero['non zero'] = non_zero_count

    bx = sns.relplot(x="acc", y="non zero", data=df_acc_non_zero)
    plt.show()

# %%##################################################################
# Gender classification on sparse fp (parameter search)

with open('results/gender/gender_phi_results_GCNN_0.4_search.txt', 'r') as f:
    train_result = json.load(f)
    result = analyse_result_object(train_result)

# %%##################################################################
# Gender classification result exploration

GENDER_EDGE = 'results/gender/EdgeVariGNN_Gender_results_20200316165853.txt'
GENDER_GCNN_2L = 'results/gender/2_layer_GCNN_gender_results_20200316125137.txt'
GENDER_GCNN = 'results/gender/GCNN_gender_results_20200312161323.txt'
GENDER_LINEAR_KNN = 'results/gender/knn_results_gender.txt'
GENDER_LINEAR_SVM = 'results/gender/svm_results_gender.txt'
GENDER_MEN_SO = 'results/gender/GCNN_gender_male_SO_20200410105100.txt'
GENDER_EDGE_GCNN = 'results/gender/Edge_GCNN_gender_20200409191412.txt'
GENDER_PHI_PERC = 'results/gender/GCNN_gender_phi_results_03.txt'
GENDER_PHI_SVM = 'results/gender/SVC_gender_phi_results_03.txt'

df_gender_comparison = pd.DataFrame(index=['best_acc', 'best_comb', 'std', 'best_f1', 'best_auc'])

gender_box_plot_data = []
gender_box_plot_data_fp = []


def get_avg(data):
    acc = np.average([item['acc'] for item in data])
    f1 = np.average([item['f1'] for item in data])
    auc = np.average([item['auc'] for item in data])
    prec = np.average([item['prec'] for item in data])
    std = np.std([item['acc'] for item in data])

    return acc, f1, auc, prec, std


with open(GENDER_EDGE, 'r') as f:
    train_result = json.load(f)
    best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(train_result)
    result = {'best_acc': best_acc, 'best_comb': best_comb, 'std': std, 'best_f1': best_f1, 'best_auc': best_auc}
    df_gender_comparison['EdgeNet'] = list(result.values())
    gender_box_plot_data.append(train_result[best_comb]['acc'])

with open(GENDER_GCNN_2L, 'r') as f:
    train_result = json.load(f)
    best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(train_result)
    result = {'best_acc': best_acc, 'best_comb': best_comb, 'std': std, 'best_f1': best_f1, 'best_auc': best_auc}
    df_gender_comparison['GCNN (2L)'] = list(result.values())
    gender_box_plot_data.append(train_result[best_comb]['acc'])

with open(GENDER_GCNN, 'r') as f:
    train_result = json.load(f)
    best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(train_result)
    result = {'best_acc': best_acc, 'best_comb': best_comb, 'std': std, 'best_f1': best_f1, 'best_auc': best_auc}
    df_gender_comparison['GCNN'] = list(result.values())
    gender_box_plot_data.append(train_result[best_comb]['acc'])
#
# with open(GENDER_LINEAR_KNN, 'r') as f:
#     train_result = json.load(f)
#     result = analyze_linear_results(train_result)
#     result = {'best_acc': result['abbott'], 'best_comb': '', 'std': '', 'best_f1': '', 'best_auc': ''}
#     df_gender_comparison['GENDER_LINEAR_KNN'] = list(result.values())

with open(GENDER_LINEAR_SVM, 'r') as f:
    train_result = json.load(f)
    result, result_std = analyze_linear_results(train_result)
    result = {'best_acc': result['abbott'], 'best_comb': '', 'std': result_std['abbott'], 'best_f1': '', 'best_auc': ''}
    df_gender_comparison['SVM'] = list(result.values())
    gender_box_plot_data.append(train_result['abbott'])

# with open(GENDER_MEN_SO, 'r') as f:
#     train_result = json.load(f)
#     best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(train_result)
#     result = {'best_acc': best_acc, 'best_comb': best_comb, 'std': std, 'best_f1': best_f1, 'best_auc': best_auc}
#     df_gender_comparison['GENDER_MEN_SO'] = list(result.values())
#
with open(GENDER_EDGE_GCNN, 'r') as f:
    train_result = json.load(f)
    best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(train_result)
    result = {'best_acc': best_acc, 'best_comb': best_comb, 'std': std, 'best_f1': best_f1, 'best_auc': best_auc}
    df_gender_comparison['Edg-GCNN'] = list(result.values())
    gender_box_plot_data.append(train_result[best_comb]['acc'])

with open(GENDER_PHI_PERC, 'r') as f:
    train_result = json.load(f)
    best_acc, best_comb, std, best_f1, best_auc = get_avg(train_result)
    result = {'best_acc': best_acc, 'best_comb': 'test', 'std': std, 'best_f1': best_f1, 'best_auc': best_auc}
    df_gender_comparison['GCNN (FP)'] = list(result.values())
    # gender_box_plot_data_fp.append([x['acc'] for x in train_result])

with open(GENDER_PHI_SVM, 'r') as f:
    train_result = json.load(f)
    best_acc, best_f1, best_auc, prec, std = get_avg(train_result)
    result = {'best_acc': best_acc, 'best_comb': 'test', 'std': std, 'best_f1': best_f1, 'best_auc': best_auc}
    df_gender_comparison['SVM (FP)'] = list(result.values())
    gender_box_plot_data_fp.append([x['acc'] for x in train_result])

gender_box_plot_labels = list(df_gender_comparison.columns)
gender_box_plot_labels_fp = ['GCNN', 'SVM']

import plot_tools

# plot_tools.box_plot_with_dist(gender_box_plot_labels, gender_box_plot_data, 'Gender Classification', 'Accuracy')
# plot_tools.box_plot_with_dist(gender_box_plot_labels_fp, gender_box_plot_data_fp, 'Gender Classification (FP)',
#                               'Accuracy')
# create a dataframe with info from GCNN and FF2
# df_gender_comparison = df_gender_comparison.T
df_gender_comparison = df_gender_comparison.drop(df_gender_comparison.index[1])

# %%##################################################################
# Gender-nationality classification result exploration

GENDER_NATIONALITY_EDGE = 'results/gender_nationalit/gender_nationality_edgenet_results.txt'
GENDER_NATIONALITY_GCNN = 'results/gender_nationalit/gender_nationality_GCNN_64-4.txt'
GENDER_NATIONALITY_EDGE_GCNN = 'results/gender_nationalit/Gender_nationality_GCNN_Edge_results.txt'
# GENDER_NATIONALITY_LINEAR_SVM = 'results/gender/svm_results_gender.txt'
# GENDER_NATIONALITY_PHI = 'results/gender/gender_phi_results_GCNN_0.4_search.txt'
# GENDER_NATIONALITY_PHI_SVM = 'results/gender/SVC_gender_phi_results_03.txt'

df_gender_nationality_comparison = pd.DataFrame(index=['best_acc', 'best_comb', 'std', 'best_f1', 'best_auc'])

gender_nationality_box_plot_data = []
gender_nationality_box_plot_data_fp = []


def get_avg(data):
    acc = np.average([item['acc'] for item in data])
    f1 = np.average([item['f1'] for item in data])
    auc = np.average([item['auc'] for item in data])
    prec = np.average([item['prec'] for item in data])
    std = np.std([item['acc'] for item in data])

    return acc, f1, auc, prec, std


with open(GENDER_NATIONALITY_EDGE, 'r') as f:
    train_result = json.load(f)
    best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(train_result)
    result = {'best_acc': best_acc, 'best_comb': best_comb, 'std': std, 'best_f1': best_f1, 'best_auc': best_auc}
    df_gender_nationality_comparison['EdgeNet'] = list(result.values())
    gender_nationality_box_plot_data.append(train_result[best_comb]['acc'])

with open(GENDER_NATIONALITY_GCNN, 'r') as f:
    train_result = json.load(f)
    best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(train_result)
    result = {'best_acc': best_acc, 'best_comb': best_comb, 'std': std, 'best_f1': best_f1, 'best_auc': best_auc}
    df_gender_nationality_comparison['GCNN'] = list(result.values())
    gender_nationality_box_plot_data.append(train_result[best_comb]['acc'])

with open(GENDER_NATIONALITY_EDGE_GCNN, 'r') as f:
    train_result = json.load(f)
    best_acc, best_comb, std, best_f1, best_auc = analyse_GNN_results_extra(train_result)
    result = {'best_acc': best_acc, 'best_comb': best_comb, 'std': std, 'best_f1': best_f1, 'best_auc': best_auc}
    df_gender_nationality_comparison['Edge GCNN'] = list(result.values())
    gender_nationality_box_plot_data.append(train_result[best_comb]['acc'])

df_gender_nationality_comparison = df_gender_nationality_comparison.round(2)
df_gender_nationality_comparison = df_gender_nationality_comparison.drop(df_gender_nationality_comparison.index[1])

# df_gender_nationality_comparison.to_latex('results/latex_tables/gender_nationality_table.tex')
plot_tools.box_plot_with_dist(['EdgeNet', 'GCNN', 'Edge-GCNN'], gender_nationality_box_plot_data,
                              'Classification by gender and nationality', 'Accuracy')

# %%##################################################################
# Extract SO for gender classification from Edgenets PHI matrix

with open('results/EdgeVariGNN_gender_phi.txt', 'r') as f:
    results = json.load(f)
    phi = np.array(results['phi'])

    function_words = np.array(data.functionWords)
    function_words = function_words[results['nodes']]  # we get order from taining NN

    important_pairs = [(function_words[x[0]] + " - " + function_words[x[1]]) for x in
                       np.argwhere(np.abs(phi) > np.max(phi) - 0.05 * np.max(phi))]

    indices_to_zero = [x for x in
                       np.argwhere(np.abs(phi) < np.max(np.abs(phi)) - 0.3 * np.max(np.abs(phi)))]

    for x, y in indices_to_zero:
        phi[x, y] = 0

    result = {'indices_to_zero': indices_to_zero, 'new_phi': phi}

    with open('gender_phi_percentage.txt', 'w+') as outfile:
        json.dump(phi.tolist(), outfile)

with open('results/EdgeVariGNN_nationality_phi.txt', 'r') as f:
    results = json.load(f)
    phi = np.array(results['phi'])

    function_words = np.array(data.functionWords)
    function_words = function_words[results['nodes']]  # we get order from taining NN

    important_pairs = [(function_words[x[0]] + " - " + function_words[x[1]]) for x in
                       np.argwhere(np.abs(phi) > np.max(phi) - 0.05 * np.max(phi))]

    indices_to_zero = [x for x in
                       np.argwhere(np.abs(phi) < np.max(np.abs(phi)) - 0.3 * np.max(np.abs(phi)))]

    for x, y in indices_to_zero:
        phi[x, y] = 0

    result = {'indices_to_zero': indices_to_zero, 'new_phi': phi}

    with open('nationality_phi_percentage.txt', 'w+') as outfile:
        json.dump(phi.tolist(), outfile)

# %%##################################################################
# Extract graphs for Elvin

plt.style.use('fivethirtyeight')
model_comparison_df['Error'] = 1 - model_comparison_df['best_acc']
model_comparison_df['Error_ed'] = 1 - model_comparison_df['Edge']

best_10_acc = model_comparison_df.sort_values(by=['Error'], ascending=True).head(15)
best_10_acc = best_10_acc.rename(columns={"best_acc": "Accuracy"})
# best_10_acc["Error"] = 1 - best_10_acc['Accuracy']
#

# best_10_acc = best_10_acc.drop(['austen', 'poe', 'bronte'], axis=0)
best_10_acc = best_10_acc.drop(['austen', 'poe'], axis=0)
best_10_acc = best_10_acc.round(2)

final_df = pd.DataFrame(columns=['Name', 'Error', 'Sigma', 'Type'])

for index, row in best_10_acc.iterrows():
    cur_df = pd.DataFrame([[index, 1 - row['Accuracy'], row['std'], 'FIR']],
                          columns=['Name', 'Error', 'Sigma', 'Type'])
    final_df = final_df.append(cur_df)
    # print(row['c1'], row['c2'])

for index, row in best_10_acc.iterrows():
    cur_df = pd.DataFrame([[index, 1 - row['Edge'], row['Edge_std'], 'EdgeNet']],
                          columns=['Name', 'Error', 'Sigma', 'Type'])
    final_df = final_df.append(cur_df)
    # print(row['c1'], row['c2'])

best_10_acc.index = best_10_acc.index.str.capitalize()

# ax = sns.barplot(x='Name', y="Error", hue="Type", data=final_df, xerr=final_df['Sigma'])
#
# plt.show()


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = best_10_acc.index.to_list()
fir_means = list(map(lambda x: round(x * 100, 2), best_10_acc['Error'].to_list()))
fir_std = list(map(lambda x: round(x * 100, 2), best_10_acc['std'].to_list()))
edge_means = list(map(lambda x: round(x * 100, 2), best_10_acc['Error_ed'].to_list()))
edge_std = list(map(lambda x: round(x * 100, 2), best_10_acc['Edge_std'].to_list()))

# edge_means = best_10_acc['Error_ed'].to_list()

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, fir_means, width, label='GCNN', yerr=fir_std)
rects2 = ax.bar(x + width / 2, edge_means, width, label='EdgeNet', yerr=edge_std)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Classification error')
# ax.set_title('Error by author and network type')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects, offset=0.0):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height + offset),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2, offset=0.02)

fig.tight_layout()
# fig.savefig('plots/error_by_author_and_network.fig')
plt.show()

best_10_acc = best_10_acc.drop(['GAT', 'GCAT', 'random_so', 'best_acc_phi', 'best_acc_perc_phi',
                                'non_zero_el_acc_phi',
                                'non_zero_el_perc_phi',
                                'best_acc_2layer',
                                'best_acc_ff', 'best_acc_svm', 'best_acc_knn',
                                'best_combination_2layer', 'best_comb_ff', 'std'], axis=1)

# %%##################################################################
# Export one vs all results table for summary

model_comparison_df['GCNN'] = model_comparison_df['best_acc'].to_list()

df_one_vs_all_results = pd.DataFrame()

df_one_vs_all_results = model_comparison_df[
    ['GCNN', 'std', 'EdgeNets (HP search)', 'EdgeNets (HP search) std', 'EdgeNets (best GCNN)',
     'EdgeNets (best GCNN) std',
     '2l GCNN', '2l GCNN std',
     'SVM', 'SVM std', 'KNN', 'KNN std']]

df_one_vs_all_results = df_one_vs_all_results.round(2)
df_one_vs_all_results.index = df_one_vs_all_results.index.str.capitalize()

# pd.set_option('precision', 2)
df_one_vs_all_results.to_latex("results/latex_tables/one-vs-all.tex")

# %%##################################################################
# Compare one-vs-one, one-vs-one-fingerprint and one-vs-all-fingerprint
one_vs_one_base = 'results/one_vs_one/{}'
one_vs_all_base = 'results/one_vs_all/{}'

files = {
    "1v1 GCNN": one_vs_one_base.format('One_vs_one_results_GCNN.json'),
    "1v1 SVM": one_vs_one_base.format('One_vs_one_results_SVC.json'),
    "1v1 FP GCNN": one_vs_one_base.format('One_vs_one_fingerprint_results_GCNN.json'),
    "1v1 FP SVC": one_vs_one_base.format('One_vs_one_fingerprint_results_SVC.json'),
    "1vAll FP GCNN": one_vs_all_base.format('One_vs_one_fingerprint_results_GCNN.json'),
    "1vAll FP SVC": one_vs_all_base.format('One_vs_one_fingerprint_results_SVC.json'),
}


def get_author_avg(data):
    acc = np.average([item['acc'] for item in data])
    f1 = np.average([item['f1'] for item in data])
    auc = np.average([item['auc'] for item in data])
    prec = np.average([item['prec'] for item in data])
    std = np.std([item['acc'] for item in data])

    return acc, f1, auc, prec, std


results = {}

for k, v in files.items():
    with open(v, 'r') as outfile:
        data = json.load(outfile)

        results[k] = {}
        for author in data.keys():
            acc, f1, auc, prec, std = get_author_avg(data[author])
            results[k][author] = {'acc': acc, 'f1': f1, 'auc': auc, 'prec': prec, 'std': std}
            # results[k][author] = acc

results_feature_search = {}
feature_search_file_path = "results/one_vs_one/One_vs_one_fingerprint_feature_search_granularity_results_GCNN.json"

with open(feature_search_file_path, 'r') as outfile:
    data = json.load(outfile)

    for author in data.keys():
        results_feature_search[author] = {}
        # results_feature_search[author]['10'] = {results['1v1 FP GCNN'][author]}

        for feat_count in data[author].keys():
            acc, f1, auc, prec, std = get_author_avg(data[author][feat_count])
            time = -np.average([item['time'] for item in data[author][feat_count]])

            results_feature_search[author][feat_count] = [acc, f1, auc, prec, std, time]
            # results_feature_search[author][feat_count] = str(
            #     (acc, -np.average([item['time'] for item in data[author][feat_count]])))

# %%##################################################################
# one-vs-one GCNN vs SVM
df_1v1_comparison = pd.DataFrame(
    results['1v1 GCNN']).T

df_1v1_comparison['acc'] = 1 - df_1v1_comparison['acc']
df_1v1_comparison['svm'] = [1 - x['acc'] for x in results['1v1 SVM'].values()]
df_1v1_comparison['svm std'] = [x['std'] for x in results['1v1 SVM'].values()]
df_1v1_comparison['diff'] = df_1v1_comparison['acc'] - df_1v1_comparison['svm']

df_1v1_comparison = df_1v1_comparison.sort_values(by=['diff'])

df_1v1_comparison = df_1v1_comparison.head(7)
plot_tools.stacked_bar_plot(df_1v1_comparison.index.str.capitalize().to_list(), df_1v1_comparison['acc'].to_list(),
                            df_1v1_comparison['svm'].to_list(), df_1v1_comparison['std'].to_list(),
                            df_1v1_comparison['svm std'].to_list(), 'GCNN', 'SVM')

# %%##################################################################
# one-vs-one fingerpring search results

one_vs_one_base = 'results/one_vs_one/{}'
one_vs_all_base = 'results/one_vs_all/{}'

results_feature_search = {}
files = {
    "1v1 FP GCNN": one_vs_one_base.format('One_vs_one_fingerprint_feature_search_results_GCNN.json'),
    "1v1 FP SVC": one_vs_one_base.format('One_vs_one_fingerprint_feature_search_results_SVC.json'),
    "1vAll FP GCNN": one_vs_all_base.format('One_vs_all_fingerprint_feature_search_results_GCNN.json'),
    "1vAll FP SVC": one_vs_all_base.format('One_vs_all_fingerprint_feature_search_results_SVC.json'),
}

for k, v in files.items():
    with open(v, 'r') as outfile:
        data = json.load(outfile)

        results_feature_search[k] = []

        for author in data.keys():
            # results_feature_search[k][author] = []
            # results_feature_search[k][author]['10'] = {results['1v1 FP GCNN'][author]}

            for feat_count in data[author].keys():
                acc, f1, auc, prec, std = get_author_avg(data[author][feat_count])
                time = -np.average([item['time'] for item in data[author][feat_count]])

                results_feature_search[k].append({'acc': acc, "f1": f1, "auc": auc, "prec": prec, "std": std,
                                                  "time": time,
                                                  "perc": feat_count,
                                                  "author": author})

# %%##################################################################
# 1v1 fingerprint table

pd.set_option('precision', 2)

df_1v1_fp = pd.DataFrame(
    list(results_feature_search['1v1 FP GCNN']))

is_03 = df_1v1_fp['perc'] == '0.3'

df_1v1_fp = df_1v1_fp[is_03];
df_1v1_fp = df_1v1_fp[['acc', 'std', 'author']];
df_1v1_fp['type'] = 'GCNN'

df_1v1_fp_svm = pd.DataFrame(
    list(results_feature_search['1v1 FP SVC']))

is_03 = df_1v1_fp_svm['perc'] == '0.3'

df_1v1_fp_svm = df_1v1_fp_svm[is_03];
df_1v1_fp_svm = df_1v1_fp_svm[['acc', 'std', 'author']];
df_1v1_fp_svm['type'] = 'SVM'

# df_1v1_fp['SVM'] = list(df_1v1_fp_svm['acc'])
# df_1v1_fp['SVM std'] = list(df_1v1_fp_svm['std'])
# df_1v1_fp['type'] = 'SVM'
#
# df_1v1_fp.index = df_1v1_fp['author']

df_1v1_fp = df_1v1_fp.append(df_1v1_fp_svm)
df_1v1_fp['Error'] = 1 - df_1v1_fp['acc']
# # df_1v1_fp = df_1v1_fp[['acc', 'SVM']];
# df_1v1_fp.index = df_1v1_fp.index.str.capitalize()
df_1v1_fp['author'] = df_1v1_fp['author'].str.capitalize()

df_1v1_fp = df_1v1_fp.sort_values(by=['author'])
df_1v1_fp = df_1v1_fp.round(2)
df_1v1_fp = df_1v1_fp.rename(columns={"author": "Author"})

df_1v1_fp_1 = df_1v1_fp.iloc[:10, :]
df_1v1_fp_2 = df_1v1_fp.iloc[10:20, :]
df_1v1_fp_3 = df_1v1_fp.iloc[20:30, :]
df_1v1_fp_4 = df_1v1_fp.iloc[30:, :]

import plot_tools

labels = df_1v1_fp_1[df_1v1_fp_1['type'] == 'GCNN']['Author'].to_list()
gcnn_means = df_1v1_fp_1[df_1v1_fp_1['type'] == 'GCNN']['Error'].to_list()
SVM_means = df_1v1_fp_1[df_1v1_fp_1['type'] == 'SVM']['Error'].to_list()
gcnn_std = df_1v1_fp_1[df_1v1_fp_1['type'] == 'GCNN']['std'].to_list()
SVM_std = df_1v1_fp_1[df_1v1_fp_1['type'] == 'SVM']['std'].to_list()

plot_tools.stacked_bar_plot(labels, gcnn_means, SVM_means, gcnn_std, SVM_std, 'GCNN', 'SVM')

labels = df_1v1_fp_2[df_1v1_fp_2['type'] == 'GCNN']['Author'].to_list()
gcnn_means = df_1v1_fp_2[df_1v1_fp_2['type'] == 'GCNN']['Error'].to_list()
SVM_means = df_1v1_fp_2[df_1v1_fp_2['type'] == 'SVM']['Error'].to_list()
gcnn_std = df_1v1_fp_2[df_1v1_fp_2['type'] == 'GCNN']['std'].to_list()
SVM_std = df_1v1_fp_2[df_1v1_fp_2['type'] == 'SVM']['std'].to_list()

plot_tools.stacked_bar_plot(labels, gcnn_means, SVM_means, gcnn_std, SVM_std, 'GCNN', 'SVM')

labels = df_1v1_fp_3[df_1v1_fp_3['type'] == 'GCNN']['Author'].to_list()
gcnn_means = df_1v1_fp_3[df_1v1_fp_3['type'] == 'GCNN']['Error'].to_list()
SVM_means = df_1v1_fp_3[df_1v1_fp_3['type'] == 'SVM']['Error'].to_list()
gcnn_std = df_1v1_fp_3[df_1v1_fp_3['type'] == 'GCNN']['std'].to_list()
SVM_std = df_1v1_fp_3[df_1v1_fp_3['type'] == 'SVM']['std'].to_list()

plot_tools.stacked_bar_plot(labels, gcnn_means, SVM_means, gcnn_std, SVM_std, 'GCNN', 'SVM')

labels = df_1v1_fp_4[df_1v1_fp_4['type'] == 'GCNN']['Author'].to_list()
gcnn_means = df_1v1_fp_4[df_1v1_fp_4['type'] == 'GCNN']['Error'].to_list()
SVM_means = df_1v1_fp_4[df_1v1_fp_4['type'] == 'SVM']['Error'].to_list()
gcnn_std = df_1v1_fp_4[df_1v1_fp_4['type'] == 'GCNN']['std'].to_list()
SVM_std = df_1v1_fp_4[df_1v1_fp_4['type'] == 'SVM']['std'].to_list()

plot_tools.stacked_bar_plot(labels, gcnn_means, SVM_means, gcnn_std, SVM_std, 'GCNN', 'SVM')

# %%##################################################################
# 1 vs all fingerprint table

pd.set_option('precision', 2)

df_1vAll_fp = pd.DataFrame(
    list(results_feature_search['1vAll FP GCNN']))

is_03 = df_1vAll_fp['perc'] == '0.3'

df_1vAll_fp = df_1vAll_fp[is_03];
df_1vAll_fp = df_1vAll_fp[['acc', 'author', 'std']];
df_1vAll_fp['err'] = 1 - df_1vAll_fp['acc']

df_1vAll_fp_svm = pd.DataFrame(
    list(results_feature_search['1vAll FP SVC']))

is_03 = df_1vAll_fp_svm['perc'] == '0.2'

df_1vAll_fp_svm = df_1vAll_fp_svm[is_03];

df_1vAll_fp['SVM'] = list(df_1vAll_fp_svm['acc'])
df_1vAll_fp['SVM_err'] = 1 - df_1vAll_fp['SVM']
df_1vAll_fp['SVM_std'] = list(df_1vAll_fp_svm['std'])

# df_1vAll_fp = df_1vAll_fp[['acc','author', 'SVM']];
df_1vAll_fp['author'] = df_1vAll_fp['author'].str.capitalize()

df_1vAll_fp = df_1vAll_fp.rename(columns={"acc": "GCNN"})

df_1vAll_fp = df_1vAll_fp.sort_values(by=['author'])
df_1vAll_fp = df_1vAll_fp.round(2)
df_1vAll_fp = df_1vAll_fp.rename(columns={"author": "Author"})

df_1vAll_fp_1 = df_1vAll_fp.iloc[:5, :]
df_1vAll_fp_2 = df_1vAll_fp.iloc[5:10, :]
df_1vAll_fp_3 = df_1vAll_fp.iloc[10:15, :]
df_1vAll_fp_4 = df_1vAll_fp.iloc[15:, :]

import plot_tools

labels = df_1vAll_fp_1['Author'].to_list()
gcnn_means = df_1vAll_fp_1['err'].to_list()
SVM_means = df_1vAll_fp_1['SVM_err'].to_list()
gcnn_std = df_1vAll_fp_1['std'].to_list()
SVM_std = df_1vAll_fp_1['SVM_std'].to_list()

plot_tools.stacked_bar_plot(labels, gcnn_means, SVM_means, gcnn_std, SVM_std, 'GCNN', 'SVM')

labels = df_1vAll_fp_2['Author'].to_list()
gcnn_means = df_1vAll_fp_2['err'].to_list()
SVM_means = df_1vAll_fp_2['SVM_err'].to_list()
gcnn_std = df_1vAll_fp_2['std'].to_list()
SVM_std = df_1vAll_fp_2['SVM_std'].to_list()

plot_tools.stacked_bar_plot(labels, gcnn_means, SVM_means, gcnn_std, SVM_std, 'GCNN', 'SVM')

labels = df_1vAll_fp_3['Author'].to_list()
gcnn_means = df_1vAll_fp_3['err'].to_list()
SVM_means = df_1vAll_fp_3['SVM_err'].to_list()
gcnn_std = df_1vAll_fp_3['std'].to_list()
SVM_std = df_1vAll_fp_3['SVM_std'].to_list()

plot_tools.stacked_bar_plot(labels, gcnn_means, SVM_means, gcnn_std, SVM_std, 'GCNN', 'SVM')

labels = df_1vAll_fp_4['Author'].to_list()
gcnn_means = df_1vAll_fp_4['err'].to_list()
SVM_means = df_1vAll_fp_4['SVM_err'].to_list()
gcnn_std = df_1vAll_fp_4['std'].to_list()
SVM_std = df_1vAll_fp_4['SVM_std'].to_list()

plot_tools.stacked_bar_plot(labels, gcnn_means, SVM_means, gcnn_std, SVM_std, 'GCNN', 'SVM')

# %%##################################################################
# 1vs1 simple table
pd.set_option('precision', 2)

df_1v1 = pd.DataFrame()
df_1v1['GCNN'] = [x['acc'] for x in results['1v1 GCNN'].values()]
df_1v1['std'] = [x['std'] for x in results['1v1 GCNN'].values()]
df_1v1.index = results['1v1 GCNN'].keys()

df_1v1['SVM'] = [x['acc'] for x in results['1v1 SVM'].values()]
df_1v1['svm std'] = [x['std'] for x in results['1v1 SVM'].values()]

# df_1v1['GCNN (f1)'] = [x['f1'] for x in results['1v1 GCNN'].values()]
# df_1v1['SVM (f1)'] = [x['f1'] for x in results['1v1 SVM'].values()]

df_1v1['GCNN (FP)'] = df_1v1_fp[df_1v1_fp['type'] == 'GCNN']['acc'].to_list()
df_1v1['GCNN (FP) std'] = df_1v1_fp[df_1v1_fp['type'] == 'GCNN']['std'].to_list()

df_1v1['SVM (FP)'] = df_1v1_fp[df_1v1_fp['type'] == 'SVM']['acc'].to_list()
df_1v1['SVM (FP) std'] = df_1v1_fp[df_1v1_fp['type'] == 'SVM']['std'].to_list()

df_1v1.index = df_1v1.index.str.capitalize()

df_1v1 = df_1v1.round(2)

df_1v1.to_latex("results/latex_tables/1v1_normal.tex")
# %%##################################################################
# Compare 1vsAll FP vs 1vAll simple
plt.style.use('fivethirtyeight')

df_1_all_fp = pd.DataFrame(
    list(results_feature_search['1vAll FP GCNN']))
avg = df_1_all_fp.groupby(['perc']).mean()

is_03 = df_1_all_fp['perc'] == '0.3'

df_03 = df_1_all_fp[is_03];

df_1_all_fp_svm = pd.DataFrame(list(results_feature_search['1vAll FP SVC']))
avg_svm = df_1_all_fp_svm.groupby(['perc']).mean()
avg_svm['type'] = 'SVM'

avg_svm = avg_svm[['acc', 'type']]
avg_svm = avg_svm.drop(['0.01', '0.05', '0.15', '0.35', '0.4'])
avg_svm['Percentage'] = avg_svm.index.astype(float) * 100

df_result_comp_1vAll_fp = avg[['acc']].copy()
df_result_comp_1vAll_fp['type'] = 'GCNN'
df_result_comp_1vAll_fp['Percentage'] = df_result_comp_1vAll_fp.index.astype(float) * 100
# df_result_comp_1vAll_fp['acc svm'] = avg_svm['acc']
df_result_comp_1vAll_fp = df_result_comp_1vAll_fp.drop(['0.01', '0.05', '0.15', '0.35', '0.4'])

df_result_comp_1vAll_fp = df_result_comp_1vAll_fp.append(avg_svm, ignore_index=True)

df_result_comp_1vAll_fp['Error'] = 1 - df_result_comp_1vAll_fp['acc']

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

# ax = sns.barplot(x="Percentage", y="Error", hue="type", data=df_result_comp_1vAll_fp, palette=sns.color_palette(flatui))

# %%##################################################################
# Bar plot for comparing average of  1 vs 1 fingerprints (gcnn vs svm)
plt.style.use('fivethirtyeight')

df_1_all_fp = pd.DataFrame(
    list(results_feature_search['1vAll FP GCNN']))
avg = df_1_all_fp.groupby(['perc']).mean()

df_1_all_fp_svm = pd.DataFrame(list(results_feature_search['1vAll FP SVC']))
avg_svm = df_1_all_fp_svm.groupby(['perc']).mean()

avg['Percentage'] = avg.index.astype(float).to_list()
avg['Percentage'] = avg['Percentage'] * 100
avg['Percentage'] = avg['Percentage'].astype(int)
avg['Error'] = 1 - avg['prec']
avg_svm['Error'] = 1 - avg_svm['prec']

avg_svm = avg_svm.drop(['0.01', '0.02', '0.05', '0.15', '0.35', '0.4'])
avg = avg.drop(['0.01', '0.02', '0.05', '0.15', '0.35', '0.4'])

plot_tools.stacked_bar_plot(avg['Percentage'].to_list(), avg['Error'].to_list(), avg_svm['Error'].to_list(), avg['std'],
                            avg_svm['std'], 'GCNN', 'SVM')

# %%##################################################################
# PLot box-plots for average drop for each percentage (for all authors) (GCNN)
plt.style.use('fivethirtyeight')

df_1_v_1_fingerprint_search = pd.DataFrame(list(results_feature_search.values())[2])
# df_1_v_1_fingerprint_search["best_acc"] = model_comparison_df["best_acc"].to_list()

acc_drop = []

for idx, acc in enumerate(df_1_v_1_fingerprint_search['acc'].to_list()):
    best_acc = model_comparison_df["best_acc"][df_1_v_1_fingerprint_search['author'][idx]]
    acc_drop.append(best_acc - acc)

df_1_v_1_fingerprint_search['Accuracy drop'] = acc_drop
df_1_v_1_fingerprint_search['perc'] = df_1_v_1_fingerprint_search['perc'].astype(float) * 100
df_1_v_1_fingerprint_search = df_1_v_1_fingerprint_search.rename(columns={"perc": "Percentage"})

# flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ax = sns.boxplot(x="Percentage", y="Accuracy drop", data=df_1_v_1_fingerprint_search, palette=colors,
                 linewidth=1.5)
ax = sns.swarmplot(x="Percentage", y="Accuracy drop", data=df_1_v_1_fingerprint_search, alpha=0.5, palette=colors)

plt.title("1 vs. all accuracy drop of GCNN")

plt.show()

# %%##################################################################
# PLot box-plots for average drop for each percentage (for all authors) (SVM)

df_1_v_1_fingerprint_search = pd.DataFrame(list(results_feature_search.values())[3])
# df_1_v_1_fingerprint_search["best_acc"] = model_comparison_df["best_acc"].to_list()

acc_drop = []

for idx, acc in enumerate(df_1_v_1_fingerprint_search['acc'].to_list()):
    best_acc = model_comparison_df["best_acc"][df_1_v_1_fingerprint_search['author'][idx]]
    acc_drop.append(best_acc - acc)

df_1_v_1_fingerprint_search['Accuracy drop'] = acc_drop
df_1_v_1_fingerprint_search = df_1_v_1_fingerprint_search.rename(columns={"perc": "Percentage"})

# flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ax = sns.boxplot(x="Percentage", y="Accuracy drop", data=df_1_v_1_fingerprint_search, palette=colors,
                 linewidth=1.5)
ax = sns.swarmplot(x="Percentage", y="Accuracy drop", data=df_1_v_1_fingerprint_search, alpha=0.5, palette=colors)

plt.title("1 vs. All accuracy drop of SVM")

plt.show()

# %%##################################################################
# Table for


# Path("plots/one-vs-one/").mkdir(parents=True, exist_ok=True)
#
# # PLOT degree dist
# plt.style.use('fivethirtyeight')
# i = 0
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# for key in results_feature_search.keys():
#
#     ax.plot([k for (k, v) in results_feature_search[key].items()], [v[0] for (k,
#                                                                                     v) in
#                                                                     results_feature_search[key].items()], label=key,
#             linewidth=2.0)
#
#     if i % 5 == 4 and i != 0 and i < 19:
#         plt.xlabel("No. of Features")
#         plt.ylabel("Accuracy")
#
#         plt.title("Accuracy v.s. feat count")
#         plt.legend()
#         fig.savefig("plots/one-vs-one/one-vs-one_granularity_acc_vs_features_{0}.png".format((i + 1) / 5))
#         plt.show()
#
#     if i % 5 == 4 and i != 0 and i < 19:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#     i += 1
#
# plt.xlabel("No. of Features")
# plt.ylabel("Accuracy")
# plt.title("Accuracy v.s. feat count")
# plt.legend()
# fig.savefig("plots/one-vs-one/one-vs-one_granularity_acc_vs_features_{0}.png".format((i + 1) / 5))
# plt.show()
