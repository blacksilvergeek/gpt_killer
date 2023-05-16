# %%##################################################################
#                                                                   #
#                    import library                                 #
#                                                                   #
#####################################################################
# \\\ Standard libraries:
import os

import matplotlib
import numpy as np

from Utils import graphTools

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt

import torch

torch.set_default_dtype(torch.float64)
import collections
import networkx as nx

# \\\ Own libraries:
import Utils.dataTools

# \\\ Separate functions:

# %%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################
# print current working directory
print(os.getcwd())

# os.chdir("thesis-master")

all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']

#   Load the data, which will give a specific split
name = 'poe'
thisFilename = 'authorshipGNN'  # This is the general name of all related files

saveDirRoot = 'experiments'  # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename + '-' + name)
# Dir where to save all the results from each run
dataPath = os.path.join('authorData', 'authorshipData.mat')
graphNormalizationType = 'rows'  # or 'cols' - Makes all rows add up to 1.
keepIsolatedNodes = False

force_undirected = True
force_connected = True

# data = Utils.dataTools.Authorship(name, 1, 0,
#                                   dataPath, graphNormalizationType,
#                                   keepIsolatedNodes, force_undirected,
#                                   force_connected)
data = Utils.dataTools.Authorship(name, 1, 0, dataPath)
nNodes = data.selectedAuthor['all']['wordFreq'].shape[1]
nodesToKeep = []
G = graphTools.Graph('fuseEdges', nNodes,
                     data.authorData['abbott']['WAN'],
                     'sum', graphNormalizationType, keepIsolatedNodes,
                     force_undirected, force_connected, nodesToKeep)


# %%##################################################################
def get_degree_dist(ad):
    result = []

    for i in range(ad.shape[0]):
        result.append(np.count_nonzero(ad[i]))

    # counter = collections.Counter(result)
    return result


def convert_to_ad(matrix):
    result = np.zeros(matrix.shape)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                result[i, j] = 1

    return result


def get_laplacian(ad):
    result = []

    for i in range(ad.shape[0]):
        result.append(np.count_nonzero(ad[i]))

    # counter = collections.Counter(result)
    return np.diag(result) - convert_to_ad(ad)


# %%##################################################################
# LOOP FOR EACH AUTHOR

author_info = {}

for aut_name in all_author_names:
    nodesToKeep = []
    G = graphTools.Graph('fuseEdges', nNodes,
                         data.authorData[aut_name]['WAN'],
                         'sum', graphNormalizationType, keepIsolatedNodes,
                         force_undirected, force_connected, nodesToKeep)
    G.computeGFT()

    excerpts = data.authorData[aut_name]['WAN']
    author_info[aut_name] = {}

    mean_degs = []
    degs = []
    eigenvalues = []

    no_of_excerpts = excerpts.shape[0]

    for i in range(no_of_excerpts):
        excerpt = excerpts[i]

        deg = get_degree_dist(excerpt)
        # counter = collections.Counter(deg)
        mean_degs.append(np.mean(deg))

        e, V = np.linalg.eig(get_laplacian(excerpt))
        eigenvalues.append(e)

        degs.extend(deg)

    # compute degree histogram
    counter = collections.Counter(degs)

    # Average out the degrees
    for k in counter.keys():
        counter[k] = counter[k] / no_of_excerpts

    # use nx to calculate diameter
    G_n = nx.from_numpy_matrix(G.A)

    author_info[aut_name]['meanDeg'] = np.mean(mean_degs)
    author_info[aut_name]['std'] = np.std(mean_degs)
    author_info[aut_name]['degrees'] = counter  # np.reshape(degs, (no_of_excerpts, len(degs[0])))
    author_info[aut_name]['eigenvalues'] = np.diagonal(G.E)
    author_info[aut_name]['eigenvalues_avg'] = np.mean(eigenvalues, axis=0)
    author_info[aut_name]['diameter'] = nx.diameter(G_n)

# %%##################################################################
# PLOT degree dist
plt.style.use('fivethirtyeight')
i = 0

fig = plt.figure()
ax = fig.add_subplot(111)

for key in sorted(author_info.keys()):

    deg_dist = author_info[key]['degrees']

    # deg_dist[0] = deg_dist.pop(0.1)
    items = sorted(deg_dist.items())
    ax.plot([k for (k, v) in items], [v for (k,
                                             v) in items], label=key, linewidth=2.0)

    if i % 5 == 4 and i != 0 and i < 19:
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.xlabel("Degree")
        plt.ylabel("Count")

        plt.title("Degree Distribution of WAN graph")
        plt.legend()
        fig.savefig("degree_distribution_{0}.png".format((i + 1) / 5))
        plt.show()

    if i % 5 == 4 and i != 0 and i < 19:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    i += 1

ax.set_xscale('log')
ax.set_yscale('log')

plt.xlabel("Degree")
plt.ylabel("Count")

plt.title("Degree Distribution of WAN graph")
plt.legend()
fig.savefig("degree_distribution_{0}.png".format(i / 5))
plt.show()

# %%##################################################################
# Plot Eigenvalue distributions

plt.style.use('fivethirtyeight')
fig = plt.figure()
ax = fig.add_subplot(111)

i = 0

fig = plt.figure()
ax = fig.add_subplot(111)

for key in sorted(author_info.keys()):

    ax.plot(author_info[key]['eigenvalues_avg'][:50], label=key, linewidth=2.0)

    if i % 5 == 4 and i != 0 and i < 19:
        plt.xlabel("Eigenvalue order")
        plt.ylabel("Eigenvalue")

        plt.title("Eigenvalue Distribution of WAN graph")
        plt.legend()

        # ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        # fig.savefig("eigenvalue_distribution_{0}.png".format((i + 1) / 5))
        plt.show()

    if i % 5 == 4 and i != 0 and i < 19:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    i += 1

plt.xlabel("Eigenvalue order")
plt.ylabel("Eigenvalue")

plt.title("Eigenvalue Distribution of WAN graph")
plt.legend()
# fig.savefig("eigenvalue_distribution_{0}.png".format(i / 5))
plt.show()

# %%##################################################################
# Plot mean degrees

plt.style.use('fivethirtyeight')

means = [it['meanDeg'] for it in author_info.values()]
stds = [it['std'] for it in author_info.values()]

mean_deg_author = dict(zip(all_author_names, means))
sorted_author_deg = sorted(mean_deg_author.items())
sorted_author_deg = collections.OrderedDict(sorted_author_deg)

plt.bar(np.arange(len(all_author_names)), sorted_author_deg.values(), yerr=stds)
plt.xlabel("Author name")
plt.ylabel("Mean Degree")
plt.title("Mean degree of WAN graph for each author")

plt.xticks(np.arange(len(all_author_names)), sorted_author_deg.keys(), rotation='vertical')

plt.show()

# %%##################################################################
# plot author-excerpt barchart

import json
import matplotlib.style as style

plt.style.use('fivethirtyeight')
sorted_auth_excerpt = None

with open('author_excerpt.json', 'r') as f:
    auth_excerpt = json.load(f)

    sorted_auth_excerpt = sorted(auth_excerpt.items())
    sorted_auth_excerpt = collections.OrderedDict(sorted_auth_excerpt)

    style.use('fivethirtyeight')

    plt.bar(np.arange(len(sorted_auth_excerpt.keys())), sorted_auth_excerpt.values())
    plt.xlabel("Author name")
    plt.ylabel("Number of excerpts")
    plt.title("Number of excerpts for each author")

    plt.xticks(np.arange(len(sorted_auth_excerpt.keys())), sorted_auth_excerpt.keys(), rotation='vertical')

    plt.show()

# %%##################################################################
# plot accuracy barchart for 2 layer feed forward network.

import json

with open('2_feedforward_results.txt', 'r') as f:
    train_result = json.load(f)

    for author_name in sorted(train_result.keys()):
        author_result = train_result[author_name]

        means = []
        stds = []

        for combination in author_result.keys():
            mean = np.mean(author_result[combination])
            std = np.std(author_result[combination])

            means.append(mean)
            stds.append(std)

        fig = plt.figure()

        plt.bar(np.arange(len(means)), means, yerr=stds)
        plt.xlabel("Combination of learning rate and batch size")
        plt.ylabel("Mean Accuracy")
        plt.title(author_name)

        plt.xticks(np.arange(len(means)), author_result.keys(), rotation='vertical')
        plt.show()

        fig.savefig("2_feedforward_acc_{0}.png".format(author_name))

# %%##################################################################
# plot accuracy barchart for 1 layer feed forward network.

import json

with open('feedforward_results.txt', 'r') as f:
    train_result = json.load(f)

    for author_name in sorted(train_result.keys()):
        author_result = train_result[author_name]

        means = []
        stds = []

        for combination in author_result.keys():
            mean = np.mean(author_result[combination])
            std = np.std(author_result[combination])

            means.append(mean)
            stds.append(std)

        fig = plt.figure()

        plt.bar(np.arange(len(means)), means, yerr=stds)
        plt.xlabel("Combination of learning rate and batch size")
        plt.ylabel("Mean Accuracy")
        plt.title(author_name)

        plt.xticks(np.arange(len(means)), author_result.keys(), rotation='vertical')
        plt.show()

        fig.savefig("feedforward_acc_{0}.png".format(author_name))


# %%##################################################################
# Helpers for numpy serialization


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.complex128:
                obj = obj.real
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


with open('some name.json', mode='w+') as f:
    json.dump(author_info, f, cls=NumpyEncoder, ensure_ascii=False)
