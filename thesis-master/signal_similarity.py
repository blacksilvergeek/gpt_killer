import os

import numpy
from scipy.spatial.distance import pdist, squareform

from Utils import dataTools


def get_author_average(signals):
    return numpy.mean(signals, axis=0)


def get_dissimilarity_matrix(data=None):
    if data is None:
        # the results from each run
        dataDir = 'authorData'  # Data directory
        dataFilename = 'authorshipData.mat'  # Data filename
        dataPath = os.path.join(dataDir, dataFilename)  # Data path

        data = dataTools.Authorship('poe', 1, 0, dataPath=dataPath)

    all_author_names = data.authorData.keys()

    authors_mean_signals = []
    for name in all_author_names:
        curr_signals = data.authorData[name]["wordFreq"]
        avg = get_author_average(curr_signals)

        authors_mean_signals.append(avg)

    dis_matrix = squareform(pdist(numpy.array(authors_mean_signals)))

    return dis_matrix


if __name__ == "__main__":
    m = get_dissimilarity_matrix()
