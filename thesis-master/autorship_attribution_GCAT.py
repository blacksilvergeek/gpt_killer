# 2020/02/04~
# Elvin Isufi: E.Isufi-1@tudelft.nl

# Test the movie recommendation dataset on several architectures.

# When it runs, it produces the following output:
#   - It trains the specified models and saves the best and the last model
#       parameters of each realization on a directory named 'savedModels'.
#   - It saves a pickle file with the torch random state and the numpy random
#       state for reproducibility.
#   - It saves a text file 'hyperparameters.txt' containing the specific
#       (hyper)parameters that control the run, together with the main (scalar)
#       results obtained.
#   - If desired, logs in tensorboardX the training loss and evaluation measure
#       both of the training set and the validation set. These tensorboardX logs
#       are saved in a logsTB directory.
#   - If desired, saves the vector variables of each realization (training and
#       validation loss and evaluation measure, respectively); this is saved
#       in pickle format. These variables are saved in a trainVars directory.
#   - If desired, plots the training and validation loss and evaluation
#       performance for each of the models, together with the training loss and
#       validation evaluation performance for all models. The summarizing
#       variables used to construct the plots are also saved in pickle format.
#       These plots (and variables) are in a figs directory.

# %%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

# \\\ Standard libraries:
import atexit
import json
import os
import signal
from functools import partial
from os import path

import numpy as np
from sklearn.metrics import f1_score, roc_curve, auc

import datetime
from copy import deepcopy
import pandas as pd

import torch;
from ast import literal_eval as make_tuple

from Utils import ClusterUtils

torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

# \\\ Own libraries:
import Utils.graphTools as graphTools
import Utils.dataTools
import Utils.graphML as gml
import Modules.architectures_extended as archit
import Modules.model as model
import Modules.train as train

# \\\ Separate functions:
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed

# Start measuring time
startRunTime = datetime.datetime.now()


# %%##################################################################
# Read active authors

def delete_active_author(name, active_file, signal, frame):
    print('EXIT F-TION')
    ClusterUtils.delete_from_active(active_file, name)


def test_fn(signal, frame):
    print('EXIT F-TION')


ACTIVE_AUTHORS_FILE = 'AU_GCAT_ACTIVE.txt'

# %%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']
BASE_FILE_NAME = 'Autorship_attribution_GCAT_results_'

thisFilename = 'authorEdgeNets'  # This is the general name of all related files

saveDirRoot = 'experiments'  # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename)  # Dir where to save all
# the results from each run
dataDir = 'authorData'  # Data directory
dataFilename = 'authorshipData.mat'  # Data filename
dataPath = os.path.join(dataDir, dataFilename)  # Data path

# Options:
doPrint = True  # Decide whether to print stuff while running
doLogging = False  # Log into tensorboard
doSaveVars = False  # Save (pickle) useful variables
doFigs = False  # Plot some figures (this only works if doSaveVars is True)

# \\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + today
# Create directory
# if not os.path.exists(saveDir):
#     os.makedirs(saveDir)
# # Create the file where all the (hyper)parameters and results will be saved.
# varsFile = os.path.join(saveDir, 'hyperparameters.txt')
# with open(varsFile, 'w+') as file:
#     file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

# \\\ Save seeds for reproducibility
# #   PyTorch seeds
# torchState = torch.get_rng_state()
# torchSeed = torch.initial_seed()
# #   Numpy seeds
# numpyState = np.random.RandomState().get_state()
# #   Collect all random states
# randomStates = []
# randomStates.append({})
# randomStates[0]['module'] = 'numpy'
# randomStates[0]['state'] = numpyState
# randomStates.append({})
# randomStates[1]['module'] = 'torch'
# randomStates[1]['state'] = torchState
# randomStates[1]['seed'] = torchSeed
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
# saveSeed(randomStates, saveDir)

########
# DATA #
########
# find the next author
combinations = [1, 3, 5, 7, 9]

authorName = ClusterUtils.get_author_name(ACTIVE_AUTHORS_FILE, BASE_FILE_NAME, combinations)

try:
    atexit.register(delete_active_author, authorName, ACTIVE_AUTHORS_FILE, None, None)

    for sig in signal.Signals:
        try:
            # signal.signal(sig, test_fn)
            signal.signal(sig, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
        except (ValueError, OSError):
            print('invalid: ' + str(sig))

    # signal.signal(signal.SIGHUP, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGINT, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGQUIT, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGILL, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGTRAP, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGABRT, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGBUS, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGFPE, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # # signal.signal(signal.SIGKILL, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGUSR1, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGSEGV, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGUSR2, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGPIPE, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGALRM, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGTERM, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))

    # signal.signal(signal.SIGINT, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # # signal.signal(signal.SIGKILL, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))
    # signal.signal(signal.SIGTERM, partial(delete_active_author, authorName, ACTIVE_AUTHORS_FILE))

    file_name = "{0}{1}.txt".format(BASE_FILE_NAME, authorName)

    # create empty files so that other jobs would skip this author
    with open(file_name, mode='w+') as f:
        pass

    # Possible authors: (just use the names in ' ')
    # jacob 'abbott', robert louis 'stevenson', louisa may 'alcott',
    # horatio 'alger', james 'allen', jane 'austen', emily 'bronte', james 'cooper',
    # charles 'dickens', hamlin 'garland', nathaniel 'hawthorne', henry 'james',
    # herman 'melville', 'page', herny 'thoreau', mark 'twain',
    # arthur conan 'doyle', washington 'irving', edgar allan 'poe',
    # sarah orne 'jewett', edith 'wharton'

    # load best performing hyperparameters
    comparison_df = pd.read_json('model_comparison_df.json')
    tuples = [make_tuple(x) for x in comparison_df['best_comb'].to_list()]

    author_name_comb = dict(zip(all_author_names, tuples))

    nFeatures = author_name_comb[authorName][0]  # F: number of output features of the only layer
    nShifts = author_name_comb[authorName][1]  # K: number of shift tap

    if doPrint:
        print('Author: {0}, Combination: {1}'.format(authorName, str((nFeatures, nShifts))))

    # set training params
    nClasses = 1  # Either authorName or not
    ratioTrain = 0.6  # Ratio of training samples
    ratioValid = 0.2  # Ratio of validation samples (out of the total training
    # samples)
    # Final split is:
    #   nValidation = round(ratioValid * ratioTrain * nTotal)
    #   nTrain = round((1 - ratioValid) * ratioTrain * nTotal)
    #   nTest = nTotal - nTrain - nValidation

    nDataSplits = 10  # Number of data realizations
    # Obs.: The built graph depends on the split between training, validation and
    # testing. Therefore, we will run several of these splits and average across
    # them, to obtain some result that is more robust to this split.

    # Every training excerpt has a WAN associated to it. We combine all these WANs
    # into a single graph to use as the supporting graph for all samples. This
    # combination happens under some extra options:
    graphNormalizationType = 'rows'  # or 'cols' - Makes all rows add up to 1.
    keepIsolatedNodes = False  # If True keeps isolated nodes
    forceUndirected = True  # If True forces the graph to be undirected (symmetrizes)
    forceConnected = True  # If True removes nodes (from lowest to highest degree)
    # until the resulting graph is connected.

    # # \\\ Save values:
    # writeVarValues(varsFile,
    #                {'authorName': authorName,
    #                 'nClasses': nClasses,
    #                 'ratioTrain': ratioTrain,
    #                 'ratioValid': ratioValid,
    #                 'nDataSplits': nDataSplits,
    #                 'graphNormalizationType': graphNormalizationType,
    #                 'keepIsolatedNodes': keepIsolatedNodes,
    #                 'forceUndirected': forceUndirected,
    #                 'forceConnected': forceConnected})

    ############
    # TRAINING #
    ############

    # \\\ Individual model training options
    trainer = 'ADAM'  # Options: 'SGD', 'ADAM', 'RMSprop'
    learningRate = 0.005  # In all options
    beta1 = 0.9  # beta1 if 'ADAM', alpha if 'RMSprop'
    beta2 = 0.999  # ADAM option only

    # \\\ Loss function choice
    lossFunction = nn.BCELoss()
    minRatings = 0  # Discard samples (rows and columns) with less than minRatings
    # ratings
    interpolateRatings = False  # Interpolate ratings with nearest-neighbors rule
    # before feeding them into the GNN

    # \\\ Overall training options
    nEpochs = 25  # Number of epochs
    batchSize = 16  # Batch size
    doLearningRateDecay = False  # Learning rate decay
    learningRateDecayRate = 0.9  # Rate
    learningRateDecayPeriod = 1  # How many epochs after which update the lr
    validationInterval = 5  # How many training steps to do the validation

    # # \\\ Save values
    # writeVarValues(varsFile,
    #                {'trainer': trainer,
    #                 'learningRate': learningRate,
    #                 'beta1': beta1,
    #                 'beta2': beta2,
    #                 'lossFunction': lossFunction,
    #                 'minRatings': minRatings,
    #                 'interpolateRatings': interpolateRatings,
    #                 'nEpochs': nEpochs,
    #                 'batchSize': batchSize,
    #                 'doLearningRateDecay': doLearningRateDecay,
    #                 'learningRateDecayRate': learningRateDecayRate,
    #                 'learningRateDecayPeriod': learningRateDecayPeriod,
    #                 'validationInterval': validationInterval})

    #################
    # ARCHITECTURES #
    #################

    # Just four architecture one- and two-layered Selection and Local GNN. The main
    # difference is that the Local GNN is entirely local (i.e. the output is given
    # by a linear combination of the features at a single node, instead of a final
    # MLP layer combining the features at all nodes).

    # Select desired architectures
    doSelectionGNN = False
    do1Layer = False
    doEdgeVariantGNN = False
    doNodeVariantGNN = False
    dohParamsHEVDeg = False
    doARMA = False
    doGAT = False
    doGCAT = True
    doEVGAT = False

    # In this section, we determine the (hyper)parameters of models that we are
    # going to train. This only sets the parameters. The architectures need to be
    # created later below. Do not forget to add the name of the architecture
    # to modelList.

    # If the hyperparameter dictionary is called 'hParams' + name, then it can be
    # picked up immediately later on, and there's no need to recode anything after
    # the section 'Setup' (except for setting the number of nodes in the 'N'
    # variable after it has been coded).

    # The name of the keys in the hyperparameter dictionary have to be the same
    # as the names of the variables in the architecture call, because they will
    # be called by unpacking the dictionary.

    modelList = []

    # \\\\\\\\\\\\\\\\\\\\\
    # \\\ SELECTION GNN \\\
    # \\\\\\\\\\\\\\\\\\\\\

    if doSelectionGNN:
        # \\\ Basic parameters for all the Selection GNN architectures

        hParamsSelGNN = {}  # Hyperparameters (hParams) for the Selection GNN (SelGNN)

        hParamsSelGNN['name'] = 'SelGNN'
        # Chosen architecture
        hParamsSelGNN['archit'] = archit.SelectionGNN

        # Graph convolutional parameters
        hParamsSelGNN['dimNodeSignals'] = [1, 32]  # Features per layer
        hParamsSelGNN['nFilterTaps'] = [5]  # Number of filter taps per layer
        hParamsSelGNN['bias'] = True  # Decide whether to include a bias term
        # Nonlinearity
        hParamsSelGNN['nonlinearity'] = nn.ReLU  # Selected nonlinearity
        # Pooling
        hParamsSelGNN['poolingFunction'] = gml.NoPool  # Summarizing function
        hParamsSelGNN['nSelectedNodes'] = None  # To be determined later on
        hParamsSelGNN['poolingSize'] = [1]  # poolingSize-hop neighborhood that
        # is affected by the summary
        # Full MLP readout layer (this layer breaks the locality of the solution)
        hParamsSelGNN['dimLayersMLP'] = [1]  # Dimension of the fully connected
        # layers after the GCN layers, we just need to output a single scalar
        # Graph structure
        hParamsSelGNN['GSO'] = None  # To be determined later on, based on data
        hParamsSelGNN['order'] = None  # Not used because there is no pooling

        # \\\\\\\\\\\\
        # \\\ MODEL 1: Selection GNN with 1 less layer
        # \\\\\\\\\\\\

        hParamsSelGNN1Ly = deepcopy(hParamsSelGNN)

        hParamsSelGNN1Ly['name'] += '1Ly'  # Name of the architecture

        # \\\ Save Values:
        # writeVarValues(varsFile, hParamsSelGNN1Ly)
        modelList += [hParamsSelGNN1Ly['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 2: Full edge-variant
    # \\\\\\\\\\\\

    if doEdgeVariantGNN:
        ##############
        # PARAMETERS #
        ##############

        hParamsFullEdgeVariant = {}

        hParamsFullEdgeVariant['name'] = 'FullEdgeVariant'
        # Chosen architecture
        hParamsFullEdgeVariant['archit'] = archit.EdgeVariantGNN

        # \\\ Architecture parameters
        hParamsFullEdgeVariant['dimNodeSignals'] = [1, 32]  # Features per layer
        hParamsFullEdgeVariant['nShiftTaps'] = [5]  # Number of shift taps per layer
        hParamsFullEdgeVariant['nFilterNodes'] = None
        hParamsFullEdgeVariant['nSelectedNodes'] = None
        hParamsFullEdgeVariant['bias'] = True  # Decide whether to include a bias term
        hParamsFullEdgeVariant['nonlinearity'] = nn.ReLU  # Selected nonlinearity
        hParamsFullEdgeVariant['poolingFunction'] = gml.NoPool  # Summarizing function
        hParamsFullEdgeVariant['poolingSize'] = [1]  # These are ignored when there is no pooling,
        # better set it to 1 to make everything slightly faster
        hParamsFullEdgeVariant['dimLayersMLP'] = [1]  # Dimension of the fully
        # connected layers after the GCN layers
        # Graph structure
        hParamsFullEdgeVariant['GSO'] = None  # To be determined later on, based on data
        hParamsFullEdgeVariant['order'] = None  # Not used because there is no pooling

        # \\\ Save Values:
        # writeVarValues(varsFile, hParamsFullEdgeVariant)
        modelList += [hParamsFullEdgeVariant['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL3: Node-Variant GNN ordered by Degree, single layer
    # \\\\\\\\\\\\

    if doNodeVariantGNN:
        hParamsNodeVariantDeg = {}  # Hyperparameters (hParams) for the Selection GNN (SelGNN)

        hParamsNodeVariantDeg['name'] = 'NodeVariantDeg'  # Name of the architecture
        # Chosen architecture
        hParamsNodeVariantDeg['archit'] = archit.NodeVariantGNN
        # \\\ Architecture parameters
        hParamsNodeVariantDeg['dimNodeSignals'] = [1, 32]  # Features per layer
        hParamsNodeVariantDeg['nShiftTaps'] = [5]  # Number of shift taps per layer
        hParamsNodeVariantDeg['nNodeTaps'] = None  # Number of node taps per layer
        hParamsNodeVariantDeg['bias'] = True  # Decide whether to include a bias term
        hParamsNodeVariantDeg['nonlinearity'] = nn.ReLU  # Selected nonlinearity
        hParamsNodeVariantDeg['poolingFunction'] = gml.NoPool  # Summarizing function
        hParamsNodeVariantDeg['poolingSize'] = [1]  # alpha-hop neighborhood that is
        # affected by the summary
        hParamsNodeVariantDeg['dimLayersMLP'] = [1]  # Dimension of the fully
        # connected layers after the GCN layers
        # Graph structure
        hParamsNodeVariantDeg['GSO'] = None  # To be determined later on, based on data
        hParamsNodeVariantDeg['order'] = None  # Not used because there is no pooling

        # \\\ Save Values:
        # writeVarValues(varsFile, hParamsNodeVariantDeg)
        modelList += [hParamsNodeVariantDeg['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 4: Hybrid edge-variant ordered by Degree
    # \\\\\\\\\\\\

    if dohParamsHEVDeg:
        ##############
        # PARAMETERS #
        ##############

        hParamsHybEdgeVariantDeg = {}

        hParamsHybEdgeVariantDeg['name'] = 'HybEdgeVariantDeg'
        # Chosen architecture
        hParamsHybEdgeVariantDeg['archit'] = archit.EdgeVariantGNN

        # \\\ Architecture parameters
        hParamsHybEdgeVariantDeg['dimNodeSignals'] = [1, 32]  # Features per layer
        hParamsHybEdgeVariantDeg['nShiftTaps'] = [5]  # Number of shift taps per layer
        hParamsHybEdgeVariantDeg['nFilterNodes'] = None
        hParamsHybEdgeVariantDeg['nSelectedNodes'] = None
        hParamsHybEdgeVariantDeg['bias'] = True  # Decide whether to include a bias term
        hParamsHybEdgeVariantDeg['nonlinearity'] = nn.ReLU  # Selected nonlinearity
        hParamsHybEdgeVariantDeg['poolingFunction'] = gml.NoPool  # Summarizing function
        hParamsHybEdgeVariantDeg['poolingSize'] = [1]  # These are ignored when there is no pooling,
        # better set it to 1 to make everything slightly faster
        hParamsHybEdgeVariantDeg['dimLayersMLP'] = [1]  # Dimension of the fully
        # connected layers after the GCN layers

        # Graph structure
        hParamsHybEdgeVariantDeg['GSO'] = None  # To be determined later on, based on data
        hParamsHybEdgeVariantDeg['order'] = None  # Not used because there is no pooling

        # \\\ Save Values:
        # writeVarValues(varsFile, hParamsHybEdgeVariantDeg)
        modelList += [hParamsHybEdgeVariantDeg['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 5: ARMANet with No Pooling
    # \\\\\\\\\\\\

    if doARMA:
        ##############
        # PARAMETERS #
        ##############

        hParamsARMANet = {}

        hParamsARMANet['name'] = 'ARMANet'

        # Chosen architecture
        hParamsARMANet['archit'] = archit.ARMAfilterGNN

        # \\\ Architecture parameters
        hParamsARMANet['dimNodeSignals'] = [1, 32]  # Features per layer
        hParamsARMANet['nDenominatorTaps'] = [5]  # Number of filter taps per layer
        hParamsARMANet['nResidueTaps'] = [5]  # Number of filter taps per layer
        hParamsARMANet['nSelectedNodes'] = None
        hParamsARMANet['tMax'] = 1  # Number of Jacobi iterations
        hParamsARMANet['bias'] = True  # Decide whether to include a bias term
        hParamsARMANet['nonlinearity'] = nn.ReLU  # Selected nonlinearity
        # \\\ Architecture parameters
        hParamsARMANet['poolingFunction'] = gml.NoPool  # Summarizing function
        hParamsARMANet['poolingSize'] = [1]  # These are ignored when there is no pooling,
        # better set it to 1 to make everything slightly faster
        hParamsARMANet['dimLayersMLP'] = [1]  # Dimension of the fully
        # connected layers after all the aggregation layers

        # Graph structure
        hParamsARMANet['GSO'] = None  # To be determined later on, based on data
        hParamsARMANet['order'] = None  # Not used because there is no pooling

        # \\\ Save Values:
        # writeVarValues(varsFile, hParamsARMANet)
        modelList += [hParamsARMANet['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 6: Graph Attention Network
    # \\\\\\\\\\\\

    if doGAT:
        hParamsGAT = {}  # Hyperparameters (hParams) for the Selection GNN (SelGNN)

        hParamsGAT['name'] = 'GAT'  # Name of the architecture

        # Chosen architecture
        hParamsGAT['archit'] = archit.GraphAttentionNetwork

        # \\\ Architecture parameters
        hParamsGAT['dimNodeSignals'] = nFeatures  # Features per layer
        hParamsGAT['nAttentionHeads'] = [1]  # Number of attention heads
        hParamsGAT['nonlinearity'] = nn.functional.relu  # Selected nonlinearity
        hParamsGAT['poolingFunction'] = gml.NoPool  # Summarizing function
        hParamsGAT['poolingSize'] = [1]  # alpha-hop neighborhood that is
        # affected by the summary
        hParamsGAT['dimLayersMLP'] = [1]  # Dimension of the fully
        # connected layers after the GCN layers
        hParamsGAT['bias'] = True  # Decide whether to include a bias term
        hParamsGAT['nSelectedNodes'] = None

        # Graph structure
        hParamsGAT['GSO'] = None  # To be determined later on, based on data
        hParamsGAT['order'] = None  # Not used because there is no pooling

        # \\\ Save Values:
        # writeVarValues(varsFile, hParamsGAT)
        modelList += [hParamsGAT['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 7 GCAT
    # \\\\\\\\\\\\

    if doGCAT:
        hParamsGCAT = {}  # Hyperparameters (hParams) for the Selection GNN (SelGNN)

        hParamsGCAT['name'] = 'GCAT'  # Name of the architecture

        # Chosen architecture
        hParamsGCAT['archit'] = archit.GraphConvolutionAttentionNetwork

        # \\\ Architecture parameters
        hParamsGCAT['dimNodeSignals'] = nFeatures  # Features per layer
        hParamsGCAT['nFilterTaps'] = nShifts  # Number of filter taps per layer
        hParamsGCAT['nAttentionHeads'] = [1]  # Number of attention heads per layer
        hParamsGCAT['nonlinearity'] = nn.functional.relu  # Selected nonlinearity
        hParamsGCAT['poolingFunction'] = gml.NoPool  # Summarizing function
        hParamsGCAT['poolingSize'] = [1]  # alpha-hop neighborhood that is
        # affected by the summary
        hParamsGCAT['dimLayersMLP'] = [1]  # Dimension of the fully
        # connected layers after the GCN layers
        hParamsGCAT['bias'] = True  # Decide whether to include a bias term
        hParamsGCAT['nSelectedNodes'] = None

        # Graph structure
        hParamsGCAT['GSO'] = None  # To be determined later on, based on data
        hParamsGCAT['order'] = None  # Not used because there is no pooling

        # \\\ Save Values:
        # writeVarValues(varsFile, hParamsGCAT)
        modelList += [hParamsGCAT['name']]

    # \\\\\\\\\\\\
    # \\\ MODEL 8: Graph Attention Network with EVGF
    # \\\\\\\\\\\\

    if doEVGAT:
        hParamsEVGAT = {}  # Hyperparameters (hParams) for the Selection GNN (SelGNN)

        hParamsEVGAT['name'] = 'EVGAT'  # Name of the architecture

        # Chosen architecture
        hParamsEVGAT['archit'] = archit.EdgeVariantAttention

        # \\\ Architecture parameters
        hParamsEVGAT['dimNodeSignals'] = [1, 32]  # Features per layer
        hParamsEVGAT['nFilterTaps'] = [5]  # Number of filter taps per layer
        hParamsEVGAT['nAttentionHeads'] = [1]  # Number of attention heads per layer
        hParamsEVGAT['nonlinearity'] = nn.functional.relu  # Selected nonlinearity
        hParamsEVGAT['poolingFunction'] = gml.NoPool  # Summarizing function
        hParamsEVGAT['poolingSize'] = [1]  # alpha-hop neighborhood that is
        # affected by the summary
        hParamsEVGAT['dimLayersMLP'] = [1]  # Dimension of the fully
        # connected layers after the GCN layers
        hParamsEVGAT['bias'] = True  # Decide whether to include a bias term
        hParamsEVGAT['nSelectedNodes'] = None

        # Graph structure
        hParamsEVGAT['GSO'] = None  # To be determined later on, based on data
        hParamsEVGAT['order'] = None  # Not used because there is no pooling

        # \\\ Save Values:
        # writeVarValues(varsFile, hParamsEVGAT)
        modelList += [hParamsEVGAT['name']]

    ###########
    # LOGGING #
    ###########

    # Options:
    doPrint = True  # Decide whether to print stuff while running
    doLogging = False  # Log into tensorboard
    doSaveVars = True  # Save (pickle) useful variables
    doFigs = True  # Plot some figures (this only works if doSaveVars is True)
    # Parameters:
    printInterval = 0  # After how many training steps, print the partial results
    #   0 means to never print partial results while training
    xAxisMultiplierTrain = 100  # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
    xAxisMultiplierValid = 20  # How many validation steps in between those shown,
    # same as above.
    figSize = 5  # Overall size of the figure that contains the plot
    lineWidth = 2  # Width of the plot lines
    markerShape = 'o'  # Shape of the markers
    markerSize = 3  # Size of the markers

    # \\\ Save values:
    # writeVarValues(varsFile,
    #                {'doPrint': doPrint,
    #                 'doLogging': doLogging,
    #                 'doSaveVars': doSaveVars,
    #                 'doFigs': doFigs,
    #                 'saveDir': saveDir,
    #                 'printInterval': printInterval,
    #                 'figSize': figSize,
    #                 'lineWidth': lineWidth,
    #                 'markerShape': markerShape,
    #                 'markerSize': markerSize})

    # %%##################################################################
    #                                                                   #
    #                    SETUP                                          #
    #                                                                   #
    #####################################################################
    useGPU = True

    # \\\ Determine processing unit:
    if useGPU and torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
    # Notify:
    if doPrint:
        print("Device selected: %s" % device)

    # \\\ Logging options
    if doLogging:
        # If logging is on, load the tensorboard visualizer and initialize it
        from Utils.visualTools import Visualizer

        logsTB = os.path.join(saveDir, 'logsTB')
        logger = Visualizer(logsTB, name='visualResults')

    # \\\ Save variables during evaluation.
    # We will save all the evaluations obtained for each of the trained models.
    # It basically is a dictionary, containing a list. The key of the
    # dictionary determines the model, then the first list index determines
    # which split realization. Then, this will be converted to numpy to compute
    # mean and standard deviation (across the split dimension).
    accBest = {}  # Accuracy for the best model
    f1_best = {}  # Accuracy for the best model
    roc_best = {}  # Accuracy for the best model
    accLast = {}  # Accuracy for the last model
    for thisModel in modelList:  # Create an element for each split realization,
        accBest[thisModel] = [None] * nDataSplits
        f1_best[thisModel] = [None] * nDataSplits
        roc_best[thisModel] = [None] * nDataSplits
        accLast[thisModel] = [None] * nDataSplits

    ####################
    # TRAINING OPTIONS #
    ####################

    # Training phase. It has a lot of options that are input through a
    # dictionary of arguments.
    # The value of these options was decided above with the rest of the parameters.
    # This just creates a dictionary necessary to pass to the train function.

    trainingOptions = {}

    if doLogging:
        trainingOptions['logger'] = logger
    if doSaveVars:
        trainingOptions['saveDir'] = saveDir
    if doPrint:
        trainingOptions['printInterval'] = printInterval
    if doLearningRateDecay:
        trainingOptions['learningRateDecayRate'] = learningRateDecayRate
        trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
    trainingOptions['validationInterval'] = validationInterval

    training_results = {}

    # Start generating a new data split for each of the number of data splits that
    # we previously specified
    if path.exists(file_name) and os.stat(file_name).st_size > 0:
        with open(file_name, 'r') as f:
            training_results = json.load(f)

    #   Load the data, which will give a specific split
    data = Utils.dataTools.Authorship(authorName, ratioTrain, ratioValid, dataPath)

    for combination in combinations:
        hParamsGCAT['nAttentionHeads'] = [combination]

        if str(combination) in list(training_results.keys()):
            print("SKIPPING COMBINATION: %s" % str(combination))
            continue

        if doPrint:
            print("COMBINATION: %s" % str(combination))

        training_results[str(combination)] = []

        # %%##################################################################
        #                                                                   #
        #                    DATA SPLIT REALIZATION                         #
        #                                                                   #
        #####################################################################

        # Start generating a new data split for each of the number of data splits that
        # we previously specified

        for split in range(nDataSplits):

            # %%##################################################################
            #                                                                   #
            #                    DATA HANDLING                                  #
            #                                                                   #
            #####################################################################

            ############
            # DATASETS #
            ############

            data.get_split(authorName, ratioTrain, ratioValid)

            # Now, we are in position to know the number of nodes (for now; this might
            # change later on when the graph is created and the options on whether to
            # make it connected, etc., come into effect)
            nNodes = data.selectedAuthor['all']['wordFreq'].shape[1]

            #########
            # GRAPH #
            #########

            # Create graph
            nodesToKeep = []  # here we store the list of nodes kept after all
            # modifications to the graph, so we can then update the data samples
            # accordingly; since lists are passed as pointers (mutable objects)
            # we can store the node list without necessary getting an output to the
            # function
            G = graphTools.Graph('fuseEdges', nNodes,
                                 data.selectedAuthor['train']['WAN'],
                                 'sum', graphNormalizationType, keepIsolatedNodes,
                                 forceUndirected, forceConnected, nodesToKeep)
            G.computeGFT()  # Compute the GFT of the stored GSO

            # And re-update the number of nodes for changes in the graph (due to
            # enforced connectedness, for instance)
            nNodes = G.N
            nodesToKeep = np.array(nodesToKeep)
            # And re-update the data (keep only the nodes that are kept after isolated
            # nodes or nodes to make the graph connected have been removed)
            data.samples['train']['signals'] = \
                data.samples['train']['signals'][:, nodesToKeep]
            data.samples['valid']['signals'] = \
                data.samples['valid']['signals'][:, nodesToKeep]
            data.samples['test']['signals'] = \
                data.samples['test']['signals'][:, nodesToKeep]

            # Once data is completely formatted and in appropriate fashion, change its
            # type to torch and move it to the appropriate device
            data.astype(torch.float64)
            data.to(device)

            # %%##################################################################
            #                                                                   #
            #                    MODELS INITIALIZATION                          #
            #                                                                   #
            #####################################################################

            # This is the dictionary where we store the models (in a model.Model
            # class, that is then passed to training).
            modelsGNN = {}

            # If a new model is to be created, it should be called for here.

            if doPrint:
                print("Model initialization...", flush=True)

            for thisModel in modelList:

                # Get the corresponding parameter dictionary
                hParamsDict = deepcopy(eval('hParams' + thisModel))

                # Now, this dictionary has all the hyperparameters that we need to pass
                # to the architecture, but it also has the 'name' and 'archit' that
                # we do not need to pass them. So we are going to get them out of
                # the dictionary
                thisName = hParamsDict.pop('name')
                callArchit = hParamsDict.pop('archit')

                # If more than one graph or data realization is going to be carried out,
                # we are going to store all of thos models separately, so that any of
                # them can be brought back and studied in detail.
                if nDataSplits > 1:
                    thisName += 'G%02d' % split

                if doPrint:
                    print("\tInitializing %s..." % thisName,
                          end=' ', flush=True)

                ##############
                # PARAMETERS #
                ##############

                # \\\ Optimizer options
                #   (If different from the default ones, change here.)
                thisTrainer = trainer
                thisLearningRate = learningRate
                thisBeta1 = beta1
                thisBeta2 = beta2

                # \\\ Ordering
                if 'NodeVariantDeg' or 'HybEdgeVariantDeg' in thisName:
                    S, order = graphTools.permDegree(G.S.copy() / np.max(np.diag(G.E)))
                else:
                    S = G.S.copy() / np.max(np.real(G.E))
                # Do not forget to add the GSO to the input parameters of the archit
                hParamsDict['GSO'] = S
                # Add the number of nodes for the no-pooling part
                if '1Ly' in thisName:
                    hParamsDict['nSelectedNodes'] = [nNodes]
                elif '2Ly' in thisName:
                    hParamsDict['nSelectedNodes'] = [nNodes, nNodes]

                if 'FullEdgeVariant' in thisName:
                    hParamsDict['nSelectedNodes'] = [nNodes]
                    hParamsDict['nFilterNodes'] = [nNodes]

                if 'NodeVariantDeg' in thisName:
                    SelN_NV = np.floor(0.1 * nNodes)
                    hParamsDict['nNodeTaps'] = [SelN_NV.astype(int)]  # Number of node taps per layer
                    hParamsDict['nSelectedNodes'] = [nNodes]

                if 'HybEdgeVariantDeg' in thisName:
                    SelN_NV = np.floor(0.1 * nNodes)
                    hParamsDict['nFilterNodes'] = [SelN_NV.astype(int)]  # Number of node taps per layer
                    hParamsDict['nSelectedNodes'] = [nNodes]

                if 'ARMANet' in thisName:
                    hParamsDict['nSelectedNodes'] = [nNodes]

                if 'GAT' in thisName:
                    hParamsDict['nSelectedNodes'] = [nNodes]

                if 'GCAT' in thisName:
                    hParamsDict['nSelectedNodes'] = [nNodes]

                if 'EVGAT' in thisName:
                    hParamsDict['nSelectedNodes'] = [nNodes]
                ################
                # ARCHITECTURE #
                ################

                thisArchit = callArchit(**hParamsDict)
                thisArchit.to(device)

                #############
                # OPTIMIZER #
                #############

                if thisTrainer == 'ADAM':
                    thisOptim = optim.Adam(thisArchit.parameters(),
                                           lr=learningRate,
                                           betas=(beta1, beta2))
                elif thisTrainer == 'SGD':
                    thisOptim = optim.SGD(thisArchit.parameters(),
                                          lr=learningRate)
                elif thisTrainer == 'RMSprop':
                    thisOptim = optim.RMSprop(thisArchit.parameters(),
                                              lr=learningRate, alpha=beta1)

                # \\\ Ordering
                S, order = graphTools.permIdentity(G.S / np.max(np.diag(G.E)))
                # order is an np.array with the ordering of the nodes with respect
                # to the original GSO (the original GSO is kept in G.S).

                ########
                # LOSS #
                ########

                # thisLossFunction = loss.adaptExtraDimensionLoss(lossFunction)

                #########
                # MODEL #
                #########

                modelCreated = model.Model(thisArchit,
                                           lossFunction,
                                           thisOptim,
                                           thisName, saveDir, order)

                modelsGNN[thisName] = modelCreated

                # writeVarValues(varsFile,
                #                {'name': thisName,
                #                 'thisTrainer': thisTrainer,
                #                 'thisLearningRate': thisLearningRate,
                #                 'thisBeta1': thisBeta1,
                #                 'thisBeta2': thisBeta2})

                if doPrint:
                    print("OK")

            if doPrint:
                print("Model initialization... COMPLETE")

            # %%##################################################################
            #                                                                   #
            #                    TRAINING                                       #
            #                                                                   #
            #####################################################################

            ############
            # TRAINING #
            ############

            # On top of the rest of the training options, we pass the identification
            # of this specific data split realization.

            if nDataSplits > 1:
                trainingOptions['graphNo'] = split

            # This is the function that trains the models detailed in the dictionary
            # modelsGNN using the data data, with the specified training options.
            train.MultipleModels(modelsGNN, data,
                                 nEpochs=nEpochs, batchSize=batchSize,
                                 **trainingOptions)

            # %%##################################################################
            #                                                                   #
            #                    EVALUATION                                     #
            #                                                                   #
            #####################################################################

            # Now that the model has been trained, we evaluate them on the test
            # samples.

            # We have two versions of each model to evaluate: the one obtained
            # at the best result of the validation step, and the last trained model.

            ########
            # DATA #
            ########

            xTest, yTest = data.getSamples('test')

            ##############
            # BEST MODEL #
            ##############

            if doPrint:
                print("Total testing accuracy (Best):", flush=True)

            for key in modelsGNN.keys():
                # Update order and adapt dimensions (this data has one input feature,
                # so we need to add that dimension)
                xTestOrdered = xTest[:, modelsGNN[key].order].unsqueeze(1)

                with torch.no_grad():
                    # Process the samples
                    yHatTest = modelsGNN[key].archit(xTestOrdered)
                    # yHatTest is of shape
                    #   testSize x numberOfClasses
                    # We compute the accuracy
                    thisAccBest = data.evaluate(yHatTest, yTest)
                    # convert to binary for metrics
                    yHatTest = np.round(yHatTest)
                    yHatTest = yHatTest.squeeze(1)

                    f1_score_test = f1_score(yTest, yHatTest, average='macro')
                    fpr, tpr, _ = roc_curve(yTest, yHatTest)
                    roc_auc = auc(fpr, tpr)

                if doPrint:
                    print("%s: %4.2f%%" % (key, thisAccBest * 100.), flush=True)

                # Save value
                # writeVarValues(varsFile,
                #                {'accBest%s' % key: thisAccBest})

                # Now check which is the model being trained
                for thisModel in modelList:
                    # If the name in the modelList is contained in the name with
                    # the key, then that's the model, and save it
                    # For example, if 'SelGNNDeg' is in thisModelList, then the
                    # correct key will read something like 'SelGNNDegG01' so
                    # that's the one to save.
                    if thisModel in key:
                        accBest[thisModel][split] = thisAccBest
                        f1_best[thisModel][split] = f1_score_test
                        roc_best[thisModel][split] = roc_auc
                        # This is so that we can later compute a total accuracy with
                    # the corresponding error.

        training_results[str(combination)] = {"acc": list(accBest['GCAT']), "f1": list(f1_best['GCAT']),
                                              "auc": list(roc_best['GCAT'])}

        with open('{1}{0}.txt'.format(authorName, BASE_FILE_NAME), 'w+') as outfile:
            json.dump(training_results, outfile)

        del yHatTest
        del thisAccBest

    ClusterUtils.delete_from_active(ACTIVE_AUTHORS_FILE, authorName)
except KeyboardInterrupt:
    delete_active_author(authorName, ACTIVE_AUTHORS_FILE, None, None)
    if training_results is not None:
        with open('{1}{0}.txt'.format(authorName, BASE_FILE_NAME), 'w+') as outfile:
            json.dump(training_results, outfile)
finally:
    delete_active_author(authorName, ACTIVE_AUTHORS_FILE, None, None)
