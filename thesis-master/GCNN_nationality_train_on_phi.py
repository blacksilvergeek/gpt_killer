# 2019/02/27~2019/03/04.
# Fernando Gama, fgama@seas.upenn.edu

# Authorship attribution problem, testing the following models
#   Spectral GNN
#   Polynomial GNN
#   Node Variant GNN (Deg, EDS, SP)
#   Edge Variant GNN
#   Hybrid Edge Variant GNN (Deg, EDS, SP)

# We will not consider any kind of pooling, and just one layer architectures.
# The number of parameters of every architecture will be tried to be kept
# the same (or, at least, the same order).

# The problem is that of authorship attribution. This runs several realizations
# to average out the randomness in the split of the training/test datasets.

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
#       both in pickle and in Matlab(R) format. These variables are saved in a
#       trainVars directory.
#   - If desired, plots the training and validation loss and evaluation
#       performance for each of the models, together with the training loss and
#       validation evaluation performance for all models. The summarizing
#       variables used to construct the plots are also saved in both pickle and
#       Matlab(R) format. These plots (and variables) are in a figs directory.

# %%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

# import matplotlib
#
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['font.family'] = 'serif'
# import matplotlib.pyplot as plt
import datetime
# \\\ Standard libraries:
import itertools
import json
import os.path
from os import path

import numpy as np
import pandas as pd
import torch

# from scipy.io import savemat
from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_curve, average_precision_score, auc
from sklearn.svm import SVC

torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
from ast import literal_eval as make_tuple

# \\\ Own libraries:
import Utils.graphTools as graphTools
import Utils.dataTools
import Utils.graphML as gml
import Modules.architectures_sigmoid as archit
import Modules.model as model
import Modules.train as train

# \\\ Separate functions:
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed

# %%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']

BASE_FILE_NAME = 'GCNN_nationality_phi_results_0.6'

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
doSaveVars = True  # Save (pickle) useful variables
doFigs = True  # Plot some figures (this only works if doSaveVars is True)

# \\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

file_name = BASE_FILE_NAME + ".txt"
svc_file_name = "SVC_nationality_phi_results_0.6.txt"

# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + today
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
# Create the file where all the (hyper)parameters and results will be saved.
varsFile = os.path.join(saveDir, 'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

# \\\ Save seeds for reproducibility
#   PyTorch seeds
torchState = torch.get_rng_state()
torchSeed = torch.initial_seed()
#   Numpy seeds
numpyState = np.random.RandomState().get_state()
#   Collect all random states
randomStates = []
randomStates.append({})
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
randomStates.append({})
randomStates[1]['module'] = 'torch'
randomStates[1]['state'] = torchState
randomStates[1]['seed'] = torchSeed
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(randomStates, saveDir)

########
# DATA #
########
# Possible authors: (just use the names in ' ')
# jacob 'abbott', robert louis 'stevenson', louisa may 'alcott',
# horatio 'alger', james 'allen', jane 'austen', emily 'bronte', james 'cooper',
# charles 'dickens', hamlin 'garland', nathaniel 'hawthorne', henry 'james',
# herman 'melville', 'page', herny 'thoreau', mark 'twain',
# arthur conan 'doyle', washington 'irving', edgar allan 'poe',
# sarah orne 'jewett', edith 'wharton'


nFeatures = [1, 32]  # F: number of output features of the only layer
nShifts = [4]  # K: number of shift tap

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

# \\\ Save values:
writeVarValues(varsFile,
               {'authorName': 'gender',
                'nClasses': nClasses,
                'ratioTrain': ratioTrain,
                'ratioValid': ratioValid,
                'nDataSplits': nDataSplits,
                'graphNormalizationType': graphNormalizationType,
                'keepIsolatedNodes': keepIsolatedNodes,
                'forceUndirected': forceUndirected,
                'forceConnected': forceConnected})

############
# TRAINING #
############

# \\\ Individual model training options
trainer = 'ADAM'  # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.001  # In all options
beta1 = 0.9  # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999  # ADAM option only

# \\\ Loss function choice
# lossFunction = nn.CrossEntropyLoss()  # This applies a softmax before feeding
lossFunction = nn.BCELoss()  # This applies a softmax before feeding
# it into the NLL, so we don't have to apply the softmax ourselves.

# \\\ Overall training options
nEpochs = 10  # Number of epochs
batchSize = 32  # Batch size
doLearningRateDecay = False  # Learning rate decay
learningRateDecayRate = 0.9  # Rate
learningRateDecayPeriod = 1  # How many epochs after which update the lr
validationInterval = 5  # How many training steps to do the validation

# \\\ Save values
writeVarValues(varsFile,
               {'trainer': trainer,
                'learningRate': learningRate,
                'beta1': beta1,
                'lossFunction': lossFunction,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'doLearningRateDecay': doLearningRateDecay,
                'learningRateDecayRate': learningRateDecayRate,
                'learningRateDecayPeriod': learningRateDecayPeriod,
                'validationInterval': validationInterval})

#################
# ARCHITECTURES #
#################

# Select which architectures to train and run

# Select desired node-orderings (for hybrid EV and node variant EV) so that
# the selected privileged nodes follows this criteria
doDegree = True
doSpectralProxies = True
doEDS = True

# Select desired architectures
doPolynomialGNN = True

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. That is, any new architecture in this part, needs also
# to be coded later on. This is just to be easy to change the parameters once
# the architecture is created. Do not forget to add the name of the architecture
# to modelList.

modelList = []

# Parameters for all models, so we don't need to be changing each one in each
# of the models (this guarantees comparable computational complexity)

# \\\\\\\\\\\\
# \\\ MODEL 2: Polynomial GNN
# \\\\\\\\\\\\

if doPolynomialGNN:
    hParamsPolynomial = {'name': 'PolynomiGNN', 'F': nFeatures, 'K': nShifts, 'bias': True, 'sigma': nn.ReLU,
                         'rho': gml.NoPool, 'alpha': [1], 'dimLayersMLP': [nClasses]}  # Hyperparameters (hParams)

    # \\\ Architecture parameters
    # affected by the summary
    # connected layers after the GCN layers

    # \\\ Save Values:
    writeVarValues(varsFile, hParamsPolynomial)
    modelList += [hParamsPolynomial['name']]

###########
# LOGGING #
###########

# Parameters:
printInterval = 0  # After how many training steps, print the partial results
#   0 means to never print partial results while training
xAxisMultiplierTrain = 10  # How many training steps in between those shown in
# the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 2  # How many validation steps in between those shown,
# same as above.

# \\\ Save values:
writeVarValues(varsFile,
               {'doPrint': doPrint,
                'doLogging': doLogging,
                'doSaveVars': doSaveVars,
                'doFigs': doFigs,
                'saveDir': saveDir,
                'printInterval': printInterval})

# %%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

# \\\ Determine processing unit:
if torch.cuda.is_available():
    device = 'cuda:0'
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
accLast = {}  # Accuracy for the last model
for thisModel in modelList:  # Create an element for each split realization,
    accBest[thisModel] = [None] * nDataSplits
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

# %%##################################################################
#                                                                   #
#                    DATA SPLIT REALIZATION                         #
#                                                                   #
#####################################################################
F = [nFeatures]
K = [nShifts]
# F = [16, 32, 64]
# K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]

combinations = list(itertools.product(F, K))

training_results = []
svc_results = []

# Start generating a new data split for each of the number of data splits that
# we previously specified
if path.exists(file_name) and os.stat(file_name).st_size > 0:
    with open(file_name, 'r') as f:
        training_results = json.load(f)

#   Load the data, which will give a specific split
data = Utils.dataTools.AutorshipNationality(ratioTrain, ratioValid, dataPath)


# %%##################################################################
#                                                                   #
#                    LOAD PHI MATRICES                              #
#                                                                   #
#####################################################################

def load_phi(data, phi_matrix_path='EdgeVariGNN_nationality_phi.txt', percentage=0.4, eps=0.0001):
    with open(phi_matrix_path, 'r') as f:
        file = json.load(f)

        phi_whole = np.array(file['phi'])
        data.reduce_dim(file['nodes'])

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

        print(
            "PHI matrix for loaded. Number of dimensions: {0}".format(len(ind_X)))

        return phi, np.array(ind_X)


def get_results(y_hat, y_val):
    totalErrors = np.sum(np.abs(y_hat - y_val) > 1e-9)
    accuracy = 1 - totalErrors.item() / len(y_val)

    f1 = f1_score(y_val, y_hat)
    fpr, tpr, _ = roc_curve(y_val, y_hat)
    roc_auc = auc(fpr, tpr)
    average_precision = average_precision_score(y_val, y_hat)

    result = {'acc': accuracy, 'f1': f1, 'auc': roc_auc, 'prec': average_precision}

    return result


def evaluate_svc(arch, data):
    X_valid, y_val = data.getSamples('valid')
    X_valid = preprocessing.scale(X_valid)

    y_hat = arch.predict(X_valid)

    return get_results(y_hat, y_val.numpy())


# with open('EdgeVariGNN_nationality_phi.txt', 'r') as f:
#     file = json.load(f)
#     phi = np.array(file['phi'])
#
#     indices_to_zero = [x for x in
#                        np.argwhere(np.abs(phi) < np.max(np.abs(phi)) - 0.3 * np.max(np.abs(phi)))]
#
#     for x, y in indices_to_zero:
#         phi[x, y] = 0
#
#     phi_matrix = np.copy(phi)
#     nodes_to_keep = np.array(file['nodes'])
#     function_words = np.array(data.functionWords)
#
# nonzero_el_count = np.count_nonzero(phi_matrix)
#
# if doPrint:
#     print("Loaded matrix PHI with shape: ({0}, {1}) and {2} non zero elements".format(phi_matrix.shape[0],
#                                                                                       phi_matrix.shape[1],
#                                                                                       nonzero_el_count))
#
# if nonzero_el_count < 1:
#     print("Number of non zero elements in Matrix PHI is too small: {0}".format(nonzero_el_count))
#     exit(-1)

# %%##################################################################

phi_matrix, indices = load_phi(data, percentage=0.6)

for combination in combinations:
    #
    # if str(combination) in list(training_results.keys()):
    #     if len(list(filter(None, training_results[str(combination)]))) >= 10:
    #         print("SKIPPING COMBINATION: %s" % str(combination))
    #         continue
    #
    #     training_results[str(combination)]['acc'] = list(filter(None, training_results[str(combination)]['acc']))
    #     training_results[str(combination)]['f1'] = list(filter(None, training_results[str(combination)]['f1']))
    #     # training_results[str(combination)]['auc'] = list(filter(None, training_results[str(combination)]['auc']))
    #
    #     for idx, item in enumerate(training_results[str(combination)]['acc']):
    #         accBest[hParamsPolynomial['name']][idx] = item
    #         f1_best[hParamsPolynomial['name']][idx] = training_results[str(combination)]['f1'][idx]
    #         # roc_best[hParamsPolynomial['name']][idx] = training_results[str(combination)]['auc'][idx]
    #
    # else:

    if doPrint:
        print("COMBINATION: %s" % str(combination))

    for split in range(nDataSplits):
        data.get_split(ratioTrain, ratioValid)
        data.reduce_dim(indices)

        ###################################################################
        #                                                                   #
        #                    DATA HANDLING                                  #
        #                                                                   #
        #####################################################################

        ############
        # DATASETS #
        ############

        # And re-update the number of nodes for changes in the graph (due to
        # enforced connectedness, for instance)
        nNodes = phi_matrix.shape[0]

        # nodesToKeep = np.array(nodes_to_keep)
        # # And re-update the data (keep only the nodes that are kept after isolated
        # # nodes or nodes to make the graph connected have been removed)
        # data.samples['train']['signals'] = \
        #     data.samples['train']['signals'][:, nodes_to_keep]
        # data.samples['valid']['signals'] = \
        #     data.samples['valid']['signals'][:, nodes_to_keep]
        # data.samples['test']['signals'] = \
        #     data.samples['test']['signals'][:, nodes_to_keep]

        # Once data is completely formatted and in appropriate fashion, change its
        # type to torch and move it to the appropriate device
        data.astype(torch.float64)
        data.to(device)

        #####################################################################
        #                                                                   #
        #                    MODELS INITIALIZATION                          #
        #                                                                   #
        #####################################################################

        # This is the dictionary where we store the models (in a model.Model
        # class, that is then passed to training).
        modelsGNN = {}

        # If a new model is to be created, it should be called for here.

        # \\\\\\\\\\
        # \\\ MODEL 2: Polynomial GNN
        # \\\\\\\\\\\\

        if doPolynomialGNN:
            thisName = hParamsPolynomial['name']

            if nDataSplits > 1:
                thisName += 'G%02d' % split

            ##############
            # PARAMETERS #
            ##############

            # \\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            # compute the Eigenvalues of matrix
            e, V = np.linalg.eig(phi_matrix)
            # \\\ Ordering
            highest_eig_val = np.max(np.diag(e)).real

            if highest_eig_val == 0:
                S, order = graphTools.permIdentity(phi_matrix)
            else:
                S, order = graphTools.permIdentity(phi_matrix / highest_eig_val)
            # order is an np.array with the ordering of the nodes with respect
            # to the original GSO (the original GSO is kept in G.S).

            ################
            # ARCHITECTURE #
            ################
            # Override parameters with grid parameters.
            hParamsPolynomial['F'] = nFeatures
            hParamsPolynomial['K'] = nShifts
            hParamsPolynomial['N'] = [nNodes]

            if doPrint:
                print('COMBINATION {0}, {1}'.format(str(hParamsPolynomial['F']), str(hParamsPolynomial['K'])))

            thisArchit = archit.SelectionGNN(  # Graph filtering
                hParamsPolynomial['F'],
                hParamsPolynomial['K'],
                hParamsPolynomial['bias'],
                # Nonlinearity
                hParamsPolynomial['sigma'],
                # Pooling
                hParamsPolynomial['N'],
                hParamsPolynomial['rho'],
                hParamsPolynomial['alpha'],
                # MLP
                hParamsPolynomial['dimLayersMLP'],
                # Structure
                S)
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr=learningRate, betas=(beta1, beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr=learningRate, alpha=beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction

            #########
            # MODEL #
            #########

            Polynomial = model.Model(thisArchit, thisLossFunction, thisOptim,
                                     thisName, saveDir, order)

            modelsGNN[thisName] = Polynomial

            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        ###################################################################
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

        # TRAIN SVM
        X, y = data.getSamples('train')
        X = preprocessing.scale(X)

        svc = SVC()

        svc.fit(X, y)
        # ###################################################################
        # #                                                                   #
        # #                    EMBEDDING HOOK                                 #
        # #                                                                   #
        # #####################################################################
        # import collections
        # from functools import partial
        #
        # activations = collections.defaultdict(list)
        #
        #
        # def save_activation(name, mod, inp, out):
        #     activations[name].append(out.cpu())
        #
        #
        # # Registering hooks for all the Conv2d layers
        # # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
        # # called repeatedly at different stages of the forward pass (like RELUs), this will save different
        # # activations. Editing the forward pass code to save activations is the way to go for these cases.
        # for name, m in thisArchit.named_modules():
        #     if name.strip() == "GFL.2":
        #         # partial to assign the layer name to each hook
        #         m.register_forward_hook(partial(save_activation, name))

        ###################################################################
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

                yHatTest = np.round(yHatTest)
                yHatTest = yHatTest.squeeze(1).numpy()

                res = get_results(yHatTest, yTest.numpy())
                res['acc'] = thisAccBest
                training_results.append(res)

                svc_result = evaluate_svc(svc, data)
                svc_results.append(svc_result)
            if doPrint:
                print("%s: %4.2f%%" % (key, thisAccBest * 100.), flush=True)

            # Save value
            writeVarValues(varsFile,
                           {'accBest%s' % key: thisAccBest})

            # Now check which is the model being trained
            for thisModel in modelList:
                # If the name in the modelList is contained in the name with
                # the key, then that's the model, and save it
                # For example, if 'SelGNNDeg' is in thisModelList, then the
                # correct key will read something like 'SelGNNDegG01' so
                # that's the one to save.
                if thisModel in key:
                    accBest[thisModel][split] = thisAccBest
                # This is so that we can later compute a total accuracy with
                # the corresponding error.

        with open(file_name, 'w+') as outfile:
            json.dump(training_results, outfile)

        with open(svc_file_name, 'w+') as outfile:
            json.dump(svc_results, outfile)
