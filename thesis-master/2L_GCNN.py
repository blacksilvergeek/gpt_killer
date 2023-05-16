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

# \\\ Standard libraries:
import itertools
import json
import os
import os.path
from os import path
import numpy as np
# import matplotlib
#
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['font.family'] = 'serif'
# import matplotlib.pyplot as plt
import pickle
import datetime
# from scipy.io import savemat

import torch

torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

# \\\ Own libraries:
import Utils.graphTools as graphTools
import Utils.dataTools
import Utils.graphML as gml
import Modules.architectures as archit
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

BASE_FILE_NAME = 'GNN_Polynomial_2layers_results_'

thisFilename = 'authorEdgeNets'  # This is the general name of all related files

saveDirRoot = 'experiments'  # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename)  # Dir where to save all
# the results from each run
dataDir = 'authorData'  # Data directory
dataFilename = 'authorshipData.mat'  # Data filename
dataPath = os.path.join(dataDir, dataFilename)  # Data path

# \\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
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
nFeatures = 16  # F: number of output features of the only layer
nShifts = 7  # K: number of shift taps

authorName = 'dickens'
# Possible authors: (just use the names in ' ')
# jacob 'abbott', robert louis 'stevenson', louisa may 'alcott',
# horatio 'alger', james 'allen', jane 'austen', emily 'bronte', james 'cooper',
# charles 'dickens', hamlin 'garland', nathaniel 'hawthorne', henry 'james',
# herman 'melville', 'page', herny 'thoreau', mark 'twain',
# arthur conan 'doyle', washington 'irving', edgar allan 'poe',
# sarah orne 'jewett', edith 'wharton'
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
               {'authorName': authorName,
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
nEpochs = 25  # Number of epochs
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
    hParamsPolynomial = {'name': 'PolynomiGNN', 'F': [1, nFeatures], 'K': [nShifts], 'bias': True, 'sigma': nn.ReLU,
                         'rho': gml.NoPool, 'alpha': [1, 1], 'dimLayersMLP': [nClasses]}  # Hyperparameters (hParams)

    # \\\ Architecture parameters
    # affected by the summary
    # connected layers after the GCN layers

    # \\\ Save Values:
    writeVarValues(varsFile, hParamsPolynomial)
    modelList += [hParamsPolynomial['name']]

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

F = [16, 32, 64]
K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]

combinations = list(itertools.product(F, K))

training_results = {}

# Start generating a new data split for each of the number of data splits that
# we previously specified

if path.exists("{0}{1}.txt".format(BASE_FILE_NAME, authorName)):
    with open("{0}{1}.txt".format(BASE_FILE_NAME, authorName), 'r') as f:
        training_results = json.load(f)

#   Load the data, which will give a specific split
data = Utils.dataTools.Authorship(authorName, ratioTrain, ratioValid, dataPath)

for combination in combinations:
    # if str(combination) in ['([1, 8], [1])', '([1, 8], [2])', '([1, 8], [3])', '([1, 8], [4])', '([1, 8], [5])',
    #                         '([1, 8], [6])']:
    #     continue

    if str(combination) in list(training_results.keys()):
        print("SKIPPING COMBINATION: %s" % str(combination))
        continue

    if doPrint:
        print("COMBINATION: %s" % str(combination))

    training_results[str(combination)] = []

    for split in range(nDataSplits):

        ###################################################################
        #                                                                   #
        #                    DATA HANDLING                                  #
        #                                                                   #
        #####################################################################

        ############
        # DATASETS #
        ############

        # if split is not 0:
        #     data.get_split(authorName, ratioTrain, ratioValid)

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

        ##################################################################
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

            # \\\ Ordering
            S, order = graphTools.permIdentity(G.S / np.max(np.diag(G.E)))
            # order is an np.array with the ordering of the nodes with respect
            # to the original GSO (the original GSO is kept in G.S).

            ################
            # ARCHITECTURE #
            ################
            # Override parameters with grid parameters.
            hParamsPolynomial['F'] = [1, nFeatures, combination[0]]
            hParamsPolynomial['K'] = [nShifts, combination[1]]
            hParamsPolynomial['N'] = [nNodes, nNodes]

            if doPrint:
                print('')
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

    training_results[str(combination)] = list(accBest['PolynomiGNN'])

    with open('{1}{0}.txt'.format(authorName, BASE_FILE_NAME), 'w+') as outfile:
        json.dump(training_results, outfile)

        # ##############
        # # LAST MODEL #
        # ##############
        #
        # # And repeat for the last model
        #
        # if doPrint:
        #     print("Total testing accuracy (Last):", flush=True)
        #
        # # Update order and adapt dimensions
        # for key in modelsGNN.keys():
        #     modelsGNN[key].load(label='Last')
        #     xTestOrdered = xTest[:, modelsGNN[key].order].unsqueeze(1)
        #
        #     with torch.no_grad():
        #         # Process the samples
        #         yHatTest = modelsGNN[key].archit(xTestOrdered)
        #         # yHatTest is of shape
        #         #   testSize x numberOfClasses
        #         # We compute the accuracy
        #         thisAccLast = data.evaluate(yHatTest, yTest)
        #
        #     if doPrint:
        #         print("%s: %4.2f%%" % (key, thisAccLast * 100), flush=True)
        #
        #     # Save values:
        #     writeVarValues(varsFile,
        #                    {'accLast%s' % key: thisAccLast})
        #     # And repeat for the last model:
        #     for thisModel in modelList:
        #         if thisModel in key:
        #             accLast[thisModel][split] = thisAccLast.item()
    # # %%##################################################################
    #
    # ############################
    # # FINAL EVALUATION RESULTS #
    # ############################
    #
    # # Now that we have computed the accuracy of all runs, we can obtain a final
    # # result (mean and standard deviation)
    #
    #
    # meanAccBest = {}  # Mean across data splits
    # meanAccLast = {}  # Mean across data splits
    # stdDevAccBest = {}  # Standard deviation across data splits
    # stdDevAccLast = {}  # Standard deviation across data splits
    #
    # if doPrint:
    #     print("\nFinal evaluations (%02d data splits)" % (nDataSplits))
    #
    # for thisModel in modelList:
    #     # Convert the lists into a nDataSplits vector
    #     accBest[thisModel] = np.array(accBest[thisModel])
    #     # accLast[thisModel] = np.array(accLast[thisModel])
    #
    #     # And now compute the statistics (across graphs)
    #     meanAccBest[thisModel] = np.mean(accBest[thisModel])
    #     # meanAccLast[thisModel] = np.mean(accLast[thisModel])
    #     stdDevAccBest[thisModel] = np.std(accBest[thisModel])
    #     # stdDevAccLast[thisModel] = np.std(accLast[thisModel])
    #
    #     # And print it:
    #     if doPrint:
    #         print("\t%s: %6.2f%% (+-%6.2f%%) [Best]" % (
    #             # print("\t%s: %6.2f%% (+-%6.2f%%) [Best] %6.2f%% (+-%6.2f%%) [Last]" % (
    #             thisModel,
    #             meanAccBest[thisModel] * 100,
    #             stdDevAccBest[thisModel] * 100)
    #               # meanAccLast[thisModel] * 100,
    #               # stdDevAccLast[thisModel] * 100)
    #               )
    #
    #     # Save values
    #     writeVarValues(varsFile,
    #                    {'meanAccBest%s' % thisModel: meanAccBest[thisModel],
    #                     'stdDevAccBest%s' % thisModel: stdDevAccBest[thisModel],
    #                     'meanAccLast%s' % thisModel: meanAccLast[thisModel],
    #                     'stdDevAccLast%s' % thisModel: stdDevAccLast[thisModel]})

# # %%##################################################################
# # Preprocess the activations
# # # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
# # activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}
# #
# # There is a dimenssion with all zeros, remove it
# activations = activations['GFL.2'][0].numpy()
# activations = activations[:, 1, :]
#
# # %%##################################################################
# # TSNE
# from sklearn.manifold import TSNE
# import time
# import seaborn as sns
# import pandas as pd
#
# time_start = time.time()
# tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=300)
# tsne_results = tsne.fit_transform(activations)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
#
# df_subset = pd.DataFrame()
# df_subset['tsne-2d-one'] = tsne_results[:, 0]
# df_subset['tsne-2d-two'] = tsne_results[:, 1]
# df_subset['y'] = yTest.numpy()
#
# plt.figure(figsize=(16, 10))
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 2),
#     data=df_subset,
#     legend="full",
#     alpha=1
# )
#
# plt.show()
#
# # %%##################################################################
# # PCA
# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(activations)
#
# df = pd.DataFrame()
#
# df['pca-one'] = pca_result[:, 0]
# df['pca-two'] = pca_result[:, 1]
# # df['pca-three'] = pca_result[:,2]
# df['y'] = yTest.numpy()
#
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#
# plt.figure(figsize=(16, 10))
# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     hue="y",
#     palette=sns.color_palette("hls", 2),
#     data=df,
#     legend="full",
#     alpha=0.9
# )
#
# plt.show()
# # %%##################################################################
# #                                                                   #
# #                    PLOT                                           #
# #                                                                   #
# #####################################################################
#
# # Finally, we might want to plot several quantities of interest
#
# if doFigs and doSaveVars:
#
#     ###################
#     # DATA PROCESSING #
#     ###################
#
#     # Again, we have training and validation metrics (loss and accuracy
#     # -evaluation-) for many runs, so we need to carefully load them and compute
#     # the relevant statistics from these realizations.
#
#     # \\\ SAVE SPACE:
#     # Create the variables to save all the realizations. This is, again, a
#     # dictionary, where each key represents a model, and each model is a list
#     # for each data split.
#     # Each data split, in this case, is not a scalar, but a vector of
#     # length the number of training steps (or of validation steps)
#     lossTrain = {}
#     evalTrain = {}
#     lossValid = {}
#     evalValid = {}
#     # Initialize the splits dimension
#     for thisModel in modelList:
#         lossTrain[thisModel] = [None] * nDataSplits
#         evalTrain[thisModel] = [None] * nDataSplits
#         lossValid[thisModel] = [None] * nDataSplits
#         evalValid[thisModel] = [None] * nDataSplits
#
#     # \\\ FIGURES DIRECTORY:
#     saveDirFigs = os.path.join(saveDir, 'figs')
#     # If it doesn't exist, create it.
#     if not os.path.exists(saveDirFigs):
#         os.makedirs(saveDirFigs)
#
#     # \\\ LOAD DATA:
#     # Path where the saved training variables should be
#     pathToTrainVars = os.path.join(saveDir, 'trainVars')
#     # Get all the training files:
#     allTrainFiles = next(os.walk(pathToTrainVars))[2]
#     # Go over each of them (this can't be empty since we are also checking for
#     # doSaveVars to be true, what guarantees that the variables have been
#     # saved.)
#     for file in allTrainFiles:
#         # Check that it is a pickle file
#         if '.pkl' in file:
#             # Open the file
#             with open(os.path.join(pathToTrainVars, file), 'rb') as fileTrainVars:
#                 # Load it
#                 thisVarsDict = pickle.load(fileTrainVars)
#                 # store them
#                 nBatches = thisVarsDict['nBatches']
#                 thisLossTrain = thisVarsDict['lossTrain']
#                 thisEvalTrain = thisVarsDict['evalTrain']
#                 thisLossValid = thisVarsDict['lossValid']
#                 thisEvalValid = thisVarsDict['evalValid']
#                 # This graph is, actually, the data split dimension
#                 if 'graphNo' in thisVarsDict.keys():
#                     thisG = thisVarsDict['graphNo']
#                 else:
#                     thisG = 0
#                 # And add them to the corresponding variables
#                 for key in thisLossTrain.keys():
#                     # This part matches each data realization (matched through
#                     # the graphNo key) with each specific model.
#                     for thisModel in modelList:
#                         if thisModel in key:
#                             lossTrain[thisModel][thisG] = thisLossTrain[key]
#                             evalTrain[thisModel][thisG] = thisEvalTrain[key]
#                             lossValid[thisModel][thisG] = thisLossValid[key]
#                             evalValid[thisModel][thisG] = thisEvalValid[key]
#     # Now that we have collected all the results, we have that each of the four
#     # variables (lossTrain, evalTrain, lossValid, evalValid) has a list for
#     # each key in the dictionary. This list goes through the data split.
#     # Each split realization is actually an np.array.
#
#     # \\\ COMPUTE STATISTICS:
#     # The first thing to do is to transform those into a matrix with all the
#     # realizations, so create the variables to save that.
#     meanLossTrain = {}
#     meanEvalTrain = {}
#     meanLossValid = {}
#     meanEvalValid = {}
#     stdDevLossTrain = {}
#     stdDevEvalTrain = {}
#     stdDevLossValid = {}
#     stdDevEvalValid = {}
#     # Initialize the variables
#     for thisModel in modelList:
#         # Transform into np.array
#         lossTrain[thisModel] = np.array(lossTrain[thisModel])
#         evalTrain[thisModel] = np.array(evalTrain[thisModel])
#         lossValid[thisModel] = np.array(lossValid[thisModel])
#         evalValid[thisModel] = np.array(evalValid[thisModel])
#         # Each of one of these variables should be of shape
#         # nDataSplits x numberOfTrainingSteps
#         # And compute the statistics
#         meanLossTrain[thisModel] = np.mean(lossTrain[thisModel], axis=0)
#         meanEvalTrain[thisModel] = np.mean(evalTrain[thisModel], axis=0)
#         meanLossValid[thisModel] = np.mean(lossValid[thisModel], axis=0)
#         meanEvalValid[thisModel] = np.mean(evalValid[thisModel], axis=0)
#         stdDevLossTrain[thisModel] = np.std(lossTrain[thisModel], axis=0)
#         stdDevEvalTrain[thisModel] = np.std(evalTrain[thisModel], axis=0)
#         stdDevLossValid[thisModel] = np.std(lossValid[thisModel], axis=0)
#         stdDevEvalValid[thisModel] = np.std(evalValid[thisModel], axis=0)
#
#     ####################
#     # SAVE FIGURE DATA #
#     ####################
#
#     # And finally, we can plot. But before, let's save the variables mean and
#     # stdDev so, if we don't like the plot, we can re-open them, and re-plot
#     # them, a piacere.
#     #   Pickle, first:
#     varsPickle = {}
#     varsPickle['nEpochs'] = nEpochs
#     varsPickle['nBatches'] = nBatches
#     varsPickle['meanLossTrain'] = meanLossTrain
#     varsPickle['stdDevLossTrain'] = stdDevLossTrain
#     varsPickle['meanEvalTrain'] = meanEvalTrain
#     varsPickle['stdDevEvalTrain'] = stdDevEvalTrain
#     varsPickle['meanLossValid'] = meanLossValid
#     varsPickle['stdDevLossValid'] = stdDevLossValid
#     varsPickle['meanEvalValid'] = meanEvalValid
#     varsPickle['stdDevEvalValid'] = stdDevEvalValid
#     with open(os.path.join(saveDirFigs, 'figVars.pkl'), 'wb') as figVarsFile:
#         pickle.dump(varsPickle, figVarsFile)
#     #   Matlab, second:
#     varsMatlab = {}
#     varsMatlab['nEpochs'] = nEpochs
#     varsMatlab['nBatches'] = nBatches
#     for thisModel in modelList:
#         varsMatlab['meanLossTrain' + thisModel] = meanLossTrain[thisModel]
#         varsMatlab['stdDevLossTrain' + thisModel] = stdDevLossTrain[thisModel]
#         varsMatlab['meanEvalTrain' + thisModel] = meanEvalTrain[thisModel]
#         varsMatlab['stdDevEvalTrain' + thisModel] = stdDevEvalTrain[thisModel]
#         varsMatlab['meanLossValid' + thisModel] = meanLossValid[thisModel]
#         varsMatlab['stdDevLossValid' + thisModel] = stdDevLossValid[thisModel]
#         varsMatlab['meanEvalValid' + thisModel] = meanEvalValid[thisModel]
#         varsMatlab['stdDevEvalValid' + thisModel] = stdDevEvalValid[thisModel]
#     savemat(os.path.join(saveDirFigs, 'figVars.mat'), varsMatlab)
#
#     ########
#     # PLOT #
#     ########
#
#     # Compute the x-axis
#     xTrain = np.arange(0, nEpochs * nBatches, xAxisMultiplierTrain)
#     xValid = np.arange(0, nEpochs * nBatches, \
#                        validationInterval * xAxisMultiplierValid)
#
#     # If we do not want to plot all the elements (to avoid overcrowded plots)
#     # we need to recompute the x axis and take those elements corresponding
#     # to the training steps we want to plot
#     if xAxisMultiplierTrain > 1:
#         # Actual selected samples
#         selectSamplesTrain = xTrain
#         # Go and fetch tem
#         for thisModel in modelList:
#             meanLossTrain[thisModel] = meanLossTrain[thisModel] \
#                 [selectSamplesTrain]
#             stdDevLossTrain[thisModel] = stdDevLossTrain[thisModel] \
#                 [selectSamplesTrain]
#             meanEvalTrain[thisModel] = meanEvalTrain[thisModel] \
#                 [selectSamplesTrain]
#             stdDevEvalTrain[thisModel] = stdDevEvalTrain[thisModel] \
#                 [selectSamplesTrain]
#     # And same for the validation, if necessary.
#     if xAxisMultiplierValid > 1:
#         selectSamplesValid = np.arange(0, len(meanLossValid[thisModel]), \
#                                        xAxisMultiplierValid)
#         for thisModel in modelList:
#             meanLossValid[thisModel] = meanLossValid[thisModel] \
#                 [selectSamplesValid]
#             stdDevLossValid[thisModel] = stdDevLossValid[thisModel] \
#                 [selectSamplesValid]
#             meanEvalValid[thisModel] = meanEvalValid[thisModel] \
#                 [selectSamplesValid]
#             stdDevEvalValid[thisModel] = stdDevEvalValid[thisModel] \
#                 [selectSamplesValid]
#
#     # \\\ LOSS (Training and validation) for EACH MODEL
#     for key in meanLossTrain.keys():
#         lossFig = plt.figure(figsize=(1.61 * 5, 1 * 5))
#         plt.errorbar(xTrain, meanLossTrain[key], yerr=stdDevLossTrain[key],
#                      color='#01256E', linewidth=2,
#                      marker='o', markersize=3)
#         plt.errorbar(xValid, meanLossValid[key], yerr=stdDevLossValid[key],
#                      color='#95001A', linewidth=2,
#                      marker='o', markersize=3)
#         plt.ylabel(r'Loss')
#         plt.xlabel(r'Training steps')
#         plt.legend([r'Training', r'Validation'])
#         plt.title(r'%s' % key)
#         lossFig.savefig(os.path.join(saveDirFigs, 'loss%s.pdf' % key),
#                         bbox_inches='tight')
#
#     # \\\ ACCURACY (Training and validation) for EACH MODEL
#     for key in meanEvalTrain.keys():
#         accFig = plt.figure(figsize=(1.61 * 5, 1 * 5))
#         plt.errorbar(xTrain, meanEvalTrain[key], yerr=stdDevEvalTrain[key],
#                      color='#01256E', linewidth=2,
#                      marker='o', markersize=3)
#         plt.errorbar(xValid, meanEvalValid[key], yerr=stdDevEvalValid[key],
#                      color='#95001A', linewidth=2,
#                      marker='o', markersize=3)
#         plt.ylabel(r'Accuracy')
#         plt.xlabel(r'Training steps')
#         plt.legend([r'Training', r'Validation'])
#         plt.title(r'%s' % key)
#         accFig.savefig(os.path.join(saveDirFigs, 'eval%s.pdf' % key),
#                        bbox_inches='tight')
#
#     # LOSS (training) for ALL MODELS
#     allLossTrain = plt.figure(figsize=(1.61 * 5, 1 * 5))
#     for key in meanLossTrain.keys():
#         plt.errorbar(xTrain, meanLossTrain[key], yerr=stdDevLossTrain[key],
#                      linewidth=2, marker='o', markersize=3)
#     plt.ylabel(r'Loss')
#     plt.xlabel(r'Training steps')
#     plt.legend(list(meanLossTrain.keys()))
#     allLossTrain.savefig(os.path.join(saveDirFigs, 'allLossTrain.pdf'),
#                          bbox_inches='tight')
#
#     # ACCURACY (validation) for ALL MODELS
#     allEvalValid = plt.figure(figsize=(1.61 * 5, 1 * 5))
#     for key in meanEvalValid.keys():
#         plt.errorbar(xValid, meanEvalValid[key], yerr=stdDevEvalValid[key],
#                      linewidth=2, marker='o', markersize=3)
#     plt.ylabel(r'Accuracy')
#     plt.xlabel(r'Training steps')
#     plt.legend(list(meanEvalValid.keys()))
#     allEvalValid.savefig(os.path.join(saveDirFigs, 'allEvalValid.pdf'),
#                          bbox_inches='tight')
