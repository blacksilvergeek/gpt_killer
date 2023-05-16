# \\\ Standard libraries:
import json
import os

import matplotlib
import numpy as np

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
import datetime

import torch;

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
randomStates = [{}]
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

authorName = 'poe'
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

nDataSplits = 1  # Number of data realizations
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
lossFunction = nn.BCELoss()  # This applies a softmax before feeding
# it into the NLL, so we don't have to apply the softmax ourselves.

# \\\ Overall training options
nEpochs = 15  # Number of epochs
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
doEdgeVariantGNN = True

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. That is, any new architecture in this part, needs also
# to be coded later on. This is just to be easy to change the parameters once
# the architecture is created. Do not forget to add the name of the architecture
# to modelList.

modelList = []

# Parameters for all models, so we don't need to be changing each one in each
# of the models (this guarantees comparable computational complexity)

nFeatures = 1  # F: number of output features of the only layer
nShifts = 2  # K: number of shift taps

##############
# PARAMETERS #
##############

hParamsEdgeVariant = {'name': 'EdgeVariGNN', 'F': [1, nFeatures], 'K': [nShifts], 'bias': True, 'sigma': nn.ReLU,
                      'rho': gml.NoPool, 'alpha': [1], 'dimLayersMLP': [nClasses]}

# \\\ Architecture parameters
# better set it to 1 to make everything slightly faster
# connected layers after the GCN layers

# \\\ Save Values:
writeVarValues(varsFile, hParamsEdgeVariant)
modelList += [hParamsEdgeVariant['name']]

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
#   Load the data, which will give a specific split
data = Utils.dataTools.Authorship(authorName, ratioTrain, ratioValid, dataPath)
training_results = {}

for author_name in data.authorData.keys():
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

        data.get_split(author_name, ratioTrain, ratioValid)

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

        # %%\\\\\\\\\\
        # \\\ MODEL 6: Edge-Variant GNN
        # \\\\\\\\\\\\

        if doEdgeVariantGNN:

            thisName = hParamsEdgeVariant['name']

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

            hParamsEdgeVariant['N'] = [nNodes]

            thisArchit = archit.EdgeVariantGNN(  # Graph filtering
                hParamsEdgeVariant['F'],
                hParamsEdgeVariant['K'],
                hParamsEdgeVariant['bias'],
                # Nonlinearity
                hParamsEdgeVariant['sigma'],
                # Pooling
                hParamsEdgeVariant['N'],
                hParamsEdgeVariant['rho'],
                hParamsEdgeVariant['alpha'],
                # MLP
                hParamsEdgeVariant['dimLayersMLP'],
                # Structure
                S)
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

            EdgeVariant = model.Model(thisArchit, thisLossFunction, thisOptim,
                                      thisName, saveDir, order)

            modelsGNN[thisName] = EdgeVariant

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


        def test_acc():
            modelsGNN[key].archit.set_phi(torch.tensor(G.S))
            modelsGNN[key].archit
            # recalculate the accuracy
            xTestOrdered = xTest[:, modelsGNN[key].order].unsqueeze(1)
            with torch.no_grad():
                # Process the samples
                yHatTest = modelsGNN[key].archit(xTestOrdered)
                # yHatTest is of shape
                #   testSize x numberOfClasses
                # We compute the accuracy
                acc = data.evaluate(yHatTest, yTest)

                return acc


        def go_left(abs, best_acc, mid, phi, sorted, threshold):
            # Zero all the values to the left
            values = sorted[:mid]
            values = values[values > 0]

            indices_to_zero = np.argwhere(np.isin(abs, values))

            for x, y in indices_to_zero:
                phi[x, y] = 0
                G.S[x, y] = 0
            # phi[abs in sorted[:mid]] = 0
            # test the accuracy

            acc = test_acc()

            return acc >= best_acc - best_acc * threshold, acc


        def find_words(phi, best_acc, threshold):
            sorted = np.sort(np.abs(phi.flatten()))
            abs = np.abs(phi)
            current = np.array(sorted)

            mid_index = current.shape[0]
            acc = best_acc
            phi_copy = np.copy(phi)
            S_copy = np.copy(G.S)

            epsilon = 0.005
            target_acc = best_acc - best_acc * threshold

            while not (
                    target_acc - epsilon < acc <= target_acc + epsilon) and \
                    mid_index > 1:
                mid_index = int(current.shape[0] / 2)
                mid = current[mid_index]

                good, acc = go_left(abs, best_acc, mid_index, phi_copy, current, threshold)
                if good:
                    phi = np.copy(phi_copy)
                    current = current[mid_index:]
                    S_copy = np.copy(G.S)
                else:
                    # restore phi
                    phi_copy = np.copy(phi)
                    G.S = np.copy(S_copy)

                    # take smaller values
                    current = current[:mid_index]

            return phi


        test = modelsGNN['EdgeVariGNN'].archit.EVGFL[0].Phi

        # save important pairs
        phi = test[0, 0, 1, 0, :, :]
        phi = phi.detach().numpy()

        new_phi = find_words(phi, np.max(accBest[thisModel]), 0.1)
        phi = new_phi

        function_words = np.array(data.functionWords)
        function_words = function_words[nodesToKeep]  # we get order from taining NN

        important_pairs = [(function_words[x[0]] + " - " + function_words[x[1]]) for x in
                           np.argwhere(np.abs(phi) > 0)]

        indices = [x for x in
                   np.argwhere(np.abs(phi) > 0)]

        corr = np.corrcoef(phi.flatten(), G.S.flatten())

        result = {'indices': indices, 'nodes': nodesToKeep, 'order': order, 'pairs': important_pairs,
                  'phi': phi.tolist(), 'correlation_coef': corr}
        training_results[author_name] = result

for k in training_results.keys():
    training_results[k]['indices'] = [[int(x[0]), int(x[1])] for x in training_results[k]['indices']]
    training_results[k]['nodes'] = [int(x) for x in training_results[k]['nodes']]
    training_results[k]['order'] = [int(x) for x in training_results[k]['order']]
    training_results[k]['nodes'] = [int(x) for x in training_results[k]['nodes']]
    training_results[k]['correlation_coef'] = float(training_results[k]['correlation_coef'][0, 1])

with open('EdgeVariGNN_important_words_phi_accuracy_0-1.txt', 'w+') as outfile:
    json.dump(training_results, outfile)
