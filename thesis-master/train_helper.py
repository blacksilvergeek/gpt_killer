import datetime
import os

import torch
from torch import optim, nn

from Modules import train, model
from Utils import graphTools
import numpy as np
import Utils.graphML as gml
import Modules.architectures_sigmoid as archit

torch.set_default_dtype(torch.float64)

doPrint = True
doSaveVars = False
doLogging = False
graphNormalizationType = 'rows'  # or 'cols' - Makes all rows add up to 1.
keepIsolatedNodes = False  # If True keeps isolated nodes
forceUndirected = True  # If True forces the graph to be undirected (symmetrizes)
forceConnected = True  # If True removes nodes (from lowest to highest degree)
# until the resulting graph is connected.

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
nEpochs = 20  # Number of epochs
batchSize = 16  # Batch size
doLearningRateDecay = False  # Learning rate decay
learningRateDecayRate = 0.9  # Rate
learningRateDecayPeriod = 1  # How many epochs after which update the lr
validationInterval = 5  # How many training steps to do the validation

thisFilename = 'train_helper'  # This is the general name of all related files

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

# \\\ Determine processing unit:
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
# Notify:
if doPrint:
    print("Device selected: %s" % device)

hParamsPolynomial = {'name': 'PolynomiGNN', 'F': [1, 2], 'K': [2], 'bias': True, 'sigma': nn.ReLU, 'rho': gml.NoPool,
                     'alpha': [1], 'dimLayersMLP': [1]}  # Hyperparameters (hParams)

trainingOptions = {}

if doSaveVars:
    trainingOptions['saveDir'] = saveDir

if doLearningRateDecay:
    trainingOptions['learningRateDecayRate'] = learningRateDecayRate
    trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
trainingOptions['validationInterval'] = validationInterval


def train_net(data, h_parameters, phi=None):
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
    if phi is None:
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
    else:
        nNodes = phi.shape[0]

    # Once data is completely formatted and in appropriate fashion, change its
    # type to torch and move it to the appropriate device
    data.astype(torch.float64)
    data.to(device)

    ##################################################################
    #                                                                   #
    #                    MODELS INITIALIZATION                          #
    #                                                                   #
    #####################################################################

    # Override parameters with grid parameters.
    hParamsPolynomial['F'] = h_parameters[0]
    hParamsPolynomial['K'] = h_parameters[1]

    # This is the dictionary where we store the models (in a model.Model
    # class, that is then passed to training).
    modelsGNN = {}

    # If a new model is to be created, it should be called for here.

    # \\\\\\\\\\
    # \\\ MODEL 2: Polynomial GNN
    # \\\\\\\\\\\\

    thisName = hParamsPolynomial['name']

    ##############
    # PARAMETERS #
    ##############

    # \\\ Optimizer options
    #   (If different from the default ones, change here.)
    thisTrainer = trainer
    thisLearningRate = learningRate
    thisBeta1 = beta1
    thisBeta2 = beta2

    if phi is None:
        # \\\ Ordering
        S, order = graphTools.permIdentity(G.S / np.max(np.diag(G.E)))
        # order is an np.array with the ordering of the nodes with respect
        # to the original GSO (the original GSO is kept in G.S).
    else:
        # compute the Eigenvalues of matrix
        e, V = np.linalg.eig(phi)
        # \\\ Ordering
        highest_eig_val = np.max(np.diag(e)).real

        if highest_eig_val == 0:
            S, order = graphTools.permIdentity(phi)
        else:
            S, order = graphTools.permIdentity(phi / highest_eig_val)
        # order is an np.array with the ordering of the nodes with respect
        # to the original GSO (the original GSO is kept in G.S).

    ################
    # ARCHITECTURE #
    ################

    hParamsPolynomial['N'] = [nNodes]

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

    # This is the function that trains the models detailed in the dictionary
    # modelsGNN using the data data, with the specified training options.
    train.MultipleModels(modelsGNN, data,
                         nEpochs=nEpochs, batchSize=batchSize,
                         **trainingOptions)

    return modelsGNN['PolynomiGNN']
