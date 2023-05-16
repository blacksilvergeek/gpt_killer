# 2018/12/4~2019/03/04.
# Fernando Gama, fgama@seas.upenn.edu
"""
dataTools.py Data management module

Several tools to manage data

SourceLocalization (class): creates the datasets for a source localization problem
"""

import numpy as np
import torch
import hdf5storage  # This is required to import old Matlab(R) files.

import Utils.graphTools as graph


class _dataForClassification:
    # Internal supraclass from which data classes inherit.
    # There are certian methods that all Data classes must have:
    #   getSamples(), evluate(), to() and astype().
    # To avoid coding this methods over and over again, we create a class from
    # which the data can inherit this basic methods.
    # Note that this is called "ForClassification" since the evaluate method
    # is only for classification evaluations.
    # However, it is true that getSamples() might be useful beyond the 
    # classification problem, so we might, eventually, consider a different
    # internal class.
    def __init__(self):
        # Minimal set of attributes that all data classes should have
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['labels'] = None
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['labels'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['labels'] = None

    def getSamples(self, samplesType, *args):
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
               or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['labels']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0]  # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size=args[0],
                                                   replace=False)
                # The reshape is to avoid squeezing if only one sample is
                # requested
                x = x[selectedIndices, :].reshape([args[0], x.shape[1]])
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                x = x[args[0], :]
                # If only one element is selected, avoid squeezing. Given that
                # the element can be a list (which has property len) or an
                # np.array (which doesn't have len, but shape), then we can
                # only avoid squeezing if we check that it has been sequeezed
                # (or not)
                if len(x.shape) == 1:
                    x = x.reshape([1, x.shape[0]])
                # And assign the labels
                y = y[args[0]]

        return x, y

    def astype(self, dataType):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        if repr(dataType).find('torch') == -1:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                        = dataType(self.samples[key][secondKey])
        else:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                        = torch.tensor(self.samples[key][secondKey]).type(dataType)

        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        # This can only be done if they are torch tensors
        if repr(self.dataType).find('torch') >= 0:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                        = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device

    def reduce_dim(self, indices):
        self.samples['train']['signals'] = self.samples['train']['signals'][:, indices]
        self.samples['valid']['signals'] = self.samples['valid']['signals'][:, indices]
        self.samples['test']['signals'] = self.samples['test']['signals'][:, indices]

    def evaluate_ce(self, yHat, y, tol=1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """
        N = len(y)
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
            yHat = torch.argmax(yHat, dim=1).type(self.dataType)
            #   And compute the error
            accuracy = torch.sum(y == yHat).type(self.dataType) / N
        else:
            yHat = np.array(yHat)
            y = np.array(y)
            #   We compute the target label (hardmax)
            yHat = np.argmax(yHat, axis=1).astype(y.dtype)
            #   And compute the error
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            accuracy = 1 - totalErrors.astype(self.dataType) / N
        #   And from that, compute the accuracy
        return accuracy

    def evaluate(self, yHat, y, tol=1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """
        N = len(y)
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
            # yHat = torch.argmax(yHat, dim=1).type(self.dataType)

            yHat = np.round(yHat)
            yHat = yHat.squeeze(1)

            #   And compute the error
            totalErrors = torch.sum(torch.abs(yHat - y) > tol)
            # accuracy = 1 - totalErrors.type(self.dataType) / N
            accuracy = 1 - totalErrors.item() / N
        else:
            yHat = np.array(yHat)
            y = np.array(y)
            #   We compute the target label (hardmax)
            yHat = np.argmax(yHat, axis=1).astype(y.dtype)
            #   And compute the error
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            accuracy = 1 - totalErrors.astype(self.dataType) / N
        #   And from that, compute the accuracy
        return accuracy


def changeDataType(x, dataType):
    """
    changeDataType(x, dataType): change the dataType of variable x into dataType
    """

    # So this is the thing: To change data type it depends on both, what dtype
    # the variable already is, and what dtype we want to make it.
    # Torch changes type by .type(), but numpy by .astype()
    # If we have already a torch defined, and we apply a torch.tensor() to it,
    # then there will be warnings because of gradient accounting.

    # All of these facts make changing types considerably cumbersome. So we
    # create a function that just changes type and handles all this issues
    # inside.

    # If we can't recognize the type, we just make everything numpy.

    # Check if the variable has an argument called 'dtype' so that we can now
    # what type of data type the variable is
    if 'dtype' in dir(x):
        varType = x.dtype

    # So, let's start assuming we want to convert to numpy
    if 'numpy' in repr(dataType):
        # Then, the variable con be torch, in which case we move it to cpu, to
        # numpy, and convert it to the right type.
        if 'torch' in repr(varType):
            x = x.cpu().numpy().astype(dataType)
        # Or it could be numpy, in which case we just use .astype
        elif 'numpy' in repr(type(x)):
            x = x.astype(dataType)
    # Now, we want to convert to torch
    elif 'torch' in repr(dataType):
        # If the variable is torch in itself
        if 'torch' in repr(varType):
            x = x.type(dataType)
        # But, if it's numpy
        elif 'numpy' in repr(type(x)):
            x = torch.tensor(x, dtype=dataType)

    # This only converts between numpy and torch. Any other thing is ignored
    return x


class SourceLocalization(_dataForClassification):
    """
    SourceLocalization: Creates the dataset for a source localization problem

    Initialization:

    Input:
        G (class): Graph on which to diffuse the process, needs an attribute
            .N with the number of nodes (int) and attribute .W with the
            adjacency matrix (np.array)
        nTrain (int): number of training samples
        nValid (int): number of validation samples
        nTest (int): number of testing samples
        sourceNodes (list of int): list of indices of nodes to be used as
            sources of the diffusion process
        tMax (int): maximum diffusion time, if None, the maximum diffusion time
            is the size of the graph (default: None)
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved.

    Methods:

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """

    def __init__(self, G, nTrain, nValid, nTest, sourceNodes, tMax=None,
                 dataType=np.float64, device='cpu'):
        # Initialize parent
        super().__init__()
        # store attributes
        self.dataType = dataType
        self.device = device
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        # If no tMax is specified, set it the maximum possible.
        if tMax == None:
            tMax = G.N
        # \\\ Generate the samples
        # Get the largest eigenvalue of the weighted adjacency matrix
        EW, VW = graph.computeGFT(G.W, order='totalVariation')
        eMax = np.max(EW)
        # Normalize the matrix so that it doesn't explode
        Wnorm = G.W / eMax
        # total number of samples
        nTotal = nTrain + nValid + nTest
        # sample source nodes
        sampledSources = np.random.choice(sourceNodes, size=nTotal)
        # sample diffusion times
        sampledTimes = np.random.choice(tMax, size=nTotal)
        # Since the signals are generated as W^t * delta, this reduces to the
        # selection of a column of W^t (the column corresponding to the source
        # node). Therefore, we generate an array of size tMax x N x N with all
        # the powers of the matrix, and then we just simply select the
        # corresponding column for the corresponding time
        lastWt = np.eye(G.N, G.N)
        Wt = lastWt.reshape([1, G.N, G.N])
        for t in range(1, tMax):
            lastWt = lastWt @ Wnorm
            Wt = np.concatenate((Wt, lastWt.reshape([1, G.N, G.N])), axis=0)
        x = Wt[sampledTimes, :, sampledSources]
        # Now, we have the signals and the labels
        signals = x  # nTotal x N (CS notation)
        # Finally, we have to match the source nodes to the corresponding labels
        # which start at 0 and increase in integers.
        nodesToLabels = {}
        for it in range(len(sourceNodes)):
            nodesToLabels[sourceNodes[it]] = it
        labels = [nodesToLabels[x] for x in sampledSources]  # nTotal
        # Split and save them
        self.samples['train']['signals'] = signals[0:nTrain, :]
        self.samples['train']['labels'] = labels[0:nTrain]
        self.samples['valid']['signals'] = signals[nTrain:nTrain + nValid, :]
        self.samples['valid']['labels'] = labels[nTrain:nTrain + nValid]
        self.samples['test']['signals'] = signals[nTrain + nValid:nTotal, :]
        self.samples['test']['labels'] = labels[nTrain + nValid:nTotal]
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)


class AuthorshipOneVsOne(_dataForClassification):
    """
    Authorship: Loads the dataset of 19th century writers for the authorship
        attribution problem

    Initialization:

    Input:
        authorName (string): which is the selected author to attribute plays to
        ratioTrain (float): ratio of the total texts to be part of the training
            set
        ratioValid (float): ratio of the train texts to be part of the
            validation set
        dataPath (string): path to where the authorship data is located
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:

    .loadData(dataPath): load the data found in dataPath and store it in
        attributes .authorData and .functionWords

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """

    def __init__(self, authorName, rest_name, ratioTrain, ratioValid, dataPath,
                 dataType=np.float64, device='cpu'):
        # Initialize parent
        super().__init__()
        # Store
        self.author_indices = {}
        self.dataType = dataType
        self.device = device

        # Load data
        self.loadData(dataPath)

        # Check that the authorName is a valid name
        assert authorName in self.authorData.keys()
        assert rest_name in self.authorData.keys()

        self.authorName = authorName
        self.restName = rest_name
        self.ratioTrain = ratioTrain
        self.ratioValid = ratioValid

        self.get_split(authorName, rest_name, ratioTrain, ratioValid)

    def get_split_same_author(self):
        self.get_split(self.authorName, self.restName, self.ratioTrain, self.ratioValid)

    def get_split(self, authorName, rest_name, ratioTrain, ratioValid):
        # Get the selected author's data
        thisAuthorData = self.authorData[authorName].copy()
        nExcerpts = np.min((thisAuthorData['wordFreq'].shape[0],
                            self.authorData[rest_name]['wordFreq'].shape[0])).astype(int).item()  # Number of excerpts
        # by the selected author
        nTrainAuthor = round(ratioTrain * nExcerpts)
        nValidAuthor = round(ratioValid * nExcerpts)
        nTestAuthor = nExcerpts - nTrainAuthor - nValidAuthor
        # nTrainAuthor = nTrainAuthor - nValidAuthor
        # Now, we know how many training, validation and testing samples from
        # the required author. But we will also include an equal amount of
        # other authors, therefore
        self.nTrain = round(2 * nTrainAuthor)
        self.nValid = round(2 * nValidAuthor)
        self.nTest = round(2 * nTestAuthor)
        # Now, let's get the corresponding signals for the author
        xAuthor = thisAuthorData['wordFreq']
        # Get a random permutation of these works, and split them accordingly
        randPerm = np.random.permutation(nExcerpts)
        # Save the indices corresponding to each split
        randPermTrain = randPerm[0:nTrainAuthor]
        randPermValid = randPerm[nTrainAuthor:nTrainAuthor + nValidAuthor]
        randPermTest = randPerm[nTrainAuthor + nValidAuthor:nExcerpts]
        xAuthorTrain = xAuthor[randPermTrain, :]
        xAuthorValid = xAuthor[randPermValid, :]
        xAuthorTest = xAuthor[randPermTest, :]
        # And we will store this split
        self.selectedAuthor = {}
        # Copy all data
        self.selectedAuthor['all'] = thisAuthorData.copy()
        # Copy word frequencies
        self.selectedAuthor['train'] = {}
        self.selectedAuthor['train']['wordFreq'] = xAuthorTrain.copy()
        self.selectedAuthor['valid'] = {}
        self.selectedAuthor['valid']['wordFreq'] = xAuthorValid.copy()
        self.selectedAuthor['test'] = {}
        self.selectedAuthor['test']['wordFreq'] = xAuthorTest.copy()
        # Copy WANs
        self.selectedAuthor['train']['WAN'] = \
            thisAuthorData['WAN'][randPermTrain, :, :].copy()
        self.selectedAuthor['valid']['WAN'] = \
            thisAuthorData['WAN'][randPermValid, :, :].copy()
        self.selectedAuthor['test']['WAN'] = \
            thisAuthorData['WAN'][randPermTest, :, :].copy()

        # # Now we need to get an equal amount of works from the rest of the
        # # authors.
        x_rest_tr, x_rest_val, x_rest_te = self.get_rest(rest_name, nTrainAuthor, nValidAuthor, nTestAuthor)

        nTrainRest = self.nTrain - nTrainAuthor
        nValidRest = self.nValid - nValidAuthor
        nTestRest = self.nTest - nTestAuthor
        # Now construct the signals and labels. Signals is just the
        # concatenation of each of these excerpts. Labels is just a bunch of
        # 1s followed by a bunch of 0s
        # Obs.: The fact that the dataset is ordered now, it doesn't matter,
        # since it will be shuffled at each epoch.
        xTrain = np.concatenate((xAuthorTrain, x_rest_tr), axis=0)
        labelsTrain = np.concatenate((np.ones(nTrainAuthor),
                                      np.zeros(nTrainRest)), axis=0)
        xValid = np.concatenate((xAuthorValid, x_rest_val), axis=0)
        labelsValid = np.concatenate((np.ones(nValidAuthor),
                                      np.zeros(nValidRest)), axis=0)
        xTest = np.concatenate((xAuthorTest, x_rest_te), axis=0)
        labelsTest = np.concatenate((np.ones(nTestAuthor),
                                     np.zeros(nTestRest)), axis=0)
        # And assign them to the required attribute samples
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = xTrain
        self.samples['train']['labels'] = labelsTrain
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = xValid
        self.samples['valid']['labels'] = labelsValid
        self.samples['test'] = {}
        self.samples['test']['signals'] = xTest
        self.samples['test']['labels'] = labelsTest
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    def get_rest(self, name, nTrainAuthor, nValidAuthor, nTestAuthor):
        cur = self.authorData[name]['wordFreq']

        no_of_excerpts = nTestAuthor + nTrainAuthor + nValidAuthor

        if cur.shape[0] <= no_of_excerpts:
            no_of_excerpts = cur.shape[0]

        randPerm = np.random.permutation(no_of_excerpts)
        randPermTrain = randPerm[0:nTrainAuthor]
        randPermValid = randPerm[nTrainAuthor:nTrainAuthor + nValidAuthor]
        randPermTest = randPerm[nTrainAuthor + nValidAuthor:]

        curTrain = cur[randPermTrain, :]
        curValid = cur[randPermValid, :]
        curTest = cur[randPermTest, :]

        return curTrain, curValid, curTest

    def loadData(self, dataPath):
        # TODO: Analyze if it's worth it to create a .pkl and load that
        # directly once the data has been appropriately parsed. It's just
        # that loading with hdf5storage takes a couple of second that
        # could be saved if the .pkl file is faster.
        rawData = hdf5storage.loadmat(dataPath)
        # rawData is a dictionary with four keys:
        #   'all_authors': contains the author list
        #   'all_freqs': contains the word frequency count for each excerpt
        #   'all_wans': contains the WANS for each excerpt
        #   'function_words': a list of the functional words
        # The issue is that hdf5storage, while necessary to load old
        # Matlab(R) files, gives the data in a weird format, that we need
        # to adapt and convert.
        # The data will be structured as follows. We will have an
        # authorData dictionary of dictionaries: the first key will be the
        # author name, the second key will be either freqs or wans to
        # access either one or another.
        # We will also clean up and save the functional word list, although
        # we do not need to use it.
        authorData = {}  # Create dictionary
        for it in range(len(rawData['all_authors'])):
            thisAuthor = str(rawData['all_authors'][it][0][0][0])
            # Each element in rawData['all_authors'] is nested in a couple
            # of lists, so that's why we need the three indices [0][0][0]
            # to reach the string with the actual author name.
            # Get the word frequency
            thisWordFreq = rawData['all_freqs'][0][it]  # 1 x nWords x nData
            # Again, the [0] is due to the structure of the data
            # Let us get rid of that extra 1, and then transpose this to be
            # stored as nData x nWords (since nWords is the dimension of
            # the number of nodes the network will have; CS notation)
            thisWordFreq = thisWordFreq.squeeze(0).T  # nData x nWords
            # Finally, get the WANs
            thisWAN = rawData['all_wans'][0][it]  # nWords x nWords x nData
            thisWAN = thisWAN.transpose(2, 0, 1)  # nData x nWords x nWords
            # Obs.: thisWAN is likely not symmetric, so the way this is
            # transposed matters. In this case, since thisWAN was intended
            # to be a tensor in matlab (where the last index is the
            # collection of matrices), we just throw that last dimension to
            # the front (since numpy consider the first index as the
            # collection index).
            # Now we can create the dictionary and save the corresopnding
            # data.
            authorData[thisAuthor] = {}
            authorData[thisAuthor]['wordFreq'] = thisWordFreq
            authorData[thisAuthor]['WAN'] = thisWAN
        # And at last, gather the list of functional words
        functionWords = []  # empty list to store the functional words
        for word in rawData['function_words']:
            functionWords.append(str(word[0][0][0]))
        # Store all the data recently collected
        self.authorData = authorData
        self.functionWords = functionWords


class Authorship(_dataForClassification):
    """
    Authorship: Loads the dataset of 19th century writers for the authorship
        attribution problem

    Initialization:

    Input:
        authorName (string): which is the selected author to attribute plays to
        ratioTrain (float): ratio of the total texts to be part of the training
            set
        ratioValid (float): ratio of the train texts to be part of the
            validation set
        dataPath (string): path to where the authorship data is located
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:
        
    .loadData(dataPath): load the data found in dataPath and store it in 
        attributes .authorData and .functionWords

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """

    def __init__(self, authorName, ratioTrain, ratioValid, dataPath,
                 dataType=np.float64, device='cpu'):
        # Initialize parent
        super().__init__()
        # Store
        self.author_indices = {}
        self.dataType = dataType
        self.device = device

        # Load data
        self.loadData(dataPath)

        self.init_author_indices()

        # Check that the authorName is a valid name
        assert authorName in self.authorData.keys()

        self.authorName = authorName
        self.ratioTrain = ratioTrain
        self.ratioValid = ratioValid

        self.get_split(authorName, ratioTrain, ratioValid)

    def get_split_same_author(self):
        self.get_split(self.authorName, self.ratioTrain, self.ratioValid)

    def get_split(self, authorName, ratioTrain, ratioValid):
        # Get the selected author's data
        thisAuthorData = self.authorData[authorName].copy()
        nExcerpts = thisAuthorData['wordFreq'].shape[0]  # Number of excerpts
        # by the selected author
        nTrainAuthor = round(ratioTrain * nExcerpts)
        nValidAuthor = round(ratioValid * nExcerpts)
        nTestAuthor = nExcerpts - nTrainAuthor - nValidAuthor
        # nTrainAuthor = nTrainAuthor - nValidAuthor
        # Now, we know how many training, validation and testing samples from
        # the required author. But we will also include an equal amount of
        # other authors, therefore
        self.nTrain = round(2 * nTrainAuthor)
        self.nValid = round(2 * nValidAuthor)
        self.nTest = round(2 * nTestAuthor)
        # Now, let's get the corresponding signals for the author
        xAuthor = thisAuthorData['wordFreq']
        # Get a random permutation of these works, and split them accordingly
        randPerm = np.random.permutation(nExcerpts)
        # Save the indices corresponding to each split
        randPermTrain = randPerm[0:nTrainAuthor]
        randPermValid = randPerm[nTrainAuthor:nTrainAuthor + nValidAuthor]
        randPermTest = randPerm[nTrainAuthor + nValidAuthor:nExcerpts]
        xAuthorTrain = xAuthor[randPermTrain, :]
        xAuthorValid = xAuthor[randPermValid, :]
        xAuthorTest = xAuthor[randPermTest, :]
        # And we will store this split
        self.selectedAuthor = {}
        # Copy all data
        self.selectedAuthor['all'] = thisAuthorData.copy()
        # Copy word frequencies
        self.selectedAuthor['train'] = {}
        self.selectedAuthor['train']['wordFreq'] = xAuthorTrain.copy()
        self.selectedAuthor['valid'] = {}
        self.selectedAuthor['valid']['wordFreq'] = xAuthorValid.copy()
        self.selectedAuthor['test'] = {}
        self.selectedAuthor['test']['wordFreq'] = xAuthorTest.copy()
        # Copy WANs
        self.selectedAuthor['train']['WAN'] = \
            thisAuthorData['WAN'][randPermTrain, :, :].copy()
        self.selectedAuthor['valid']['WAN'] = \
            thisAuthorData['WAN'][randPermValid, :, :].copy()
        self.selectedAuthor['test']['WAN'] = \
            thisAuthorData['WAN'][randPermTest, :, :].copy()
        # # Now we need to get an equal amount of works from the rest of the
        # # authors.
        # xRest = np.empty([0, xAuthorTrain.shape[1]])  # Create an empty matrix
        # # to store all the works by the rest of the authors.
        # # Now go author by author gathering all works
        # for key in self.authorData.keys():
        #     # Only for authors that are not the selected author
        #     if key is not authorName:
        #         thisAuthorTexts = self.authorData[key]['wordFreq']
        #         xRest = np.concatenate((xRest, thisAuthorTexts), axis=0)
        # # After obtaining all works, xRest is of shape nRestOfData x nWords
        # # We now need to select at random from this other data, but only up
        # # to nExcerpts. Therefore, we will randperm all the indices, but keep
        # # only the first nExcerpts indices.
        # randPerm = np.random.permutation(xRest.shape[0])
        # randPerm = randPerm[0:nExcerpts]  # nExcerpts x nWords
        # # And now we should just get the appropriate number of texts from these
        # # other authors.
        # # Compute how many samples for each case
        # nTrainRest = self.nTrain - nTrainAuthor
        # nValidRest = self.nValid - nValidAuthor
        # nTestRest = self.nTest - nTestAuthor
        # # And obtain those
        # xRestTrain = xRest[randPerm[0:nTrainRest], :]
        # xRestValid = xRest[randPerm[nTrainRest:nTrainRest + nValidRest], :]
        # xRestTest = xRest[randPerm[nTrainRest + nValidRest:nExcerpts], :]

        # Now, lets get uniformly distributed signals of the rest of the authors.
        x_rest_tr, x_rest_val, x_rest_te = self.get_rest_uniform(authorName, nTrainAuthor, nValidAuthor, nTestAuthor)
        nTrainRest = self.nTrain - nTrainAuthor
        nValidRest = self.nValid - nValidAuthor
        nTestRest = self.nTest - nTestAuthor
        # Now construct the signals and labels. Signals is just the
        # concatenation of each of these excerpts. Labels is just a bunch of
        # 1s followed by a bunch of 0s
        # Obs.: The fact that the dataset is ordered now, it doesn't matter,
        # since it will be shuffled at each epoch.
        xTrain = np.concatenate((xAuthorTrain, x_rest_tr), axis=0)
        labelsTrain = np.concatenate((np.ones(nTrainAuthor),
                                      np.zeros(nTrainRest)), axis=0)
        xValid = np.concatenate((xAuthorValid, x_rest_val), axis=0)
        labelsValid = np.concatenate((np.ones(nValidAuthor),
                                      np.zeros(nValidRest)), axis=0)
        xTest = np.concatenate((xAuthorTest, x_rest_te), axis=0)
        labelsTest = np.concatenate((np.ones(nTestAuthor),
                                     np.zeros(nTestRest)), axis=0)
        # And assign them to the required attribute samples
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = xTrain
        self.samples['train']['labels'] = labelsTrain
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = xValid
        self.samples['valid']['labels'] = labelsValid
        self.samples['test'] = {}
        self.samples['test']['signals'] = xTest
        self.samples['test']['labels'] = labelsTest
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    def init_author_indices(self):
        for key in self.authorData.keys():
            current_author_n_excerpts = self.authorData[key]['wordFreq'].shape[0]
            random_perm = np.random.permutation(current_author_n_excerpts)
            self.author_indices[key] = random_perm

    def get_next_indices(self, author_name, n):
        """
        Function that returns the next N indices for a given author in order to ensure, that all excerpts of a particular author are used.
        :param author_name:
        :param n:
        :return:
        """
        auth_ind = self.author_indices[author_name]

        if len(auth_ind) >= n:
            # if there are more unused indices than needed, just return next indices, and remove them from available
            ind = auth_ind[:n]
            self.author_indices[author_name] = auth_ind[n:]

            return ind

        # take the remaining indices, permute all available indices, and take how many is still needed.
        result = auth_ind

        author_n_excerpts = self.authorData[author_name]['wordFreq'].shape[0]
        self.author_indices[author_name] = np.random.permutation(author_n_excerpts)

        additional_ind = self.author_indices[author_name][:n - len(result)]
        result = np.concatenate((result, additional_ind))

        self.author_indices[author_name] = self.author_indices[author_name][len(additional_ind):]

        return result

    def get_rest_uniform(self, author_name, nTrainAuthor, nValidAuthor, nTestAuthor):
        """
        Function, that returns excerpts from the rest of the authors uniformly distributed.
        :param author_name:
        :param nTrainAuthor:
        :param nValidAuthor:
        :param nTestAuthor:
        :return:
        """

        # Compute how many samples for each case
        no_of_authors_left = (len(self.authorData.keys()) - 1)

        nTrainRest = int((self.nTrain - nTrainAuthor) / no_of_authors_left)
        nValidRest = int((self.nValid - nValidAuthor) / no_of_authors_left)
        nTestRest = int((self.nTest - nTestAuthor) / no_of_authors_left)

        # Now we need to get an equal amount of works from the rest of the
        # authors.
        no_of_function_words = self.selectedAuthor['train']['wordFreq'].shape[1]

        x_rest_tr = np.empty([0, no_of_function_words])  # Create an empty matrix
        x_rest_te = np.empty([0, no_of_function_words])  # Create an empty matrix
        x_rest_val = np.empty([0, no_of_function_words])  # Create an empty matrix

        for key in self.authorData.keys():
            # Only for authors that are not the selected author
            if key != author_name:
                thisAuthorTexts = self.authorData[key]['wordFreq']

                # get the indices for a particular author, and add texts to training, test and validation sets.
                author_indices_tr = self.get_next_indices(key, nTrainRest)
                author_indices_te = self.get_next_indices(key, nTestRest)
                author_indices_val = self.get_next_indices(key, nValidRest)

                x_rest_tr = np.concatenate((x_rest_tr, thisAuthorTexts[author_indices_tr, :]), axis=0)
                x_rest_te = np.concatenate((x_rest_te, thisAuthorTexts[author_indices_te, :]), axis=0)
                x_rest_val = np.concatenate((x_rest_val, thisAuthorTexts[author_indices_val, :]), axis=0)

        x_rest_te, x_rest_tr, x_rest_val = self.add_rest_of_samples(author_name,
                                                                    self.nTest - nTestAuthor,
                                                                    self.nTrain - nTrainAuthor,
                                                                    self.nValid - nValidAuthor,
                                                                    x_rest_te, x_rest_tr,
                                                                    x_rest_val)

        return x_rest_tr, x_rest_val, x_rest_te

    def add_rest_of_samples(self, author_name, nTestAuthor, nTrainAuthor, nValidAuthor, x_rest_te, x_rest_tr,
                            x_rest_val):
        for key in self.authorData.keys():
            if x_rest_tr.shape[0] == nTrainAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_tr = self.get_next_indices(key, 1)
                x_rest_tr = np.concatenate((x_rest_tr, self.authorData[key]['wordFreq'][author_indices_tr, :]), axis=0)

        for key in self.authorData.keys():
            if x_rest_te.shape[0] == nTestAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_te = self.get_next_indices(key, 1)
                x_rest_te = np.concatenate((x_rest_te, self.authorData[key]['wordFreq'][author_indices_te, :]), axis=0)

        for key in self.authorData.keys():
            if x_rest_val.shape[0] == nValidAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_val = self.get_next_indices(key, 1)
                x_rest_val = np.concatenate((x_rest_val, self.authorData[key]['wordFreq'][author_indices_val, :]),
                                            axis=0)

        return x_rest_te, x_rest_tr, x_rest_val

    def loadData(self, dataPath):
        # TODO: Analyze if it's worth it to create a .pkl and load that
        # directly once the data has been appropriately parsed. It's just
        # that loading with hdf5storage takes a couple of second that
        # could be saved if the .pkl file is faster.
        rawData = hdf5storage.loadmat(dataPath)
        # rawData is a dictionary with four keys:
        #   'all_authors': contains the author list
        #   'all_freqs': contains the word frequency count for each excerpt
        #   'all_wans': contains the WANS for each excerpt
        #   'function_words': a list of the functional words
        # The issue is that hdf5storage, while necessary to load old 
        # Matlab(R) files, gives the data in a weird format, that we need
        # to adapt and convert.
        # The data will be structured as follows. We will have an
        # authorData dictionary of dictionaries: the first key will be the
        # author name, the second key will be either freqs or wans to
        # access either one or another.
        # We will also clean up and save the functional word list, although
        # we do not need to use it.
        authorData = {}  # Create dictionary
        for it in range(len(rawData['all_authors'])):
            thisAuthor = str(rawData['all_authors'][it][0][0][0])
            # Each element in rawData['all_authors'] is nested in a couple
            # of lists, so that's why we need the three indices [0][0][0] 
            # to reach the string with the actual author name.
            # Get the word frequency
            thisWordFreq = rawData['all_freqs'][0][it]  # 1 x nWords x nData
            # Again, the [0] is due to the structure of the data
            # Let us get rid of that extra 1, and then transpose this to be
            # stored as nData x nWords (since nWords is the dimension of 
            # the number of nodes the network will have; CS notation)
            thisWordFreq = thisWordFreq.squeeze(0).T  # nData x nWords
            # Finally, get the WANs
            thisWAN = rawData['all_wans'][0][it]  # nWords x nWords x nData
            thisWAN = thisWAN.transpose(2, 0, 1)  # nData x nWords x nWords
            # Obs.: thisWAN is likely not symmetric, so the way this is 
            # transposed matters. In this case, since thisWAN was intended
            # to be a tensor in matlab (where the last index is the 
            # collection of matrices), we just throw that last dimension to
            # the front (since numpy consider the first index as the 
            # collection index).
            # Now we can create the dictionary and save the corresopnding
            # data.
            authorData[thisAuthor] = {}
            authorData[thisAuthor]['wordFreq'] = thisWordFreq
            authorData[thisAuthor]['WAN'] = thisWAN
        # And at last, gather the list of functional words
        functionWords = []  # empty list to store the functional words
        for word in rawData['function_words']:
            functionWords.append(str(word[0][0][0]))
        # Store all the data recently collected
        self.authorData = authorData
        self.functionWords = functionWords


class AutorshipGenderNationality(_dataForClassification):
    """
    Authorship: Loads the dataset of 19th century writers for the authorship
        attribution problem

    Initialization:

    Input:
        authorName (string): which is the selected author to attribute plays to
        ratioTrain (float): ratio of the total texts to be part of the training
            set
        ratioValid (float): ratio of the train texts to be part of the
            validation set
        dataPath (string): path to where the authorship data is located
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:

    .loadData(dataPath): load the data found in dataPath and store it in
        attributes .authorData and .functionWords

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """

    all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                        'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving',
                        'poe',
                        'jewett', 'wharton']

    # Possible authors: (just use the names in ' ')
    # jacob 'abbott', robert louis 'stevenson', louisa may 'alcott',
    # horatio 'alger', james 'allen', jane 'austen', emily 'bronte', james 'cooper',
    # charles 'dickens', hamlin 'garland', nathaniel 'hawthorne', henry 'james',
    # herman 'melville', 'page', herny 'thoreau', mark 'twain',
    # arthur conan 'doyle', washington 'irving', edgar allan 'poe',
    # sarah orne 'jewett', edith 'wharton'

    # british = ['stevenson', 'austen', 'bronte', 'allen', 'jewett', 'wharton', 'dickens', 'james', 'doyle']
    #
    # american = ['alcott', 'abbott', 'alger', 'cooper', 'garland', 'hawthorne', 'melville',
    #             'thoreau', 'twain', 'irving', 'poe']

    women_american = ['alcott']
    women_british = ['austen', 'bronte', 'jewett', 'wharton']

    men_british = ['stevenson', 'allen', 'dickens', 'james', 'doyle']
    men_american = ['abbott', 'alger', 'cooper', 'garland', 'hawthorne', 'melville',
                    'thoreau', 'twain', 'irving', 'poe']

    def __init__(self, ratioTrain, ratioValid, dataPath,
                 dataType=np.float64, device='cpu'):
        # Initialize parent
        super().__init__()
        # Store
        self.author_indices = {}
        self.dataType = dataType
        self.device = device

        # Load data
        self.loadData(dataPath)

        self.init_author_indices()

        self.ratioTrain = ratioTrain
        self.ratioValid = ratioValid

        self.get_split(ratioTrain, ratioValid)

    def get_split(self, ratioTrain, ratioValid):
        # start with Women as there are less excerpts and concat everything.
        women_american_WAN, women_american_frq, women_american_frq_train, women_american_frq_valid, women_american_frq_test = self.get_group(
            self.women_american)
        women_british_WAN, women_british_frq, women_british_frq_train, women_british_frq_valid, women_british_frq_test = self.get_group(
            self.women_british)

        # Get excerpts for Men
        men_american_WAN, men_american_frq, men_american_frq_train, men_american_frq_valid, men_american_frq_test = self.get_group(
            self.men_american)
        men_british_WAN, men_british_frq, men_british_frq_train, men_british_frq_valid, men_british_frq_test = self.get_group(
            self.men_british)

        all_authors = np.concatenate((women_american_frq, women_british_frq, men_american_frq, men_british_frq), axis=0)
        all_authors_WAN = np.concatenate((women_american_WAN, women_british_WAN, men_american_WAN, men_british_WAN),
                                         axis=0)

        # Get the selected author's data
        n_selected_author_excerpts = int(self.get_total_no_of_excerpts() / 2)  # Number of excerpts
        # by the selected author
        nTrainAuthor = round(ratioTrain * n_selected_author_excerpts)
        nValidAuthor = round(ratioValid * n_selected_author_excerpts)

        # Now, we know how many training, validation and testing samples from
        # the required author. But we will also include an equal amount of
        # other authors, therefore

        self.nTrain = women_american_frq_train.shape[0] + women_british_frq_train.shape[0] + \
                      men_american_frq_train.shape[
                          0] + men_british_frq_train.shape[0]

        self.nValid = women_american_frq_valid.shape[0] + women_british_frq_valid.shape[0] + \
                      men_american_frq_valid.shape[
                          0] + men_british_frq_valid.shape[0]

        self.test = women_american_frq_test.shape[0] + women_british_frq_test.shape[0] + men_american_frq_test.shape[
            0] + men_british_frq_test.shape[0]

        # # all other authors times ration, plus selected author times ratio.
        # self.nTrain = round(n_excerpts_rest * ratioTrain + nTrainAuthor)
        # self.nValid = round(n_excerpts_rest * ratioValid + nValidAuthor)
        # self.nTest = self.get_total_no_of_excerpts() - self.nValid - self.nTrain
        # # Now, let's get the corresponding signals for the author
        # xAuthor = thisAuthorData['wordFreq']
        # Get a random permutation of these works, and split them accordingly
        randPerm = np.random.permutation(n_selected_author_excerpts)

        # Save the indices corresponding to each split
        randPermTrain = randPerm[0:nTrainAuthor]
        randPermValid = randPerm[nTrainAuthor:nTrainAuthor + nValidAuthor]
        randPermTest = randPerm[nTrainAuthor + nValidAuthor:]

        # xAuthor = np.concatenate((men_selected_sig, women_all_frq), axis=0)
        # xAuthor_WAN = np.concatenate((men_selected_WAN, women_all_WAN), axis=0)

        xAuthorTrain = all_authors[randPermTrain, :]
        xAuthorValid = all_authors[randPermValid, :]
        xAuthorTest = all_authors[randPermTest, :]

        # And we will store this split
        self.selectedAuthor = {'all': {"wordFreq": all_authors[randPerm, :], "WAN": all_authors_WAN[randPerm, :, :]},
                               'train': {}}
        # Copy all data
        # Copy word frequencies
        self.selectedAuthor['train']['wordFreq'] = xAuthorTrain.copy()
        self.selectedAuthor['valid'] = {}
        self.selectedAuthor['valid']['wordFreq'] = xAuthorValid.copy()
        self.selectedAuthor['test'] = {}
        self.selectedAuthor['test']['wordFreq'] = xAuthorTest.copy()
        # Copy WANs
        self.selectedAuthor['train']['WAN'] = \
            all_authors_WAN[randPermTrain, :, :].copy()
        self.selectedAuthor['valid']['WAN'] = \
            all_authors_WAN[randPermValid, :, :].copy()
        self.selectedAuthor['test']['WAN'] = \
            all_authors_WAN[randPermTest, :, :].copy()

        # # And now we should just get the appropriate number of texts from these
        # # other authors.
        # # Compute how many samples for each case
        # nTrainRest = nTrainAuthor
        # nValidRest = nValidAuthor
        # nTestRest = nTestAuthor
        # # And obtain those
        # xRestTrain = men_selected_sig[randPerm[0:nTrainRest], :]
        # xRestValid = men_selected_sig[randPerm[nTrainRest:nTrainRest + nValidRest], :]
        # xRestTest = men_selected_sig[randPerm[nTrainRest + nValidRest:], :]
        # Now construct the signals and labels. Signals is just the
        # concatenation of each of these excerpts. Labels is just a bunch of
        # 1s followed by a bunch of 0s
        # Obs.: The fact that the dataset is ordered now, it doesn't matter,
        # since it will be shuffled at each epoch.
        xTrain = np.concatenate(
            (women_american_frq_train, women_british_frq_train, men_american_frq_train, men_british_frq_train), axis=0)
        labelsTrain = np.concatenate((np.full(women_american_frq_train.shape[0], 0.0),
                                      np.full(women_british_frq_train.shape[0], 1.0),
                                      np.full(men_american_frq_train.shape[0], 2.0),
                                      np.full(men_british_frq_train.shape[0], 3.0)
                                      ), axis=0)

        xValid = np.concatenate(
            (women_american_frq_valid, women_british_frq_valid, men_american_frq_valid, men_british_frq_valid), axis=0)
        labelsValid = np.concatenate((np.full(women_american_frq_valid.shape[0], 0.0),
                                      np.full(women_british_frq_valid.shape[0], 1.0),
                                      np.full(men_american_frq_valid.shape[0], 2.0),
                                      np.full(men_british_frq_valid.shape[0], 3.0)
                                      ), axis=0)

        xTest = np.concatenate(
            (women_american_frq_test, women_british_frq_test, men_american_frq_test, men_british_frq_test), axis=0)
        labelsTest = np.concatenate((np.full(women_american_frq_test.shape[0], 0.0),
                                     np.full(women_british_frq_test.shape[0], 1.0),
                                     np.full(men_american_frq_test.shape[0], 2.0),
                                     np.full(men_british_frq_test.shape[0], 3.0)
                                     ), axis=0)

        # And assign them to the required attribute samples
        self.samples = {'train': {}}
        self.samples['train']['signals'] = xTrain
        self.samples['train']['labels'] = labelsTrain
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = xValid
        self.samples['valid']['labels'] = labelsValid
        self.samples['test'] = {}
        self.samples['test']['signals'] = xTest
        self.samples['test']['labels'] = labelsTest
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    def get_group(self, names):
        frq = np.empty([0, self.authorData['abbott']['WAN'].shape[1]])
        WAN = np.empty(
            [0, self.authorData['abbott']['WAN'].shape[1], self.authorData['abbott']['WAN'].shape[1]])

        for name in names:
            frq = np.concatenate((frq, self.authorData[name]['wordFreq']), axis=0)
            WAN = np.concatenate((WAN, self.authorData[name]['WAN']), axis=0)

        n_total = frq.shape[0]
        n_train = round(self.ratioTrain * n_total)
        n_valid = round(self.ratioValid * n_total)
        n_test = n_total - n_train - n_valid

        perm = np.random.permutation(n_total)

        frq_train = frq[perm[:n_train], :]
        frq_valid = frq[perm[n_train:n_train + n_valid], :]
        frq_test = frq[perm[n_train + n_valid:], :]

        return WAN, frq, frq_train, frq_valid, frq_test

    def get_total_no_of_excerpts(self):
        women_british = sum([i['wordFreq'].shape[0] for k, i in self.authorData.items() if k in self.women_british])
        women_american = sum([i['wordFreq'].shape[0] for k, i in self.authorData.items() if k in self.women_american])
        men_british = sum([i['wordFreq'].shape[0] for k, i in self.authorData.items() if k in self.men_british])
        men_american = sum([i['wordFreq'].shape[0] for k, i in self.authorData.items() if k in self.men_american])

        return women_british + women_american + men_american + men_british

    def get_split_same_author(self):
        self.get_split(self.ratioTrain, self.ratioValid)

    def init_author_indices(self):
        for key in self.authorData.keys():
            current_author_n_excerpts = self.authorData[key]['wordFreq'].shape[0]
            random_perm = np.random.permutation(current_author_n_excerpts)
            self.author_indices[key] = random_perm

    def get_next_indices(self, author_name, n):
        """
        Function that returns the next N indices for a given author in order to ensure, that all excerpts of a particular author are used.
        :param author_name:
        :param n:
        :return:
        """
        auth_ind = self.author_indices[author_name]

        if len(auth_ind) >= n:
            # if there are more unused indices than needed, just return next indices, and remove them from available
            ind = auth_ind[:n]
            self.author_indices[author_name] = auth_ind[n:]

            return ind

        # take the remaining indices, permute all available indices, and take how many is still needed.
        result = auth_ind

        author_n_excerpts = self.authorData[author_name]['wordFreq'].shape[0]
        self.author_indices[author_name] = np.random.permutation(author_n_excerpts)

        additional_ind = self.author_indices[author_name][:n - len(result)]
        result = np.concatenate((result, additional_ind))

        self.author_indices[author_name] = self.author_indices[author_name][len(additional_ind):]

        return result

    def get_rest_uniform(self, author_name, nTrainAuthor, nValidAuthor, nTestAuthor):
        """
        Function, that returns excerpts from the rest of the authors uniformly distributed.
        :param author_name:
        :param nTrainAuthor:
        :param nValidAuthor:
        :param nTestAuthor:
        :return:
        """

        # Compute how many samples for each case
        no_of_authors_left = (len(self.authorData.keys()) - 1)

        nTrainRest = int((self.nTrain - nTrainAuthor) / no_of_authors_left)
        nValidRest = int((self.nValid - nValidAuthor) / no_of_authors_left)
        nTestRest = int((self.nTest - nTestAuthor) / no_of_authors_left)

        # Now we need to get an equal amount of works from the rest of the
        # authors.
        no_of_function_words = self.selectedAuthor['train']['wordFreq'].shape[1]

        x_rest_tr = np.empty([0, no_of_function_words])  # Create an empty matrix
        x_rest_te = np.empty([0, no_of_function_words])  # Create an empty matrix
        x_rest_val = np.empty([0, no_of_function_words])  # Create an empty matrix

        for key in self.authorData.keys():
            # Only for authors that are not the selected author
            if key != author_name:
                thisAuthorTexts = self.authorData[key]['wordFreq']

                # get the indices for a particular author, and add texts to training, test and validation sets.
                author_indices_tr = self.get_next_indices(key, nTrainRest)
                author_indices_te = self.get_next_indices(key, nTestRest)
                author_indices_val = self.get_next_indices(key, nValidRest)

                x_rest_tr = np.concatenate((x_rest_tr, thisAuthorTexts[author_indices_tr, :]), axis=0)
                x_rest_te = np.concatenate((x_rest_te, thisAuthorTexts[author_indices_te, :]), axis=0)
                x_rest_val = np.concatenate((x_rest_val, thisAuthorTexts[author_indices_val, :]), axis=0)

        x_rest_te, x_rest_tr, x_rest_val = self.add_rest_of_samples(author_name,
                                                                    self.nTest - nTestAuthor,
                                                                    self.nTrain - nTrainAuthor,
                                                                    self.nValid - nValidAuthor,
                                                                    x_rest_te, x_rest_tr,
                                                                    x_rest_val)

        return x_rest_tr, x_rest_val, x_rest_te

    def add_rest_of_samples(self, author_name, nTestAuthor, nTrainAuthor, nValidAuthor, x_rest_te, x_rest_tr,
                            x_rest_val):
        for key in self.authorData.keys():
            if x_rest_tr.shape[0] == nTrainAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_tr = self.get_next_indices(key, 1)
                x_rest_tr = np.concatenate((x_rest_tr, self.authorData[key]['wordFreq'][author_indices_tr, :]), axis=0)

        for key in self.authorData.keys():
            if x_rest_te.shape[0] == nTestAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_te = self.get_next_indices(key, 1)
                x_rest_te = np.concatenate((x_rest_te, self.authorData[key]['wordFreq'][author_indices_te, :]), axis=0)

        for key in self.authorData.keys():
            if x_rest_val.shape[0] == nValidAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_val = self.get_next_indices(key, 1)
                x_rest_val = np.concatenate((x_rest_val, self.authorData[key]['wordFreq'][author_indices_val, :]),
                                            axis=0)

        return x_rest_te, x_rest_tr, x_rest_val

    def loadData(self, dataPath):
        # TODO: Analyze if it's worth it to create a .pkl and load that
        # directly once the data has been appropriately parsed. It's just
        # that loading with hdf5storage takes a couple of second that
        # could be saved if the .pkl file is faster.
        rawData = hdf5storage.loadmat(dataPath)
        # rawData is a dictionary with four keys:
        #   'all_authors': contains the author list
        #   'all_freqs': contains the word frequency count for each excerpt
        #   'all_wans': contains the WANS for each excerpt
        #   'function_words': a list of the functional words
        # The issue is that hdf5storage, while necessary to load old
        # Matlab(R) files, gives the data in a weird format, that we need
        # to adapt and convert.
        # The data will be structured as follows. We will have an
        # authorData dictionary of dictionaries: the first key will be the
        # author name, the second key will be either freqs or wans to
        # access either one or another.
        # We will also clean up and save the functional word list, although
        # we do not need to use it.
        authorData = {}  # Create dictionary
        for it in range(len(rawData['all_authors'])):
            thisAuthor = str(rawData['all_authors'][it][0][0][0])
            # Each element in rawData['all_authors'] is nested in a couple
            # of lists, so that's why we need the three indices [0][0][0]
            # to reach the string with the actual author name.
            # Get the word frequency
            thisWordFreq = rawData['all_freqs'][0][it]  # 1 x nWords x nData
            # Again, the [0] is due to the structure of the data
            # Let us get rid of that extra 1, and then transpose this to be
            # stored as nData x nWords (since nWords is the dimension of
            # the number of nodes the network will have; CS notation)
            thisWordFreq = thisWordFreq.squeeze(0).T  # nData x nWords
            # Finally, get the WANs
            thisWAN = rawData['all_wans'][0][it]  # nWords x nWords x nData
            thisWAN = thisWAN.transpose(2, 0, 1)  # nData x nWords x nWords
            # Obs.: thisWAN is likely not symmetric, so the way this is
            # transposed matters. In this case, since thisWAN was intended
            # to be a tensor in matlab (where the last index is the
            # collection of matrices), we just throw that last dimension to
            # the front (since numpy consider the first index as the
            # collection index).
            # Now we can create the dictionary and save the corresopnding
            # data.
            authorData[thisAuthor] = {}
            authorData[thisAuthor]['wordFreq'] = thisWordFreq
            authorData[thisAuthor]['WAN'] = thisWAN
        # And at last, gather the list of functional words
        functionWords = []  # empty list to store the functional words
        for word in rawData['function_words']:
            functionWords.append(str(word[0][0][0]))
        # Store all the data recently collected
        self.authorData = authorData
        self.functionWords = functionWords


class AutorshipGender(_dataForClassification):
    """
    Authorship: Loads the dataset of 19th century writers for the authorship
        attribution problem

    Initialization:

    Input:
        authorName (string): which is the selected author to attribute plays to
        ratioTrain (float): ratio of the total texts to be part of the training
            set
        ratioValid (float): ratio of the train texts to be part of the
            validation set
        dataPath (string): path to where the authorship data is located
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:

    .loadData(dataPath): load the data found in dataPath and store it in
        attributes .authorData and .functionWords

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """

    all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                        'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving',
                        'poe',
                        'jewett', 'wharton']

    # Possible authors: (just use the names in ' ')
    # jacob 'abbott', robert louis 'stevenson', louisa may 'alcott',
    # horatio 'alger', james 'allen', jane 'austen', emily 'bronte', james 'cooper',
    # charles 'dickens', hamlin 'garland', nathaniel 'hawthorne', henry 'james',
    # herman 'melville', 'page', herny 'thoreau', mark 'twain',
    # arthur conan 'doyle', washington 'irving', edgar allan 'poe',
    # sarah orne 'jewett', edith 'wharton'

    women = ['alcott', 'austen', 'bronte', 'jewett', 'wharton']
    men = ['abbott', 'stevenson', 'alger', 'allen', 'cooper', 'dickens', 'garland', 'hawthorne', 'james', 'melville',
           'thoreau', 'twain', 'doyle', 'irving', 'poe']

    def __init__(self, ratioTrain, ratioValid, dataPath,
                 dataType=np.float64, device='cpu'):
        # Initialize parent
        super().__init__()
        # Store
        self.author_indices = {}
        self.dataType = dataType
        self.device = device

        # Load data
        self.loadData(dataPath)

        self.init_author_indices()

        self.ratioTrain = ratioTrain
        self.ratioValid = ratioValid

        self.get_split(ratioTrain, ratioValid)

    def get_split(self, ratioTrain, ratioValid):
        # start with Women as there are less excerpts and concat everything.

        women_all_frq = np.empty([0, self.authorData['abbott']['WAN'].shape[1]])
        women_all_WAN = np.empty(
            [0, self.authorData['abbott']['WAN'].shape[1], self.authorData['abbott']['WAN'].shape[1]])

        for name in self.women:
            women_all_frq = np.concatenate((women_all_frq, self.authorData[name]['wordFreq']), axis=0)
            women_all_WAN = np.concatenate((women_all_WAN, self.authorData[name]['WAN']), axis=0)

        # Get the selected author's data
        n_selected_author_excerpts = women_all_frq.shape[0]  # Number of excerpts
        # by the selected author
        nTrainAuthor = round(ratioTrain * n_selected_author_excerpts)
        nValidAuthor = round(ratioValid * n_selected_author_excerpts)
        nTestAuthor = n_selected_author_excerpts - nTrainAuthor - nValidAuthor
        # Now, we know how many training, validation and testing samples from
        # the required author. But we will also include an equal amount of
        # other authors, therefore
        n_excerpts_rest = self.get_total_no_of_excerpts() - n_selected_author_excerpts

        self.nTrain = nTrainAuthor * 2
        self.nValid = nValidAuthor * 2
        self.nTest = nTestAuthor * 2

        men_all_frq = np.empty([0, self.authorData['abbott']['WAN'].shape[1]])
        men_all_WAN = np.empty(
            [0, self.authorData['abbott']['WAN'].shape[1], self.authorData['abbott']['WAN'].shape[1]])

        for name in self.men:
            men_all_frq = np.concatenate((men_all_frq, self.authorData[name]['wordFreq']), axis=0)
            men_all_WAN = np.concatenate((men_all_WAN, self.authorData[name]['WAN']), axis=0)

        # select men excerpts randomly, as there are too much
        rand_perm_men = np.random.permutation(n_excerpts_rest)

        men_selected_sig = men_all_frq[rand_perm_men, :]
        men_selected_WAN = men_all_WAN[rand_perm_men, :, :]

        # # all other authors times ration, plus selected author times ratio.
        # self.nTrain = round(n_excerpts_rest * ratioTrain + nTrainAuthor)
        # self.nValid = round(n_excerpts_rest * ratioValid + nValidAuthor)
        # self.nTest = self.get_total_no_of_excerpts() - self.nValid - self.nTrain
        # # Now, let's get the corresponding signals for the author
        # xAuthor = thisAuthorData['wordFreq']
        # Get a random permutation of these works, and split them accordingly
        randPerm = np.random.permutation(n_selected_author_excerpts)

        # Save the indices corresponding to each split
        randPermTrain = randPerm[0:nTrainAuthor]
        randPermValid = randPerm[nTrainAuthor:nTrainAuthor + nValidAuthor]
        randPermTest = randPerm[nTrainAuthor + nValidAuthor:]

        # xAuthor = np.concatenate((men_selected_sig, women_all_frq), axis=0)
        # xAuthor_WAN = np.concatenate((men_selected_WAN, women_all_WAN), axis=0)

        xAuthorTrain = women_all_frq[randPermTrain, :]
        xAuthorValid = women_all_frq[randPermValid, :]
        xAuthorTest = women_all_frq[randPermTest, :]

        # And we will store this split
        self.selectedAuthor = {'all': {"wordFreq": women_all_frq, "WAN": women_all_WAN}, 'train': {}}
        # Copy all data
        # Copy word frequencies
        self.selectedAuthor['train']['wordFreq'] = xAuthorTrain.copy()
        self.selectedAuthor['valid'] = {}
        self.selectedAuthor['valid']['wordFreq'] = xAuthorValid.copy()
        self.selectedAuthor['test'] = {}
        self.selectedAuthor['test']['wordFreq'] = xAuthorTest.copy()
        # Copy WANs
        self.selectedAuthor['train']['WAN'] = \
            women_all_WAN[randPermTrain, :, :].copy()
        self.selectedAuthor['valid']['WAN'] = \
            women_all_WAN[randPermValid, :, :].copy()
        self.selectedAuthor['test']['WAN'] = \
            women_all_WAN[randPermTest, :, :].copy()

        # And now we should just get the appropriate number of texts from these
        # other authors.
        # Compute how many samples for each case
        nTrainRest = nTrainAuthor
        nValidRest = nValidAuthor
        nTestRest = nTestAuthor
        # And obtain those
        xRestTrain = men_selected_sig[randPerm[0:nTrainRest], :]
        xRestValid = men_selected_sig[randPerm[nTrainRest:nTrainRest + nValidRest], :]
        xRestTest = men_selected_sig[randPerm[nTrainRest + nValidRest:], :]
        # Now construct the signals and labels. Signals is just the
        # concatenation of each of these excerpts. Labels is just a bunch of
        # 1s followed by a bunch of 0s
        # Obs.: The fact that the dataset is ordered now, it doesn't matter,
        # since it will be shuffled at each epoch.
        xTrain = np.concatenate((xAuthorTrain, xRestTrain), axis=0)
        labelsTrain = np.concatenate((np.ones(nTrainAuthor),
                                      np.zeros(nTrainRest)), axis=0)
        xValid = np.concatenate((xAuthorValid, xRestValid), axis=0)
        labelsValid = np.concatenate((np.ones(nValidAuthor),
                                      np.zeros(nValidRest)), axis=0)
        xTest = np.concatenate((xAuthorTest, xRestTest), axis=0)
        labelsTest = np.concatenate((np.ones(nTestAuthor),
                                     np.zeros(nTestRest)), axis=0)
        # And assign them to the required attribute samples
        self.samples = {'train': {}}
        self.samples['train']['signals'] = xTrain
        self.samples['train']['labels'] = labelsTrain
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = xValid
        self.samples['valid']['labels'] = labelsValid
        self.samples['test'] = {}
        self.samples['test']['signals'] = xTest
        self.samples['test']['labels'] = labelsTest
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    def get_total_no_of_excerpts(self):
        return sum([i['wordFreq'].shape[0] for k, i in self.authorData.items() if k in self.women]) * 2

    def get_split_same_author(self):
        self.get_split(self.ratioTrain, self.ratioValid)

    def init_author_indices(self):
        for key in self.authorData.keys():
            current_author_n_excerpts = self.authorData[key]['wordFreq'].shape[0]
            random_perm = np.random.permutation(current_author_n_excerpts)
            self.author_indices[key] = random_perm

    def get_next_indices(self, author_name, n):
        """
        Function that returns the next N indices for a given author in order to ensure, that all excerpts of a particular author are used.
        :param author_name:
        :param n:
        :return:
        """
        auth_ind = self.author_indices[author_name]

        if len(auth_ind) >= n:
            # if there are more unused indices than needed, just return next indices, and remove them from available
            ind = auth_ind[:n]
            self.author_indices[author_name] = auth_ind[n:]

            return ind

        # take the remaining indices, permute all available indices, and take how many is still needed.
        result = auth_ind

        author_n_excerpts = self.authorData[author_name]['wordFreq'].shape[0]
        self.author_indices[author_name] = np.random.permutation(author_n_excerpts)

        additional_ind = self.author_indices[author_name][:n - len(result)]
        result = np.concatenate((result, additional_ind))

        self.author_indices[author_name] = self.author_indices[author_name][len(additional_ind):]

        return result

    def get_rest_uniform(self, author_name, nTrainAuthor, nValidAuthor, nTestAuthor):
        """
        Function, that returns excerpts from the rest of the authors uniformly distributed.
        :param author_name:
        :param nTrainAuthor:
        :param nValidAuthor:
        :param nTestAuthor:
        :return:
        """

        # Compute how many samples for each case
        no_of_authors_left = (len(self.authorData.keys()) - 1)

        nTrainRest = int((self.nTrain - nTrainAuthor) / no_of_authors_left)
        nValidRest = int((self.nValid - nValidAuthor) / no_of_authors_left)
        nTestRest = int((self.nTest - nTestAuthor) / no_of_authors_left)

        # Now we need to get an equal amount of works from the rest of the
        # authors.
        no_of_function_words = self.selectedAuthor['train']['wordFreq'].shape[1]

        x_rest_tr = np.empty([0, no_of_function_words])  # Create an empty matrix
        x_rest_te = np.empty([0, no_of_function_words])  # Create an empty matrix
        x_rest_val = np.empty([0, no_of_function_words])  # Create an empty matrix

        for key in self.authorData.keys():
            # Only for authors that are not the selected author
            if key != author_name:
                thisAuthorTexts = self.authorData[key]['wordFreq']

                # get the indices for a particular author, and add texts to training, test and validation sets.
                author_indices_tr = self.get_next_indices(key, nTrainRest)
                author_indices_te = self.get_next_indices(key, nTestRest)
                author_indices_val = self.get_next_indices(key, nValidRest)

                x_rest_tr = np.concatenate((x_rest_tr, thisAuthorTexts[author_indices_tr, :]), axis=0)
                x_rest_te = np.concatenate((x_rest_te, thisAuthorTexts[author_indices_te, :]), axis=0)
                x_rest_val = np.concatenate((x_rest_val, thisAuthorTexts[author_indices_val, :]), axis=0)

        x_rest_te, x_rest_tr, x_rest_val = self.add_rest_of_samples(author_name,
                                                                    self.nTest - nTestAuthor,
                                                                    self.nTrain - nTrainAuthor,
                                                                    self.nValid - nValidAuthor,
                                                                    x_rest_te, x_rest_tr,
                                                                    x_rest_val)

        return x_rest_tr, x_rest_val, x_rest_te

    def add_rest_of_samples(self, author_name, nTestAuthor, nTrainAuthor, nValidAuthor, x_rest_te, x_rest_tr,
                            x_rest_val):
        for key in self.authorData.keys():
            if x_rest_tr.shape[0] == nTrainAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_tr = self.get_next_indices(key, 1)
                x_rest_tr = np.concatenate((x_rest_tr, self.authorData[key]['wordFreq'][author_indices_tr, :]), axis=0)

        for key in self.authorData.keys():
            if x_rest_te.shape[0] == nTestAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_te = self.get_next_indices(key, 1)
                x_rest_te = np.concatenate((x_rest_te, self.authorData[key]['wordFreq'][author_indices_te, :]), axis=0)

        for key in self.authorData.keys():
            if x_rest_val.shape[0] == nValidAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_val = self.get_next_indices(key, 1)
                x_rest_val = np.concatenate((x_rest_val, self.authorData[key]['wordFreq'][author_indices_val, :]),
                                            axis=0)

        return x_rest_te, x_rest_tr, x_rest_val

    def loadData(self, dataPath):
        # TODO: Analyze if it's worth it to create a .pkl and load that
        # directly once the data has been appropriately parsed. It's just
        # that loading with hdf5storage takes a couple of second that
        # could be saved if the .pkl file is faster.
        rawData = hdf5storage.loadmat(dataPath)
        # rawData is a dictionary with four keys:
        #   'all_authors': contains the author list
        #   'all_freqs': contains the word frequency count for each excerpt
        #   'all_wans': contains the WANS for each excerpt
        #   'function_words': a list of the functional words
        # The issue is that hdf5storage, while necessary to load old
        # Matlab(R) files, gives the data in a weird format, that we need
        # to adapt and convert.
        # The data will be structured as follows. We will have an
        # authorData dictionary of dictionaries: the first key will be the
        # author name, the second key will be either freqs or wans to
        # access either one or another.
        # We will also clean up and save the functional word list, although
        # we do not need to use it.
        authorData = {}  # Create dictionary
        for it in range(len(rawData['all_authors'])):
            thisAuthor = str(rawData['all_authors'][it][0][0][0])
            # Each element in rawData['all_authors'] is nested in a couple
            # of lists, so that's why we need the three indices [0][0][0]
            # to reach the string with the actual author name.
            # Get the word frequency
            thisWordFreq = rawData['all_freqs'][0][it]  # 1 x nWords x nData
            # Again, the [0] is due to the structure of the data
            # Let us get rid of that extra 1, and then transpose this to be
            # stored as nData x nWords (since nWords is the dimension of
            # the number of nodes the network will have; CS notation)
            thisWordFreq = thisWordFreq.squeeze(0).T  # nData x nWords
            # Finally, get the WANs
            thisWAN = rawData['all_wans'][0][it]  # nWords x nWords x nData
            thisWAN = thisWAN.transpose(2, 0, 1)  # nData x nWords x nWords
            # Obs.: thisWAN is likely not symmetric, so the way this is
            # transposed matters. In this case, since thisWAN was intended
            # to be a tensor in matlab (where the last index is the
            # collection of matrices), we just throw that last dimension to
            # the front (since numpy consider the first index as the
            # collection index).
            # Now we can create the dictionary and save the corresopnding
            # data.
            authorData[thisAuthor] = {}
            authorData[thisAuthor]['wordFreq'] = thisWordFreq
            authorData[thisAuthor]['WAN'] = thisWAN
        # And at last, gather the list of functional words
        functionWords = []  # empty list to store the functional words
        for word in rawData['function_words']:
            functionWords.append(str(word[0][0][0]))
        # Store all the data recently collected
        self.authorData = authorData
        self.functionWords = functionWords


class AutorshipNationalityAmericanSO(_dataForClassification):
    """
    Authorship: Loads the dataset of 19th century writers for the authorship
        attribution problem

    Initialization:

    Input:
        authorName (string): which is the selected author to attribute plays to
        ratioTrain (float): ratio of the total texts to be part of the training
            set
        ratioValid (float): ratio of the train texts to be part of the
            validation set
        dataPath (string): path to where the authorship data is located
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:

    .loadData(dataPath): load the data found in dataPath and store it in
        attributes .authorData and .functionWords

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """

    all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                        'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving',
                        'poe',
                        'jewett', 'wharton']

    # Possible authors: (just use the names in ' ')
    # jacob 'abbott', robert louis 'stevenson', louisa may 'alcott',
    # horatio 'alger', james 'allen', jane 'austen', emily 'bronte', james 'cooper',
    # charles 'dickens', hamlin 'garland', nathaniel 'hawthorne', henry 'james',
    # herman 'melville', 'page', herny 'thoreau', mark 'twain',
    # arthur conan 'doyle', washington 'irving', edgar allan 'poe',
    # sarah orne 'jewett', edith 'wharton'

    british = ['stevenson', 'austen', 'bronte', 'allen', 'jewett', 'wharton', 'dickens', 'james', 'doyle']
    american = ['alcott', 'abbott', 'alger', 'cooper', 'garland', 'hawthorne', 'melville',
                'thoreau', 'twain', 'irving', 'poe']

    def __init__(self, ratioTrain, ratioValid, dataPath,
                 dataType=np.float64, device='cpu'):
        # Initialize parent
        super().__init__()
        # Store
        self.author_indices = {}
        self.dataType = dataType
        self.device = device

        # Load data
        self.loadData(dataPath)

        self.init_author_indices()

        self.ratioTrain = ratioTrain
        self.ratioValid = ratioValid

        self.get_split(ratioTrain, ratioValid)

    def get_split(self, ratioTrain, ratioValid):
        # start with Women as there are less excerpts and concat everything.

        british_all_frq = np.empty([0, self.authorData['abbott']['WAN'].shape[1]])
        british_all_WAN = np.empty(
            [0, self.authorData['abbott']['WAN'].shape[1], self.authorData['abbott']['WAN'].shape[1]])

        for name in self.british:
            british_all_frq = np.concatenate((british_all_frq, self.authorData[name]['wordFreq']), axis=0)
            british_all_WAN = np.concatenate((british_all_WAN, self.authorData[name]['WAN']), axis=0)

        # Get the selected author's data
        n_selected_author_excerpts = british_all_frq.shape[0]  # Number of excerpts
        # by the selected author
        nTrainAuthor = round(ratioTrain * n_selected_author_excerpts)
        nValidAuthor = round(ratioValid * n_selected_author_excerpts)
        nTestAuthor = n_selected_author_excerpts - nTrainAuthor - nValidAuthor
        # Now, we know how many training, validation and testing samples from
        # the required author. But we will also include an equal amount of
        # other authors, therefore
        n_excerpts_rest = self.get_total_no_of_excerpts() - n_selected_author_excerpts

        self.nTrain = nTrainAuthor * 2
        self.nValid = nValidAuthor * 2
        self.nTest = nTestAuthor * 2

        american_all_frq = np.empty([0, self.authorData['abbott']['WAN'].shape[1]])
        american_all_WAN = np.empty(
            [0, self.authorData['abbott']['WAN'].shape[1], self.authorData['abbott']['WAN'].shape[1]])

        for name in self.american:
            american_all_frq = np.concatenate((american_all_frq, self.authorData[name]['wordFreq']), axis=0)
            american_all_WAN = np.concatenate((american_all_WAN, self.authorData[name]['WAN']), axis=0)

        # select men excerpts randomly, as there are too much
        rand_perm_men = np.random.permutation(n_excerpts_rest)

        american_selected_sig = american_all_frq[rand_perm_men, :]
        american_selected_WAN = american_all_WAN[rand_perm_men, :, :]

        # Get a random permutation of these works, and split them accordingly
        randPerm = np.random.permutation(n_selected_author_excerpts)

        # Save the indices corresponding to each split
        randPermTrain = randPerm[0:nTrainAuthor]
        randPermValid = randPerm[nTrainAuthor:nTrainAuthor + nValidAuthor]
        randPermTest = randPerm[nTrainAuthor + nValidAuthor:]

        # xAuthor = np.concatenate((men_selected_sig, women_all_frq), axis=0)
        # xAuthor_WAN = np.concatenate((men_selected_WAN, women_all_WAN), axis=0)

        xAuthorTrain = american_selected_sig[randPermTrain, :]
        xAuthorValid = american_selected_sig[randPermValid, :]
        xAuthorTest = american_selected_sig[randPermTest, :]

        # And we will store this split
        self.selectedAuthor = {'all': {"wordFreq": american_selected_sig, "WAN": american_selected_WAN}, 'train': {}}
        # Copy all data
        # Copy word frequencies
        self.selectedAuthor['train']['wordFreq'] = xAuthorTrain.copy()
        self.selectedAuthor['valid'] = {}
        self.selectedAuthor['valid']['wordFreq'] = xAuthorValid.copy()
        self.selectedAuthor['test'] = {}
        self.selectedAuthor['test']['wordFreq'] = xAuthorTest.copy()
        # Copy WANs
        self.selectedAuthor['train']['WAN'] = \
            british_all_WAN[randPermTrain, :, :].copy()
        self.selectedAuthor['valid']['WAN'] = \
            british_all_WAN[randPermValid, :, :].copy()
        self.selectedAuthor['test']['WAN'] = \
            british_all_WAN[randPermTest, :, :].copy()

        # And now we should just get the appropriate number of texts from these
        # other authors.
        # Compute how many samples for each case
        nTrainRest = nTrainAuthor
        nValidRest = nValidAuthor
        nTestRest = nTestAuthor
        # And obtain those
        xRestTrain = british_all_frq[randPerm[0:nTrainRest], :]
        xRestValid = british_all_frq[randPerm[nTrainRest:nTrainRest + nValidRest], :]
        xRestTest = british_all_frq[randPerm[nTrainRest + nValidRest:], :]
        # Now construct the signals and labels. Signals is just the
        # concatenation of each of these excerpts. Labels is just a bunch of
        # 1s followed by a bunch of 0s
        # Obs.: The fact that the dataset is ordered now, it doesn't matter,
        # since it will be shuffled at each epoch.
        xTrain = np.concatenate((xAuthorTrain, xRestTrain), axis=0)
        labelsTrain = np.concatenate((np.ones(nTrainAuthor),
                                      np.zeros(nTrainRest)), axis=0)
        xValid = np.concatenate((xAuthorValid, xRestValid), axis=0)
        labelsValid = np.concatenate((np.ones(nValidAuthor),
                                      np.zeros(nValidRest)), axis=0)
        xTest = np.concatenate((xAuthorTest, xRestTest), axis=0)
        labelsTest = np.concatenate((np.ones(nTestAuthor),
                                     np.zeros(nTestRest)), axis=0)
        # And assign them to the required attribute samples
        self.samples = {'train': {}}
        self.samples['train']['signals'] = xTrain
        self.samples['train']['labels'] = labelsTrain
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = xValid
        self.samples['valid']['labels'] = labelsValid
        self.samples['test'] = {}
        self.samples['test']['signals'] = xTest
        self.samples['test']['labels'] = labelsTest
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    def get_total_no_of_excerpts(self):
        return sum([i['wordFreq'].shape[0] for k, i in self.authorData.items() if k in self.british]) * 2

    def get_split_same_author(self):
        self.get_split(self.ratioTrain, self.ratioValid)

    def init_author_indices(self):
        for key in self.authorData.keys():
            current_author_n_excerpts = self.authorData[key]['wordFreq'].shape[0]
            random_perm = np.random.permutation(current_author_n_excerpts)
            self.author_indices[key] = random_perm

    def get_next_indices(self, author_name, n):
        """
        Function that returns the next N indices for a given author in order to ensure, that all excerpts of a particular author are used.
        :param author_name:
        :param n:
        :return:
        """
        auth_ind = self.author_indices[author_name]

        if len(auth_ind) >= n:
            # if there are more unused indices than needed, just return next indices, and remove them from available
            ind = auth_ind[:n]
            self.author_indices[author_name] = auth_ind[n:]

            return ind

        # take the remaining indices, permute all available indices, and take how many is still needed.
        result = auth_ind

        author_n_excerpts = self.authorData[author_name]['wordFreq'].shape[0]
        self.author_indices[author_name] = np.random.permutation(author_n_excerpts)

        additional_ind = self.author_indices[author_name][:n - len(result)]
        result = np.concatenate((result, additional_ind))

        self.author_indices[author_name] = self.author_indices[author_name][len(additional_ind):]

        return result

    def get_rest_uniform(self, author_name, nTrainAuthor, nValidAuthor, nTestAuthor):
        """
        Function, that returns excerpts from the rest of the authors uniformly distributed.
        :param author_name:
        :param nTrainAuthor:
        :param nValidAuthor:
        :param nTestAuthor:
        :return:
        """

        # Compute how many samples for each case
        no_of_authors_left = (len(self.authorData.keys()) - 1)

        nTrainRest = int((self.nTrain - nTrainAuthor) / no_of_authors_left)
        nValidRest = int((self.nValid - nValidAuthor) / no_of_authors_left)
        nTestRest = int((self.nTest - nTestAuthor) / no_of_authors_left)

        # Now we need to get an equal amount of works from the rest of the
        # authors.
        no_of_function_words = self.selectedAuthor['train']['wordFreq'].shape[1]

        x_rest_tr = np.empty([0, no_of_function_words])  # Create an empty matrix
        x_rest_te = np.empty([0, no_of_function_words])  # Create an empty matrix
        x_rest_val = np.empty([0, no_of_function_words])  # Create an empty matrix

        for key in self.authorData.keys():
            # Only for authors that are not the selected author
            if key != author_name:
                thisAuthorTexts = self.authorData[key]['wordFreq']

                # get the indices for a particular author, and add texts to training, test and validation sets.
                author_indices_tr = self.get_next_indices(key, nTrainRest)
                author_indices_te = self.get_next_indices(key, nTestRest)
                author_indices_val = self.get_next_indices(key, nValidRest)

                x_rest_tr = np.concatenate((x_rest_tr, thisAuthorTexts[author_indices_tr, :]), axis=0)
                x_rest_te = np.concatenate((x_rest_te, thisAuthorTexts[author_indices_te, :]), axis=0)
                x_rest_val = np.concatenate((x_rest_val, thisAuthorTexts[author_indices_val, :]), axis=0)

        x_rest_te, x_rest_tr, x_rest_val = self.add_rest_of_samples(author_name,
                                                                    self.nTest - nTestAuthor,
                                                                    self.nTrain - nTrainAuthor,
                                                                    self.nValid - nValidAuthor,
                                                                    x_rest_te, x_rest_tr,
                                                                    x_rest_val)

        return x_rest_tr, x_rest_val, x_rest_te

    def add_rest_of_samples(self, author_name, nTestAuthor, nTrainAuthor, nValidAuthor, x_rest_te, x_rest_tr,
                            x_rest_val):
        for key in self.authorData.keys():
            if x_rest_tr.shape[0] == nTrainAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_tr = self.get_next_indices(key, 1)
                x_rest_tr = np.concatenate((x_rest_tr, self.authorData[key]['wordFreq'][author_indices_tr, :]), axis=0)

        for key in self.authorData.keys():
            if x_rest_te.shape[0] == nTestAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_te = self.get_next_indices(key, 1)
                x_rest_te = np.concatenate((x_rest_te, self.authorData[key]['wordFreq'][author_indices_te, :]), axis=0)

        for key in self.authorData.keys():
            if x_rest_val.shape[0] == nValidAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_val = self.get_next_indices(key, 1)
                x_rest_val = np.concatenate((x_rest_val, self.authorData[key]['wordFreq'][author_indices_val, :]),
                                            axis=0)

        return x_rest_te, x_rest_tr, x_rest_val

    def loadData(self, dataPath):
        # TODO: Analyze if it's worth it to create a .pkl and load that
        # directly once the data has been appropriately parsed. It's just
        # that loading with hdf5storage takes a couple of second that
        # could be saved if the .pkl file is faster.
        rawData = hdf5storage.loadmat(dataPath)
        # rawData is a dictionary with four keys:
        #   'all_authors': contains the author list
        #   'all_freqs': contains the word frequency count for each excerpt
        #   'all_wans': contains the WANS for each excerpt
        #   'function_words': a list of the functional words
        # The issue is that hdf5storage, while necessary to load old
        # Matlab(R) files, gives the data in a weird format, that we need
        # to adapt and convert.
        # The data will be structured as follows. We will have an
        # authorData dictionary of dictionaries: the first key will be the
        # author name, the second key will be either freqs or wans to
        # access either one or another.
        # We will also clean up and save the functional word list, although
        # we do not need to use it.
        authorData = {}  # Create dictionary
        for it in range(len(rawData['all_authors'])):
            thisAuthor = str(rawData['all_authors'][it][0][0][0])
            # Each element in rawData['all_authors'] is nested in a couple
            # of lists, so that's why we need the three indices [0][0][0]
            # to reach the string with the actual author name.
            # Get the word frequency
            thisWordFreq = rawData['all_freqs'][0][it]  # 1 x nWords x nData
            # Again, the [0] is due to the structure of the data
            # Let us get rid of that extra 1, and then transpose this to be
            # stored as nData x nWords (since nWords is the dimension of
            # the number of nodes the network will have; CS notation)
            thisWordFreq = thisWordFreq.squeeze(0).T  # nData x nWords
            # Finally, get the WANs
            thisWAN = rawData['all_wans'][0][it]  # nWords x nWords x nData
            thisWAN = thisWAN.transpose(2, 0, 1)  # nData x nWords x nWords
            # Obs.: thisWAN is likely not symmetric, so the way this is
            # transposed matters. In this case, since thisWAN was intended
            # to be a tensor in matlab (where the last index is the
            # collection of matrices), we just throw that last dimension to
            # the front (since numpy consider the first index as the
            # collection index).
            # Now we can create the dictionary and save the corresopnding
            # data.
            authorData[thisAuthor] = {}
            authorData[thisAuthor]['wordFreq'] = thisWordFreq
            authorData[thisAuthor]['WAN'] = thisWAN
        # And at last, gather the list of functional words
        functionWords = []  # empty list to store the functional words
        for word in rawData['function_words']:
            functionWords.append(str(word[0][0][0]))
        # Store all the data recently collected
        self.authorData = authorData
        self.functionWords = functionWords


class AutorshipNationality(_dataForClassification):
    """
    Authorship: Loads the dataset of 19th century writers for the authorship
        attribution problem

    Initialization:

    Input:
        authorName (string): which is the selected author to attribute plays to
        ratioTrain (float): ratio of the total texts to be part of the training
            set
        ratioValid (float): ratio of the train texts to be part of the
            validation set
        dataPath (string): path to where the authorship data is located
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:

    .loadData(dataPath): load the data found in dataPath and store it in
        attributes .authorData and .functionWords

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """

    all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                        'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving',
                        'poe',
                        'jewett', 'wharton']

    # Possible authors: (just use the names in ' ')
    # jacob 'abbott', robert louis 'stevenson', louisa may 'alcott',
    # horatio 'alger', james 'allen', jane 'austen', emily 'bronte', james 'cooper',
    # charles 'dickens', hamlin 'garland', nathaniel 'hawthorne', henry 'james',
    # herman 'melville', 'page', herny 'thoreau', mark 'twain',
    # arthur conan 'doyle', washington 'irving', edgar allan 'poe',
    # sarah orne 'jewett', edith 'wharton'

    british = ['stevenson', 'austen', 'bronte', 'allen', 'jewett', 'wharton', 'dickens', 'james', 'doyle']
    american = ['alcott', 'abbott', 'alger', 'cooper', 'garland', 'hawthorne', 'melville',
                'thoreau', 'twain', 'irving', 'poe']

    def __init__(self, ratioTrain, ratioValid, dataPath,
                 dataType=np.float64, device='cpu'):
        # Initialize parent
        super().__init__()
        # Store
        self.author_indices = {}
        self.dataType = dataType
        self.device = device

        # Load data
        self.loadData(dataPath)

        self.init_author_indices()

        self.ratioTrain = ratioTrain
        self.ratioValid = ratioValid

        self.get_split(ratioTrain, ratioValid)

    def get_split(self, ratioTrain, ratioValid):
        # start with Women as there are less excerpts and concat everything.

        british_all_frq = np.empty([0, self.authorData['abbott']['WAN'].shape[1]])
        british_all_WAN = np.empty(
            [0, self.authorData['abbott']['WAN'].shape[1], self.authorData['abbott']['WAN'].shape[1]])

        for name in self.british:
            british_all_frq = np.concatenate((british_all_frq, self.authorData[name]['wordFreq']), axis=0)
            british_all_WAN = np.concatenate((british_all_WAN, self.authorData[name]['WAN']), axis=0)

        # Get the selected author's data
        n_selected_author_excerpts = british_all_frq.shape[0]  # Number of excerpts
        # by the selected author
        nTrainAuthor = round(ratioTrain * n_selected_author_excerpts)
        nValidAuthor = round(ratioValid * n_selected_author_excerpts)
        nTestAuthor = n_selected_author_excerpts - nTrainAuthor - nValidAuthor
        # Now, we know how many training, validation and testing samples from
        # the required author. But we will also include an equal amount of
        # other authors, therefore
        n_excerpts_rest = self.get_total_no_of_excerpts() - n_selected_author_excerpts

        self.nTrain = nTrainAuthor * 2
        self.nValid = nValidAuthor * 2
        self.nTest = nTestAuthor * 2

        american_all_frq = np.empty([0, self.authorData['abbott']['WAN'].shape[1]])
        american_all_WAN = np.empty(
            [0, self.authorData['abbott']['WAN'].shape[1], self.authorData['abbott']['WAN'].shape[1]])

        for name in self.american:
            american_all_frq = np.concatenate((american_all_frq, self.authorData[name]['wordFreq']), axis=0)
            american_all_WAN = np.concatenate((american_all_WAN, self.authorData[name]['WAN']), axis=0)

        # select men excerpts randomly, as there are too much
        rand_perm_men = np.random.permutation(n_excerpts_rest)

        american_selected_sig = american_all_frq[rand_perm_men, :]
        american_selected_WAN = american_all_WAN[rand_perm_men, :, :]

        # Get a random permutation of these works, and split them accordingly
        randPerm = np.random.permutation(n_selected_author_excerpts)

        # Save the indices corresponding to each split
        randPermTrain = randPerm[0:nTrainAuthor]
        randPermValid = randPerm[nTrainAuthor:nTrainAuthor + nValidAuthor]
        randPermTest = randPerm[nTrainAuthor + nValidAuthor:]

        # xAuthor = np.concatenate((men_selected_sig, women_all_frq), axis=0)
        # xAuthor_WAN = np.concatenate((men_selected_WAN, women_all_WAN), axis=0)

        xAuthorTrain = british_all_frq[randPermTrain, :]
        xAuthorValid = british_all_frq[randPermValid, :]
        xAuthorTest = british_all_frq[randPermTest, :]

        # And we will store this split
        self.selectedAuthor = {'all': {"wordFreq": british_all_frq, "WAN": british_all_WAN}, 'train': {}}
        # Copy all data
        # Copy word frequencies
        self.selectedAuthor['train']['wordFreq'] = xAuthorTrain.copy()
        self.selectedAuthor['valid'] = {}
        self.selectedAuthor['valid']['wordFreq'] = xAuthorValid.copy()
        self.selectedAuthor['test'] = {}
        self.selectedAuthor['test']['wordFreq'] = xAuthorTest.copy()
        # Copy WANs
        self.selectedAuthor['train']['WAN'] = \
            british_all_WAN[randPermTrain, :, :].copy()
        self.selectedAuthor['valid']['WAN'] = \
            british_all_WAN[randPermValid, :, :].copy()
        self.selectedAuthor['test']['WAN'] = \
            british_all_WAN[randPermTest, :, :].copy()

        # And now we should just get the appropriate number of texts from these
        # other authors.
        # Compute how many samples for each case
        nTrainRest = nTrainAuthor
        nValidRest = nValidAuthor
        nTestRest = nTestAuthor
        # And obtain those
        xRestTrain = american_selected_sig[randPerm[0:nTrainRest], :]
        xRestValid = american_selected_sig[randPerm[nTrainRest:nTrainRest + nValidRest], :]
        xRestTest = american_selected_sig[randPerm[nTrainRest + nValidRest:], :]
        # Now construct the signals and labels. Signals is just the
        # concatenation of each of these excerpts. Labels is just a bunch of
        # 1s followed by a bunch of 0s
        # Obs.: The fact that the dataset is ordered now, it doesn't matter,
        # since it will be shuffled at each epoch.
        xTrain = np.concatenate((xAuthorTrain, xRestTrain), axis=0)
        labelsTrain = np.concatenate((np.ones(nTrainAuthor),
                                      np.zeros(nTrainRest)), axis=0)
        xValid = np.concatenate((xAuthorValid, xRestValid), axis=0)
        labelsValid = np.concatenate((np.ones(nValidAuthor),
                                      np.zeros(nValidRest)), axis=0)
        xTest = np.concatenate((xAuthorTest, xRestTest), axis=0)
        labelsTest = np.concatenate((np.ones(nTestAuthor),
                                     np.zeros(nTestRest)), axis=0)

        tr_perm = np.random.permutation(len(xTrain))
        test_perm = np.random.permutation(len(xTest))
        valid_perm = np.random.permutation(len(xValid))

        # And assign them to the required attribute samples
        self.samples = {'train': {}}
        self.samples['train']['signals'] = xTrain[tr_perm]
        self.samples['train']['labels'] = labelsTrain[tr_perm]
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = xValid[valid_perm]
        self.samples['valid']['labels'] = labelsValid[valid_perm]
        self.samples['test'] = {}
        self.samples['test']['signals'] = xTest[test_perm]
        self.samples['test']['labels'] = labelsTest[test_perm]
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    def get_total_no_of_excerpts(self):
        return sum([i['wordFreq'].shape[0] for k, i in self.authorData.items() if k in self.british]) * 2

    def get_split_same_author(self):
        self.get_split(self.ratioTrain, self.ratioValid)

    def init_author_indices(self):
        for key in self.authorData.keys():
            current_author_n_excerpts = self.authorData[key]['wordFreq'].shape[0]
            random_perm = np.random.permutation(current_author_n_excerpts)
            self.author_indices[key] = random_perm

    def get_next_indices(self, author_name, n):
        """
        Function that returns the next N indices for a given author in order to ensure, that all excerpts of a particular author are used.
        :param author_name:
        :param n:
        :return:
        """
        auth_ind = self.author_indices[author_name]

        if len(auth_ind) >= n:
            # if there are more unused indices than needed, just return next indices, and remove them from available
            ind = auth_ind[:n]
            self.author_indices[author_name] = auth_ind[n:]

            return ind

        # take the remaining indices, permute all available indices, and take how many is still needed.
        result = auth_ind

        author_n_excerpts = self.authorData[author_name]['wordFreq'].shape[0]
        self.author_indices[author_name] = np.random.permutation(author_n_excerpts)

        additional_ind = self.author_indices[author_name][:n - len(result)]
        result = np.concatenate((result, additional_ind))

        self.author_indices[author_name] = self.author_indices[author_name][len(additional_ind):]

        return result

    def get_rest_uniform(self, author_name, nTrainAuthor, nValidAuthor, nTestAuthor):
        """
        Function, that returns excerpts from the rest of the authors uniformly distributed.
        :param author_name:
        :param nTrainAuthor:
        :param nValidAuthor:
        :param nTestAuthor:
        :return:
        """

        # Compute how many samples for each case
        no_of_authors_left = (len(self.authorData.keys()) - 1)

        nTrainRest = int((self.nTrain - nTrainAuthor) / no_of_authors_left)
        nValidRest = int((self.nValid - nValidAuthor) / no_of_authors_left)
        nTestRest = int((self.nTest - nTestAuthor) / no_of_authors_left)

        # Now we need to get an equal amount of works from the rest of the
        # authors.
        no_of_function_words = self.selectedAuthor['train']['wordFreq'].shape[1]

        x_rest_tr = np.empty([0, no_of_function_words])  # Create an empty matrix
        x_rest_te = np.empty([0, no_of_function_words])  # Create an empty matrix
        x_rest_val = np.empty([0, no_of_function_words])  # Create an empty matrix

        for key in self.authorData.keys():
            # Only for authors that are not the selected author
            if key != author_name:
                thisAuthorTexts = self.authorData[key]['wordFreq']

                # get the indices for a particular author, and add texts to training, test and validation sets.
                author_indices_tr = self.get_next_indices(key, nTrainRest)
                author_indices_te = self.get_next_indices(key, nTestRest)
                author_indices_val = self.get_next_indices(key, nValidRest)

                x_rest_tr = np.concatenate((x_rest_tr, thisAuthorTexts[author_indices_tr, :]), axis=0)
                x_rest_te = np.concatenate((x_rest_te, thisAuthorTexts[author_indices_te, :]), axis=0)
                x_rest_val = np.concatenate((x_rest_val, thisAuthorTexts[author_indices_val, :]), axis=0)

        x_rest_te, x_rest_tr, x_rest_val = self.add_rest_of_samples(author_name,
                                                                    self.nTest - nTestAuthor,
                                                                    self.nTrain - nTrainAuthor,
                                                                    self.nValid - nValidAuthor,
                                                                    x_rest_te, x_rest_tr,
                                                                    x_rest_val)

        return x_rest_tr, x_rest_val, x_rest_te

    def add_rest_of_samples(self, author_name, nTestAuthor, nTrainAuthor, nValidAuthor, x_rest_te, x_rest_tr,
                            x_rest_val):
        for key in self.authorData.keys():
            if x_rest_tr.shape[0] == nTrainAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_tr = self.get_next_indices(key, 1)
                x_rest_tr = np.concatenate((x_rest_tr, self.authorData[key]['wordFreq'][author_indices_tr, :]), axis=0)

        for key in self.authorData.keys():
            if x_rest_te.shape[0] == nTestAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_te = self.get_next_indices(key, 1)
                x_rest_te = np.concatenate((x_rest_te, self.authorData[key]['wordFreq'][author_indices_te, :]), axis=0)

        for key in self.authorData.keys():
            if x_rest_val.shape[0] == nValidAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_val = self.get_next_indices(key, 1)
                x_rest_val = np.concatenate((x_rest_val, self.authorData[key]['wordFreq'][author_indices_val, :]),
                                            axis=0)

        return x_rest_te, x_rest_tr, x_rest_val

    def loadData(self, dataPath):
        # TODO: Analyze if it's worth it to create a .pkl and load that
        # directly once the data has been appropriately parsed. It's just
        # that loading with hdf5storage takes a couple of second that
        # could be saved if the .pkl file is faster.
        rawData = hdf5storage.loadmat(dataPath)
        # rawData is a dictionary with four keys:
        #   'all_authors': contains the author list
        #   'all_freqs': contains the word frequency count for each excerpt
        #   'all_wans': contains the WANS for each excerpt
        #   'function_words': a list of the functional words
        # The issue is that hdf5storage, while necessary to load old
        # Matlab(R) files, gives the data in a weird format, that we need
        # to adapt and convert.
        # The data will be structured as follows. We will have an
        # authorData dictionary of dictionaries: the first key will be the
        # author name, the second key will be either freqs or wans to
        # access either one or another.
        # We will also clean up and save the functional word list, although
        # we do not need to use it.
        authorData = {}  # Create dictionary
        for it in range(len(rawData['all_authors'])):
            thisAuthor = str(rawData['all_authors'][it][0][0][0])
            # Each element in rawData['all_authors'] is nested in a couple
            # of lists, so that's why we need the three indices [0][0][0]
            # to reach the string with the actual author name.
            # Get the word frequency
            thisWordFreq = rawData['all_freqs'][0][it]  # 1 x nWords x nData
            # Again, the [0] is due to the structure of the data
            # Let us get rid of that extra 1, and then transpose this to be
            # stored as nData x nWords (since nWords is the dimension of
            # the number of nodes the network will have; CS notation)
            thisWordFreq = thisWordFreq.squeeze(0).T  # nData x nWords
            # Finally, get the WANs
            thisWAN = rawData['all_wans'][0][it]  # nWords x nWords x nData
            thisWAN = thisWAN.transpose(2, 0, 1)  # nData x nWords x nWords
            # Obs.: thisWAN is likely not symmetric, so the way this is
            # transposed matters. In this case, since thisWAN was intended
            # to be a tensor in matlab (where the last index is the
            # collection of matrices), we just throw that last dimension to
            # the front (since numpy consider the first index as the
            # collection index).
            # Now we can create the dictionary and save the corresopnding
            # data.
            authorData[thisAuthor] = {}
            authorData[thisAuthor]['wordFreq'] = thisWordFreq
            authorData[thisAuthor]['WAN'] = thisWAN
        # And at last, gather the list of functional words
        functionWords = []  # empty list to store the functional words
        for word in rawData['function_words']:
            functionWords.append(str(word[0][0][0]))
        # Store all the data recently collected
        self.authorData = authorData
        self.functionWords = functionWords


class AuthorshipAll(_dataForClassification):
    """
    Authorship: Loads the dataset of 19th century writers for the authorship
        attribution problem

    Initialization:

    Input:
        authorName (string): which is the selected author to attribute plays to
        ratioTrain (float): ratio of the total texts to be part of the training
            set
        ratioValid (float): ratio of the train texts to be part of the
            validation set
        dataPath (string): path to where the authorship data is located
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:

    .loadData(dataPath): load the data found in dataPath and store it in
        attributes .authorData and .functionWords

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """

    def __init__(self, authorName, ratioTrain, ratioValid, dataPath,
                 dataType=np.float64, device='cpu'):
        # Initialize parent
        super().__init__()
        # Store
        self.author_indices = {}
        self.dataType = dataType
        self.device = device

        # Load data
        self.loadData(dataPath)

        self.init_author_indices()

        # Check that the authorName is a valid name
        assert authorName in self.authorData.keys()

        self.authorName = authorName
        self.ratioTrain = ratioTrain
        self.ratioValid = ratioValid

        self.get_split(authorName, ratioTrain, ratioValid)

    def get_split(self, authorName, ratioTrain, ratioValid):
        # Get the selected author's data
        thisAuthorData = self.authorData[authorName].copy()
        n_selected_author_excerpts = thisAuthorData['wordFreq'].shape[0]  # Number of excerpts
        # by the selected author
        nTrainAuthor = round(ratioTrain * n_selected_author_excerpts)
        nValidAuthor = round(ratioValid * n_selected_author_excerpts)
        nTestAuthor = n_selected_author_excerpts - nTrainAuthor - nValidAuthor
        # Now, we know how many training, validation and testing samples from
        # the required author. But we will also include an equal amount of
        # other authors, therefore
        n_excerpts_rest = self.get_total_no_of_excerpts() - n_selected_author_excerpts

        # all other authors times ration, plus selected author times ratio.
        self.nTrain = round(n_excerpts_rest * ratioTrain + nTrainAuthor)
        self.nValid = round(n_excerpts_rest * ratioValid + nValidAuthor)
        self.nTest = self.get_total_no_of_excerpts() - self.nValid - self.nTrain
        # Now, let's get the corresponding signals for the author
        xAuthor = thisAuthorData['wordFreq']
        # Get a random permutation of these works, and split them accordingly
        randPerm = np.random.permutation(n_selected_author_excerpts)

        # Save the indices corresponding to each split
        randPermTrain = randPerm[0:nTrainAuthor]
        randPermValid = randPerm[nTrainAuthor:nTrainAuthor + nValidAuthor]
        randPermTest = randPerm[nTrainAuthor + nValidAuthor:]

        xAuthorTrain = xAuthor[randPermTrain, :]
        xAuthorValid = xAuthor[randPermValid, :]
        xAuthorTest = xAuthor[randPermTest, :]
        # And we will store this split
        self.selectedAuthor = {'all': thisAuthorData.copy(), 'train': {}}
        # Copy all data
        # Copy word frequencies
        self.selectedAuthor['train']['wordFreq'] = xAuthorTrain.copy()
        self.selectedAuthor['valid'] = {}
        self.selectedAuthor['valid']['wordFreq'] = xAuthorValid.copy()
        self.selectedAuthor['test'] = {}
        self.selectedAuthor['test']['wordFreq'] = xAuthorTest.copy()
        # Copy WANs
        self.selectedAuthor['train']['WAN'] = \
            thisAuthorData['WAN'][randPermTrain, :, :].copy()
        self.selectedAuthor['valid']['WAN'] = \
            thisAuthorData['WAN'][randPermValid, :, :].copy()
        self.selectedAuthor['test']['WAN'] = \
            thisAuthorData['WAN'][randPermTest, :, :].copy()
        # Now we need to get an equal amount of works from the rest of the
        # authors.
        xRest = np.empty([0, xAuthorTrain.shape[1]])  # Create an empty matrix
        # to store all the works by the rest of the authors.
        # Now go author by author gathering all works
        for key in self.authorData.keys():
            # Only for authors that are not the selected author
            if key != authorName:
                thisAuthorTexts = self.authorData[key]['wordFreq']
                xRest = np.concatenate((xRest, thisAuthorTexts), axis=0)
        # After obtaining all works, xRest is of shape nRestOfData x nWords
        # We now need to select at random from this other data, but only up
        # to nExcerpts. Therefore, we will randperm all the indices, but keep
        # only the first nExcerpts indices.
        randPerm = np.random.permutation(xRest.shape[0])
        # randPerm = randPerm[0:nExcerpts]  # nExcerpts x nWords
        # And now we should just get the appropriate number of texts from these
        # other authors.
        # Compute how many samples for each case
        nTrainRest = self.nTrain - nTrainAuthor
        nValidRest = self.nValid - nValidAuthor
        nTestRest = self.nTest - nTestAuthor
        # And obtain those
        xRestTrain = xRest[randPerm[0:nTrainRest], :]
        xRestValid = xRest[randPerm[nTrainRest:nTrainRest + nValidRest], :]
        xRestTest = xRest[randPerm[nTrainRest + nValidRest:], :]
        # Now construct the signals and labels. Signals is just the
        # concatenation of each of these excerpts. Labels is just a bunch of
        # 1s followed by a bunch of 0s
        # Obs.: The fact that the dataset is ordered now, it doesn't matter,
        # since it will be shuffled at each epoch.
        xTrain = np.concatenate((xAuthorTrain, xRestTrain), axis=0)
        labelsTrain = np.concatenate((np.ones(nTrainAuthor),
                                      np.zeros(nTrainRest)), axis=0)
        xValid = np.concatenate((xAuthorValid, xRestValid), axis=0)
        labelsValid = np.concatenate((np.ones(nValidAuthor),
                                      np.zeros(nValidRest)), axis=0)
        xTest = np.concatenate((xAuthorTest, xRestTest), axis=0)
        labelsTest = np.concatenate((np.ones(nTestAuthor),
                                     np.zeros(nTestRest)), axis=0)
        # And assign them to the required attribute samples
        self.samples = {'train': {}}
        self.samples['train']['signals'] = xTrain
        self.samples['train']['labels'] = labelsTrain
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = xValid
        self.samples['valid']['labels'] = labelsValid
        self.samples['test'] = {}
        self.samples['test']['signals'] = xTest
        self.samples['test']['labels'] = labelsTest
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    def get_total_no_of_excerpts(self):
        return sum([i['wordFreq'].shape[0] for i in self.authorData.values()])

    def get_split_same_author(self):
        self.get_split(self.authorName, self.ratioTrain, self.ratioValid)

    def init_author_indices(self):
        for key in self.authorData.keys():
            current_author_n_excerpts = self.authorData[key]['wordFreq'].shape[0]
            random_perm = np.random.permutation(current_author_n_excerpts)
            self.author_indices[key] = random_perm

    def get_next_indices(self, author_name, n):
        """
        Function that returns the next N indices for a given author in order to ensure, that all excerpts of a particular author are used.
        :param author_name:
        :param n:
        :return:
        """
        auth_ind = self.author_indices[author_name]

        if len(auth_ind) >= n:
            # if there are more unused indices than needed, just return next indices, and remove them from available
            ind = auth_ind[:n]
            self.author_indices[author_name] = auth_ind[n:]

            return ind

        # take the remaining indices, permute all available indices, and take how many is still needed.
        result = auth_ind

        author_n_excerpts = self.authorData[author_name]['wordFreq'].shape[0]
        self.author_indices[author_name] = np.random.permutation(author_n_excerpts)

        additional_ind = self.author_indices[author_name][:n - len(result)]
        result = np.concatenate((result, additional_ind))

        self.author_indices[author_name] = self.author_indices[author_name][len(additional_ind):]

        return result

    def get_rest_uniform(self, author_name, nTrainAuthor, nValidAuthor, nTestAuthor):
        """
        Function, that returns excerpts from the rest of the authors uniformly distributed.
        :param author_name:
        :param nTrainAuthor:
        :param nValidAuthor:
        :param nTestAuthor:
        :return:
        """

        # Compute how many samples for each case
        no_of_authors_left = (len(self.authorData.keys()) - 1)

        nTrainRest = int((self.nTrain - nTrainAuthor) / no_of_authors_left)
        nValidRest = int((self.nValid - nValidAuthor) / no_of_authors_left)
        nTestRest = int((self.nTest - nTestAuthor) / no_of_authors_left)

        # Now we need to get an equal amount of works from the rest of the
        # authors.
        no_of_function_words = self.selectedAuthor['train']['wordFreq'].shape[1]

        x_rest_tr = np.empty([0, no_of_function_words])  # Create an empty matrix
        x_rest_te = np.empty([0, no_of_function_words])  # Create an empty matrix
        x_rest_val = np.empty([0, no_of_function_words])  # Create an empty matrix

        for key in self.authorData.keys():
            # Only for authors that are not the selected author
            if key != author_name:
                thisAuthorTexts = self.authorData[key]['wordFreq']

                # get the indices for a particular author, and add texts to training, test and validation sets.
                author_indices_tr = self.get_next_indices(key, nTrainRest)
                author_indices_te = self.get_next_indices(key, nTestRest)
                author_indices_val = self.get_next_indices(key, nValidRest)

                x_rest_tr = np.concatenate((x_rest_tr, thisAuthorTexts[author_indices_tr, :]), axis=0)
                x_rest_te = np.concatenate((x_rest_te, thisAuthorTexts[author_indices_te, :]), axis=0)
                x_rest_val = np.concatenate((x_rest_val, thisAuthorTexts[author_indices_val, :]), axis=0)

        x_rest_te, x_rest_tr, x_rest_val = self.add_rest_of_samples(author_name,
                                                                    self.nTest - nTestAuthor,
                                                                    self.nTrain - nTrainAuthor,
                                                                    self.nValid - nValidAuthor,
                                                                    x_rest_te, x_rest_tr,
                                                                    x_rest_val)

        return x_rest_tr, x_rest_val, x_rest_te

    def add_rest_of_samples(self, author_name, nTestAuthor, nTrainAuthor, nValidAuthor, x_rest_te, x_rest_tr,
                            x_rest_val):
        for key in self.authorData.keys():
            if x_rest_tr.shape[0] == nTrainAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_tr = self.get_next_indices(key, 1)
                x_rest_tr = np.concatenate((x_rest_tr, self.authorData[key]['wordFreq'][author_indices_tr, :]), axis=0)

        for key in self.authorData.keys():
            if x_rest_te.shape[0] == nTestAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_te = self.get_next_indices(key, 1)
                x_rest_te = np.concatenate((x_rest_te, self.authorData[key]['wordFreq'][author_indices_te, :]), axis=0)

        for key in self.authorData.keys():
            if x_rest_val.shape[0] == nValidAuthor:
                break
            # Only for authors that are not the selected author
            if key != author_name:
                author_indices_val = self.get_next_indices(key, 1)
                x_rest_val = np.concatenate((x_rest_val, self.authorData[key]['wordFreq'][author_indices_val, :]),
                                            axis=0)

        return x_rest_te, x_rest_tr, x_rest_val

    def loadData(self, dataPath):
        # TODO: Analyze if it's worth it to create a .pkl and load that
        # directly once the data has been appropriately parsed. It's just
        # that loading with hdf5storage takes a couple of second that
        # could be saved if the .pkl file is faster.
        rawData = hdf5storage.loadmat(dataPath)
        # rawData is a dictionary with four keys:
        #   'all_authors': contains the author list
        #   'all_freqs': contains the word frequency count for each excerpt
        #   'all_wans': contains the WANS for each excerpt
        #   'function_words': a list of the functional words
        # The issue is that hdf5storage, while necessary to load old
        # Matlab(R) files, gives the data in a weird format, that we need
        # to adapt and convert.
        # The data will be structured as follows. We will have an
        # authorData dictionary of dictionaries: the first key will be the
        # author name, the second key will be either freqs or wans to
        # access either one or another.
        # We will also clean up and save the functional word list, although
        # we do not need to use it.
        authorData = {}  # Create dictionary
        for it in range(len(rawData['all_authors'])):
            thisAuthor = str(rawData['all_authors'][it][0][0][0])
            # Each element in rawData['all_authors'] is nested in a couple
            # of lists, so that's why we need the three indices [0][0][0]
            # to reach the string with the actual author name.
            # Get the word frequency
            thisWordFreq = rawData['all_freqs'][0][it]  # 1 x nWords x nData
            # Again, the [0] is due to the structure of the data
            # Let us get rid of that extra 1, and then transpose this to be
            # stored as nData x nWords (since nWords is the dimension of
            # the number of nodes the network will have; CS notation)
            thisWordFreq = thisWordFreq.squeeze(0).T  # nData x nWords
            # Finally, get the WANs
            thisWAN = rawData['all_wans'][0][it]  # nWords x nWords x nData
            thisWAN = thisWAN.transpose(2, 0, 1)  # nData x nWords x nWords
            # Obs.: thisWAN is likely not symmetric, so the way this is
            # transposed matters. In this case, since thisWAN was intended
            # to be a tensor in matlab (where the last index is the
            # collection of matrices), we just throw that last dimension to
            # the front (since numpy consider the first index as the
            # collection index).
            # Now we can create the dictionary and save the corresopnding
            # data.
            authorData[thisAuthor] = {}
            authorData[thisAuthor]['wordFreq'] = thisWordFreq
            authorData[thisAuthor]['WAN'] = thisWAN
        # And at last, gather the list of functional words
        functionWords = []  # empty list to store the functional words
        for word in rawData['function_words']:
            functionWords.append(str(word[0][0][0]))
        # Store all the data recently collected
        self.authorData = authorData
        self.functionWords = functionWords
