import itertools
import json
import os
import numpy as np
import torch
from torch import optim, nn
import Utils.dataTools
from torch.utils.data import Dataset, TensorDataset, DataLoader

# the results from each run
N_EPOCH = 2
N_DATA_SPLITS = 1

dataDir = 'authorData'  # Data directory
dataFilename = 'authorshipData.mat'  # Data filename
dataPath = os.path.join(dataDir, dataFilename)  # Data path

ratio_train = 0.8
ratio_valid = 0.05
ratio_test = 0.15

doPrint = True

# \\\ Determine processing unit:
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
# Notify:
if doPrint:
    print("Device selected: %s" % device)


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


# %%##################################################################
########## DEFINE THE MODEL #########

class SingleFeedforward(torch.nn.Module):
    def __init__(self, input_size):
        super(SingleFeedforward, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(self.input_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        output = self.sigmoid(hidden)
        return output


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


class ThreeFeedForward(torch.nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        super(ThreeFeedForward, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, hidden_size_2)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(self.hidden_size_2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu1(hidden)
        output = self.fc2(relu)
        relu = self.relu2(output)
        output = self.fc3(relu)
        output = self.sigmoid(output)

        return output


# %%##################################################################
# LOAD DATA
# data = Utils.dataTools.Authorship('poe', ratio_train, ratio_valid, dataPath)
# xTrain, yTrain = data.getSamples('train')
#
# x_train_tensor = torch.from_numpy(xTrain).float()
# y_train_tensor = torch.from_numpy(yTrain).long()
#
# train_data = CustomDataset(x_train_tensor, y_train_tensor)
# train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)
#
# nNodes = data.selectedAuthor['all']['wordFreq'].shape[1]

# %%##################################################################
# TRAIN


print('Finished Training')


# %%##################################################################
# TRAINING FUNCTIONS
def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        optimizer.zero_grad()
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(yhat, y)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


def do_training(lr, nNodes, loader):
    # network = Feedforward(nNodes, 8)
    network = ThreeFeedForward(nNodes, nNodes * 4, nNodes)
    # network = SingleFeedforward(nNodes)

    network = network.double()

    optimizer = optim.Adam(network.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()

    train_step = make_train_step(network, criterion, optimizer)

    running_loss = 0.0
    for epoch in range(N_EPOCH):  # loop over the dataset multiple times
        i = 0
        for x_batch, y_batch in loader:
            # x_batch = x_batch.to(device)
            # y_batch = y_batch.to(device)

            # forward + backward + optimize
            loss = train_step(x_batch, y_batch)

            # print statistics
            running_loss += loss

            # if i % 10 == 9:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1))
            running_loss = 0.0

            i += 1

    return network


def validate(net, data):
    xTest, yTest = data.getSamples('test')
    # x_test_tensor = torch.from_numpy(xTest).float()
    # y_test_tensor = torch.from_numpy(yTest).long()
    x_test_tensor = xTest
    y_test_tensor = yTest.type(torch.double)

    if doPrint:
        print("Total testing accuracy (Best):", flush=True)
    # Update order and adapt dimensions (this data has one input feature,
    # so we need to add that dimension)
    with torch.no_grad():
        # Process the samples
        yHatTest = net(x_test_tensor)
        yHatTest = np.round(yHatTest)
        yHatTest = yHatTest.squeeze(1)

        totalErrors = torch.sum(torch.abs(yHatTest - yTest) > 1e-9)
        accuracy = 1 - totalErrors.item() / len(yTest)

        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the accuracy
        thisAccBest = accuracy
    if doPrint:
        print("%s: %4.2f%%" % ("SingleFeedforward", thisAccBest * 100.), flush=True)

    return thisAccBest


# %%##################################################################
# LEARNING LOOP
def main_loop():
    LEARNING_RATES = [0.001]
    BATCH_SIZES = [32]
    combinations = list(itertools.product(LEARNING_RATES, BATCH_SIZES))

    data = Utils.dataTools.AuthorshipAll('poe', ratio_train, ratio_valid, dataPath)

    results = {}

    if doPrint:
        print('TOTAL COMBINATIONS FOR GRID SEARCH: {}'.format(len(combinations)))

    # loop through each author
    for author_name in data.authorData.keys():
        if doPrint:
            print('AUTHOR: {}'.format(author_name))

        data = Utils.dataTools.AuthorshipAll(author_name, ratio_train, ratio_valid, dataPath)

        data.astype(torch.double)
        data.to(device)

        results[author_name] = {}

        # Loop through each combination
        for combination in combinations:
            if doPrint:
                print('COMBINATION: {}'.format(combination))

            results[author_name][str(combination)] = []
            # For each author, do 10 splits
            for split in range(N_DATA_SPLITS):

                # initialize the split, but no need for the first one.
                if split is not 0:
                    data.get_split(author_name, ratio_train, ratio_valid)

                xTrain, yTrain = data.getSamples('train')

                # x_train_tensor = torch.from_numpy(xTrain).float()
                # y_train_tensor = torch.from_numpy(yTrain).long()
                x_train_tensor = xTrain.unsqueeze(1)
                y_train_tensor = yTrain.type(torch.double)

                train_data = CustomDataset(x_train_tensor, y_train_tensor)
                train_loader = DataLoader(dataset=train_data, batch_size=combination[1], shuffle=True)

                nNodes = data.selectedAuthor['all']['wordFreq'].shape[1]

                net = do_training(combination[0], nNodes, train_loader)
                acc = validate(net, data)

                results[author_name][str(combination)].append(acc)

        # with open('2_feedforward_results_BCLoss.txt', 'w+') as outfile:
        #     json.dump(results, outfile)

    return results


training_results = main_loop()
# %%##################################################################
# # VALIDATE
#

data = Utils.dataTools.Authorship('poe', ratio_train, ratio_valid, dataPath)
data.astype(torch.double)
data.to(device)

# %%##################################################################

xTrain, yTrain = data.getSamples('train')

# x_train_tensor = torch.from_numpy(xTrain).float()
# y_train_tensor = torch.from_numpy(yTrain).long()
x_train_tensor = xTrain.unsqueeze(1)
y_train_tensor = yTrain.type(torch.double)

train_data = CustomDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

nNodes = data.selectedAuthor['all']['wordFreq'].shape[1]

net = do_training(0.001, nNodes, train_loader)
acc = validate(net, data)
