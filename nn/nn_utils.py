import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

def save_model(net, output_file='model.state'):
    """
    Saves a pytorch model
    :param net:
    :param output_file:
    :return:
    """
    torch.save(net.state_dict(), output_file)


def load_model(net, input_file='model.state'):
    """
    Loads a pytorch model
    :param net:
    :param input_file:
    :return:
    """
    state_dict = torch.load(input_file)
    net.load_state_dict(state_dict)


def train_model(net, optimizer, criterion, train_loader, epochs=10, layer=-1):
    """
    Trains a classification model
    :param net:
    :param optimizer:
    :param criterion:
    :param train_loader:
    :param epochs:
    :return:
    """
    for epoch in range(epochs):
        net.train()

        train_loss, correct, total = 0.0, 0.0, 0.0
        for (inputs, targets) in tqdm(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            if layer == -1:
                outputs = net(inputs)
            else:
                outputs = net.get_features(inputs, layers=[layer])[0]

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Calculate statistics
            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        print("\nLoss, acc = ", train_loss, correct / total)


def extract_features(net, test_loader, layer=-1):
    """
    Extracts features from a neural network
    :param net: a network that must implement net.get_features()
    :param test_loader:
    :return:
    """
    net.eval()

    features = []
    labels = []
    with torch.no_grad():
        for (inputs, targets) in tqdm(test_loader):
            inputs = inputs.cuda()
            labels.append(targets.numpy())

            inputs = Variable(inputs)

            outputs = net.get_features(inputs, layers=[layer])[0]
            outputs = outputs.cpu()

            features.append(outputs.data.numpy())

    return np.concatenate(features), np.concatenate(labels).reshape((-1,))


def extract_features_raw(test_loader):
    """
    Extracts features from a neural network
    :param net: a network that must implement net.get_features()
    :param test_loader:
    :return:
    """

    features = []
    labels = []
    with torch.no_grad():
        for (inputs, targets) in tqdm(test_loader):
            labels.append(targets.numpy())
            features.append(inputs.data.numpy())

    return np.concatenate(features), np.concatenate(labels).reshape((-1,))
