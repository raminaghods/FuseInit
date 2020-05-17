# -*- coding: utf-8 -*-
"""
Code for paper:

Ghods, Ramina and Lan, Andrew S and Goldstein, Tom and Studer, Christoph, ``MSE-optimal
neural network initialization via layer fusion", 2020 54th Annual Conference on
Information Sciences and Systems (CISS)

fuses multiple layers from of the network one at a time, you can fuse any layer

data is from:
 D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. Reyes-Ortiz, “A public
domain dataset for human activity recognition using smartphones,” in
21th European Symposium on Artificial Neural Networks, Computational
Intelligence and Machine Learning (ESANN). CIACO, Apr. 2013, pp.
437–442.


# HAR classification
# Author: Burak Himmetoglu
# 8/15/2017
"""

import pandas as pd
import numpy as np
import os

def read_data(data_path, split = "train"):
    """ Read data """

    # Fixed params
    n_class = 6
    n_steps = 128

    # Paths
    path_ = os.path.join(data_path, split)
    IS = "Inertial Signals"
    path_signals = os.path.join(data_path, split,IS)

    # Read labels and one-hot encode
    label_path = os.path.join(path_, "y_" + split + ".txt")
    labels = pd.read_csv(label_path, header = None)

    # Read time-series data
    channel_files = os.listdir(path_signals)
    channel_files.sort()
    n_channels = len(channel_files)
    posix = len(split) + 5

    # Initiate array
    list_of_channels = []
    X = np.zeros((len(labels), n_steps, n_channels))
    i_ch = 0
    for fil_ch in channel_files:
        channel_name = fil_ch[:-posix]
        dat_ = pd.read_csv(os.path.join(path_signals,fil_ch), delim_whitespace = True, header = None)
        X[:,:,i_ch] = dat_.as_matrix()

        # Record names
        list_of_channels.append(channel_name)

        # iterate
        i_ch += 1

    # Return
    return X, labels[0].values, list_of_channels

def standardize(train, test):
    """ Standardize data """

    # Standardize train and test
    X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
    X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

    return X_train, X_test

def one_hot(labels, n_class = 6):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels-1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"

    return y

def get_batches(X, y, batch_size = 100):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]
