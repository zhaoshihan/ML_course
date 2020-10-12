# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    N = len(x)
    indexes = np.random.permutation(N)
    split = int(ratio * N)
    
    x_train = x[indexes[:split]]
    x_test = x[indexes[split:]]
    
    y_train = y[indexes[:split]]
    y_test = y[indexes[split:]]
    
    return x_train, y_train, x_test, y_test
