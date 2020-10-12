# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    N = len(y)
    dimension = tx.shape[1]
    lambda_prime = 2 * N * lambda_
    
    w = np.linalg.inv(np.dot(tx.T, tx) + lambda_prime * np.eye(dimension)).dot(tx.T).dot(y)
    return w
