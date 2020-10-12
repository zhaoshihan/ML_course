# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w = np.linalg.inv(np.dot(tx.T, tx)).dot(tx.T).dot(y)
    
    e = y - tx.dot(w)
    mse = (1 / 2) * np.mean(e * e)
    
    return mse, w
