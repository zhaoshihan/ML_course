# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    N = len(x)
    
    ans = np.ones((degree + 1, N))
    for d in range(1, degree + 1):
        arr = ans[d - 1] * x
        ans[d] = arr
    
    return ans.T
