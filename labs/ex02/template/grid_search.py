# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
from costs import *


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def grid_search(y, tx, w0, w1, loss_type=Loss.MSE):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss for each combination of w0 and w1.
    # ***************************************************
    rows, cols = np.indices((w0.shape[0], w1.shape[0]))
    params = np.column_stack((w0[rows.ravel()], w1[cols.ravel()]))
    params = params.reshape((len(w0), len(w1), 2))

#     losses = np.column_stack((w0[rows.ravel()], w1[cols.ravel()]))
#     losses = losses.reshape((len(w0), len(w1), 2))
    
#     for i in range(len(w0)):
#         for j in range(len(w1)):
#             losses[i, j] = compute_loss(y, tx, params[i, j], type)
    for index in np.ndindex(len(w0), len(w1)):
        losses[index] = compute_loss(y, tx, params[index], loss_type)
    
    return losses

