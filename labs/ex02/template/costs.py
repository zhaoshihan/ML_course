# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np
from enum import Enum, auto


class Loss(Enum):
    MSE = auto()
    MAE = auto()

    
def compute_loss(y, tx, w, loss_type=Loss.MSE):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************
    N = len(y)
    e = y - tx.dot(w)
    
#     import pdb; pdb.set_trace()
    if loss_type is Loss.MSE:
        return (1 / (2 * N)) * np.dot(e.T, e)
    elif loss_type is Loss.MAE:
        return (1 / N ) * np.sum(np.abs(e))
    else:
        raise NotImplementedError