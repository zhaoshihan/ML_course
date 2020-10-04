# -*- coding: utf-8 -*-
"""Gradient Descent"""
from costs import *


def compute_gradient(y, tx, w, type=Loss.MSE):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and loss
    # ***************************************************
    N = len(y)
    e = y - tx.dot(w)
    
    if type == Loss.MSE:
        gradient = (-1 / N) * np.dot(tx.T, e)
#         loss = compute_loss(y, tx, w, type=Loss.MSE)
        return gradient
    
    elif type == Loss.MAE:
        subgradient = (-1 / N) * np.dot(tx.T, np.sign(e))
#         loss = compute_loss(y, tx, w, type=Loss.MAE)
        return subgradient
    
    else:
        raise NotImplementedError
        

def gradient_descent(y, tx, initial_w, max_iters, gamma, loss_type=Loss.MSE):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y, tx, initial_w)]
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        gradient = compute_gradient(y, tx, w, loss_type)
#         raise NotImplementedError

        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        
        w = w - gamma * gradient
        loss = compute_loss(y, tx, w, loss_type)
#         raise NotImplementedError

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws