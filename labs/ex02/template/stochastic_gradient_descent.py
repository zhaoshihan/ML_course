# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
from gradient_descent import *
from helpers import batch_iter

def compute_stoch_gradient(y, tx, w, loss_type=Loss.MSE):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    return compute_gradient(y, tx, w, loss_type)
#     raise NotImplementedError


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, loss_type=Loss.MSE):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    ws = [initial_w]
    losses = [compute_loss(y, tx, initial_w, loss_type)]
    w = initial_w
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Ln(w)
            gradient_n = compute_stoch_gradient(minibatch_y, minibatch_tx, w, loss_type)
       
            w = w - gamma * gradient_n
            loss = compute_loss(y, tx, w, loss_type)
            
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    return losses, ws