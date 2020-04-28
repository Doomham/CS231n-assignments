import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  train_num = X.shape[0]
  score = X.dot(W)
  score_exp = np.exp(score)
  score_sum_rows = np.sum(score_exp, axis=1)
  for i in xrange(train_num):
    loss -= np.log(score_exp[i, y[i]] / score_sum_rows[i])
    dW += np.dot(np.reshape(X[i], (-1, 1)), np.reshape(score_exp[i, :] / score_sum_rows[i], (1, -1)))
    dW[:, y[i]] -= X[i].T
  dW = dW / train_num
  dW += 2 * reg * W
  loss /= train_num
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  train_num = X.shape[0]
  score = X.dot(W)
  score_exp = np.exp(score)
  expsum = np.sum(score_exp, axis=1)
  exp_correct = score_exp[xrange(train_num), y[xrange(train_num)]]
  Li = -np.log(exp_correct / expsum)
  loss += np.sum(Li) / train_num + reg * np.sum(W * W)
  diag_sum = np.diag(1 / expsum)
  score_exp = np.dot(diag_sum, score_exp)
  score_exp[xrange(train_num),y[xrange(train_num)]] -= 1
  dW = np.dot(X.T, score_exp) / train_num + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

