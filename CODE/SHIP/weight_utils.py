import os
import sys
import numpy as np


def normalize_weights(w_vec):
    len_w = w_vec.dot(w_vec)
    if np.isnan(len_w):
        # logger.debug("w_new:\n%s" % str(list(w_new)))
        raise ArithmeticError("weight vector contains nan")

    w_vec = w_vec/np.sqrt(len_w)
    return w_vec
    
def init_weights(dims, method):
    method = "uniform"
    if(method == "uniform"):
        w_unif = np.ones(dims, dtype=float)
        return normalize_weights(w_unif)