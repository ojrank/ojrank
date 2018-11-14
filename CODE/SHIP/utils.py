'''
Contains functions that are common.

- Sigmoid
- Tau
- NDCG
- DCG
- Dataset Loader
- Make Params
- Get Queried Indexes
'''

import os
import sys
import pandas as pd
import time
import numpy as np
import operator
import pickle
from scipy.special import expit
from scipy.sparse import csr_matrix

def transform_labels(x):
    if x=='nominal':
        return 0
    elif x=='anomaly':
        return 1
    
def exponentiate(x,p=-0.33333):
    '''
        Exponentiates the given value as follows:
        g(x) = (1+p * x) ^ (1/p)
    '''
    return np.power(1+(p*x),float(1)/p)
    
def load_benchmark_dataset(ds_path):
    df = pd.read_csv(ds_path)
    pt_ids = np.array(df['point.id'])
    labels = np.array(df['ground.truth'])
    labels = [0 if labels[i] == 'nominal' else 1 for i in range(len(labels))]
    labels = np.array(labels)
    X = np.array(df.iloc[:,6:len(df.columns)])
    return X, labels

def load_dataset(ds_name, is_predefined=False):
    '''
    Loads the dataset, and if it is predefined, load from
    the stored path, else load from the provided path.
    
    Args:
        ds_filename: Name of the Dataset/Path.
        is_predefined: In set of datasets to consider.

    Returns:
        X: Data Samples
        y: Labels if it is anomalous or not.
    '''
    if(is_predefined):
        data_path = os.path.join(DATA_DIR,'anomaly/%s/fullsamples/%s_1.csv'%(ds_name, ds_name))
    else:
        data_path = ds_name

    data = pd.DataFrame.from_csv(data_path, sep=',', index_col=None)
    X_train = np.zeros(shape=(data.shape[0], data.shape[1]-1))
    for i in range(1,X_train.shape[1]):
        X_train[:, i-1] = data.iloc[:, i]

    labels = data.iloc[:, 0]
    labels = labels.apply(lambda x: transform_labels(x))
    
    return X_train, labels

def load_scores_file(score_file):
    labels=[]
    scores=[]
    f = open(score_file,"r")
    line = f.readline()
    line = f.readline()
    while line:
        temp=[]
        line = line.replace("\n","")
        split = line.split(",")
        labels.append(1 if split[0] == 'anomaly' else 0)
        for i in range(1,len(split)):
            value = float(split[i])
            temp.append(value)
        temp = np.array(temp)
        scores.append(np.array(temp))
        line = f.readline()
    f.close()
    scores = np.array(scores)
    labels = np.array(labels)
    
    return scores, labels

def normalize_weights(w_vec):
    len_w = w_vec.dot(w_vec)
    if np.isnan(len_w):
        # logger.debug("w_new:\n%s" % str(list(w_new)))
        raise ArithmeticError("weight vector contains nan")

    w_vec = w_vec/np.sqrt(len_w)
    return w_vec
    
def sort(x, decreasing=False):
    if decreasing:
        return np.argsort(-x)
    else:
        return np.argsort(x)
    
def normalize_and_computesig(vec):
    vec = normalize(vec)
    return expit(vec)

def compute_sig(vec):
    '''
        Normalizes and computes Sigmoid of the given vector.
    '''
    #vec = np.normalize(vec)
    #sig = 1./(1+np.exp(-vec))
    #return sig
    return expit(vec)

def tau_with_sampling(arr1,arr2,sampling_fraction):
    '''
        Computes Kendall Tau distance between 2 rank_lists
    '''
    sampling_length = int(float(sampling_fraction) * len(arr1))
    x = np.random.choice(arr1, sampling_length, replace=False)
    c = np.zeros_like(x)
    d = np.zeros_like(x)
    
    for i,elem in enumerate(x):
        c[i] = np.where(arr1 == elem)[0]
        d[i] = np.where(arr2 == elem)[0]
        
    distance = 0
    
    for i in range(len(c)):
        for j in range(i+1,len(c)):
            if(np.sign(c[i]-c[j])!= np.sign(d[i]-d[j])):
                distance+=1
                
    distance = (2*float(distance))/(sampling_length*(sampling_length-1))
    return float(distance)

def compute_ndcg(orig_rl, new_rl, k):
    dcg = compute_dcg(orig_rl, new_rl, k) 
    best_dcg = compute_dcg(orig_rl, orig_rl, k)
    return float(dcg)/best_dcg
   
def compute_dcg(orig_rl, new_rl, k):
    dcg = 0
    max_comp = 0
    for i in range(k):
        item = new_rl[i]
        orig_index = list(orig_rl).index(item)
        component  = (1/(float(orig_index)+1))/np.log2(2+i)
        dcg += component
    return dcg

def get_queried_indexes(scores, labels, budget):
    queried = np.argsort(-scores)[0:budget]
    num_seen = np.cumsum(labels[queried[np.arange(budget)]])
    return num_seen, queried

def norm1_normalize(scores):
    scores = scores/(scores.dot(scores))
    return scores
    
def normalize(scores):
    scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores))
    return scores