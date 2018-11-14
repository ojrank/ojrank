from SHIP.approach_wnorm2 import *
from SHIP.utils import *

import os, sys
import numpy as np
import cPickle as pickle

def make_filename(params):
    filename = params['ds_name']
    filename+="_"+str(params['cd']['coords'])
    
    return filename
    
def make_params(X, y, meta_info, ds_name):
    num_points = X.shape[0]
    num_anomalies = np.sum(y)
    params = {}
    params['weight_initialize']='uniform'   #Keep it as uniform. But it doesn't really matter.
    params['budget']=meta_info[ds_name][1]
    #============================
    params['IForest']={}
    params['cd']={}
    params['anchor_points']={}
    params['learning']={}
    #============================
    #==========IForest Params====================
    params['IForest']['n_trees']=100    #Number of Trees
    params['IForest']['forest_n_trees'] = 100
    params['IForest']['forest_n_samples']=meta_info[ds_name][1] #Number of samples per tree
    params['IForest']['forest_score_type'] = "InvScoring"   #scoring type
    params['IForest']['forest_add_leaf_nodes_only'] = True
    params['IForest']['forest_max_depth'] = 10
    params['IForest']['ensemble_score'] = "ENSEMBLE_SCORE_LINEAR"
    params['IForest']['detector_type'] = "AAD_IFOREST"
    params['IForest']['n_jobs'] = 1
    #============================================
    #========Active Discovery Params=============
    params['learning']['learning_rate'] = float(sys.argv[6])
    params['learning']['num_iters'] = 1000
    params['learning']['reg_constant']=float(sys.argv[5])
    #============================================
    params['anchor_points']['K'] = int(sys.argv[7])
    params['anchor_points']['epsilon']=0.1
    #============================================
    params['cd']['coords'] = sys.argv[3]
    #===========================================
    params['normalize']=True
    params['ds_name'] = ds_name

    print "Running for:"+str(params['anchor_points']['K'])+" and Learning Rate="+str(params['learning']['learning_rate'])
    return params

def read_meta_info(meta_file):
    meta_info = {}
    f_meta = open(meta_file,"r")
    line=f_meta.readline()
    line = f_meta.readline()
    while line:
        line = line.replace("\n","")
        split = line.split(",")
        print split
        meta_info[split[0]] = [int(split[3]),int(split[4]), int(float(split[2]))]
        line = f_meta.readline()
    f_meta.close()
    
    return meta_info
    
def run(ds_name):
    meta_file = "PATH_TO_META_INFO_FILE"
    BASE_DIR = "PATH_TO_SCORES_DIR"

    cd = sys.argv[2]
    learning_rate = float(sys.argv[3])
    K = int(sys.argv[4])

    print "K="+str(K)+" and Learning Rate="+str(learning_rate)
    OUT_DIR = "PATH_TO_RESULTS_DIR"

    if(not os.path.exists(OUT_DIR)):
        os.mkdir(OUT_DIR)
    
    meta_info = read_meta_info(meta_file)
    ds_dir = os.path.join(OUT_DIR, ds_name)
    
    if(not os.path.exists(ds_dir)):
        os.mkdir(ds_dir)
        
    num_sampled_arr= []
    for i in range(10):    #Number of different trees
        print "\t\tRun:"+str(i) +" and K="+str(K)
        score_file = os.path.join(BASE_DIR, ds_name, ds_name+"_"+str(i)+"_Sample_256"+"TREE.csv")
        out_file = os.path.join(ds_dir, ds_name+"_"+str(i)+"_Sample_256TREE.csv")
        X,y = load_scores_file(score_file)
        params = make_params(X,y,meta_info,ds_name)
        print "Budget="+str(params['budget'])
        out_file_pkl = os.path.join(ds_dir,make_filename(params)+"_PosNeg"+str(K)+"_"+str(learning_rate)+"_Sample_"+str(i)+".pkl")

        arr_time = []
        queried, queried_labels, stop_arrs, time_per_update = approach_PosNeg_Known_TIME(X, y, params)
        out_file_pkl = os.path.join(ds_dir, make_filename(params) +
                                    "_PosNegKnownSGDSampl_"+str(i) + ".pkl")
        out_file_pkl_time = os.path.join(ds_dir, make_filename(params) +
                                         "_PosNegKnownSGDSampl_"+ str(i) + "_TIME.pkl")
        arr_time.append(time_per_update)

        np.savetxt(out_file, np.cumsum(queried_labels))
        pickle.dump([queried,queried_labels,np.cumsum(queried_labels),stop_arrs], open(out_file_pkl,"w"))
        pickle.dump(arr_time, open(out_file_pkl_time,"w"))

        arr = np.cumsum(queried_labels)
        budget = meta_info[ds_name][2]
        b = np.sum(arr[min(len(arr)-1,int(budget))])

    np.savetxt(out_file+"_NUMSAMPLED", num_sampled_arr)

if __name__ == '__main__':
    ds_name = sys.argv[1]
    run(ds_name)