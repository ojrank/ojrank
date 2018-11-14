import os
import numpy as np

crossEnt = lambda p, p_opt: np.sum(np.multiply(-p_opt, np.log(p)), axis=1)

def cross_entropy_loss(puvhat, puv):
    p_opt = np.concatenate((puvhat[:,None], (1-puvhat)[:,None]), axis=1)
    p = np.concatenate((puv[:,None], (1-puv)[:,None]), axis=1)
    #loss = np.mean(crossEnt(p, p_opt))
    loss = np.sum(crossEnt(p, p_opt))
    return loss

def cross_entropy_loss_l2(puvhat, puv, w_curr, w_orig, reg_constant):
    ce_loss = cross_entropy_loss(puvhat, puv)
    #reg_term = np.sum((w_curr - w_orig) ** 2)/(puv.shape[0])
    #reg_term2 = np.sum((w_curr - w_orig) ** 2)/(w_orig.shape[0])
    reg_term3 = np.sum((w_curr - w_orig) ** 2)
    loss = ce_loss + reg_constant * reg_term3
    return loss

def cross_entropy_loss_pairs(known_puvhat, sampled_puvhat, puv_known, puv_sampled):
    Cs=1
    Ck=1
    known_loss = cross_entropy_loss(known_puvhat, puv_known)
    sampled_loss = cross_entropy_loss(sampled_puvhat, puv_sampled)
    
    loss = Ck*known_loss + Cs*sampled_loss
    return loss

if __name__ == '__main__':
    puvhat = np.array([0.62, 0.62, 0.61, 0.60, 0.59])
    puv =  np.array([0.52, 0.51, 0.53, 0.50, 0.49])
    
    w_curr =  np.random.normal(size=(30,))
    w_orig = np.ones(shape=[30,])
    w_curr = w_curr/np.sqrt(w_curr.dot(w_curr))
    w_orig = w_orig/np.sqrt(w_orig.dot(w_orig))
    print cross_entropy_loss_l2(puvhat, puv, w_curr, w_orig)
    print "=========================="
    puvhat = np.array([0.62, 0.62, 0.61, 0.60, 0.59])
    puv =  np.array([0.52, 0.51, 0.53, 0.50, 0.60])
    w_curr[0] = 0
    w_orig[0] = 1
    w_curr = w_curr/np.sqrt(w_curr.dot(w_curr))
    w_orig = w_orig/np.sqrt(w_orig.dot(w_orig))
    print cross_entropy_loss_l2(puvhat, puv, w_curr, w_orig)
    print "=========================="