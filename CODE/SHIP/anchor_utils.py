import os
import sys
import numpy as np
import pandas as pd
from utils import *
from weight_utils import *
from scipy.special import expit

def choose_pairwise_anchors(sel_instance, sel_label, queried, queried_labels, sorted_indexes, agg_scores, params):
    pos_known_neg=[]
    neg_known_pos=[]
    pos_sampled_neg=[]
    neg_sampled_pos=[]
    
    ###First choose from anchor
    if(params['fix_anchor'] == True):
        queried = np.array(queried)
        queried_labels = np.array(queried_labels)
        pos_queried = queried[queried_labels == 1]
        neg_queried = queried[queried_labels == 0]
        
        if(sel_label==1):
            #Sampled from known neg
            if(len(neg_queried) > 0):
                pos_known_neg = np.random.choice(neg_queried, min(len(neg_queried), params['K']),replace=False)
        elif(sel_label==0):
            #Sampled from known pos
            if(len(pos_queried)>0):
                neg_known_pos = np.random.choice(pos_queried, min(len(pos_queried), params['K']),replace=False)
                
    ios = 1./(1+agg_scores)
    if(np.max(ios) == np.min(ios)):
        ios = np.zeros(ios.shape)
    else:
        ios = (ios - np.min(ios))/(np.max(ios) - np.min(ios))
    ios = exponentiate(ios, p = -0.99)
    #Choose Remaining
    if(sel_label == 1):
        #Are there any points remaining to sample.
        pts_to_sample = params['K'] -  len(pos_known_neg)
        #Sample these points from probs1
        if(pts_to_sample>0):
            probs = np.copy(ios)
            probs[sel_instance] = 0
            probs[queried] = 0
            probs = probs/np.sum(probs)
            
            pos_sampled_neg = np.random.choice(range(agg_scores.shape[0]), p = probs, size=pts_to_sample, replace=False)
            
    elif(sel_label == 0):
        pts_to_sample = params['K'] - len(neg_known_pos)
        if(pts_to_sample > 0):
            probs2 = 1./(ios)
            probs2[sel_instance] = 0
            probs2[queried] = 0
            probs2 = probs2/np.sum(probs2)
            
            neg_sampled_pos = np.random.choice(range(agg_scores.shape[0]), p = probs2, size=pts_to_sample, replace=False)
            
    if(sel_label==1):
        return pos_known_neg, pos_sampled_neg
    else:
        return neg_known_pos, neg_sampled_pos
    
def choose_anchor_points(top_instance, queried, sort_indexes, agg_scores, params):
    ios = 1./(1+agg_scores)
    if np.max(ios) ==  np.min(ios):
        ios = np.zeros(ios.shape)
    else:
        ios = (ios - np.min(ios))/(np.max(ios) - np.min(ios))
    ios = exponentiate(ios, p =-0.99)
    probs = np.copy(ios)
    probs[top_instance] = 0
    probs = probs/np.sum(probs)
    single_arr = np.array([range(agg_scores.shape[0]), probs])

    try:
        indr = np.random.choice(range(agg_scores.shape[0]), p = probs, size = params['K'], replace=False)
    except ValueError, e:
        print "ERROR"
        sys.exit(0)
        
    if(params['fix_anchor'] == True):
        select_from_queried = []
        if(len(queried)>0):
            select_from_queried = np.random.choice(queried, min(len(queried), params['K']), replace=False)
            
        for j in range(len(select_from_queried)):
            indr[j] = select_from_queried[j]
    
    return indr

def choose_negonly_sampling_anchors(sel_instance, sel_label, queried, queried_labels, sorted_indexes, agg_scores, params):
    ios = 1. / (1 + agg_scores)
    if (np.max(ios) == np.min(ios)):
        ios = np.zeros(ios.shape)
    else:
        ios = (ios - np.min(ios)) / (np.max(ios) - np.min(ios))
    ios = exponentiate(ios, p=-0.99)

    if (sel_label == 0):
        pts_to_sample = params['K']
        if (pts_to_sample > 0):
            probs2 = 1. / (ios)
            probs2[sel_instance] = 0
            probs2[queried] = 0
            probs2[int(0.5 * len(ios)):len(ios)] = 0
            probs2 = probs2 / np.sum(probs2)

            neg_sampled_pos = np.random.choice(range(agg_scores.shape[0]), p=probs2, size=pts_to_sample, replace=False)

    neg_known_pos = np.empty(0,)
    return neg_sampled_pos, neg_known_pos

def choose_posneg_anchors(sel_instance, sel_label, queried, queried_labels, sorted_indexes, agg_scores, params):
    ios = 1. / (1 + agg_scores)
    if np.max(ios) == np.min(ios):
        ios = np.zeros(ios.shape)
    else:
        ios = (ios - np.min(ios)) / (np.max(ios) - np.min(ios))
    ios = exponentiate(ios, p=-0.99)

    if (sel_label == 0):
        pts_to_sample = params['K']
        if (pts_to_sample > 0):
            probs2 = 1. / (ios)
            probs2[sel_instance] = 0
            probs2[queried] = 0
            probs2[int(0.5 * len(ios)):len(ios)] = 0
            probs2 = probs2 / np.sum(probs2)

            neg_sampled_pos = np.random.choice(range(agg_scores.shape[0]), p=probs2, size=pts_to_sample, replace=False)
            neg_known_pos = np.empty(0, )

            return neg_sampled_pos, neg_known_pos

    if (sel_label == 1):
        pts_to_sample = params['K']
        if (pts_to_sample > 0):
            probs2 = ios
            probs2[sel_instance] = 0
            probs2[queried] = 0
            probs2[0:int(0.5 * len(ios))] = 0
            probs2 = probs2 / np.sum(probs2)

            pos_sampled_neg = np.random.choice(range(agg_scores.shape[0]), p = probs2, size = pts_to_sample, replace=False)
            pos_known_neg = np.empty(0,)

            return pos_sampled_neg, pos_known_neg

def choose_posneg_anchors_knowndsample(sel_instance, sel_label, queried, queried_labels,
                                       sorted_indexes, agg_scores, params):
    ios = 1. / (1 + agg_scores)
    if np.max(ios) == np.min(ios):
        ios = np.zeros(ios.shape)
    else:
        ios = (ios - np.min(ios)) / (np.max(ios) - np.min(ios))
    #ios = exponentiate(ios, p=-0.99)

    if (sel_label == 0):
        pts_to_sample = params['K']
        if (pts_to_sample > 0):
            probs2 = 1. / (ios)
            probs2[sel_instance] = 0
            probs2[queried] = 0
            probs2 = probs2 / np.sum(probs2)

            neg_sampled_pos = np.random.choice(range(agg_scores.shape[0]), p=probs2, size=pts_to_sample, replace=False)
            neg_known_pos = np.empty(0, )

            return neg_sampled_pos, neg_known_pos

    if (sel_label == 1):
        pts_to_sample = params['K']
        if (pts_to_sample > 0):
            probs2 = ios
            probs2[sel_instance] = 0
            probs2[queried] = 0
            probs2 = probs2 / np.sum(probs2)

            pos_sampled_neg = np.random.choice(range(agg_scores.shape[0]), p=probs2, size=pts_to_sample, replace=False)
            pos_known_neg = np.empty(0, )

            return pos_sampled_neg, pos_known_neg

def set_puvhat_posneg(sel_label, sel_instance, agg_scores, known_anchors, sampled_anchors, params):
    diff = expit(agg_scores[sel_instance] - agg_scores[sampled_anchors])

    if sel_label == 0:
        puvhat_sampled = diff - params['epsilon'] * diff
    if sel_label == 1:
        puvhat_sampled = diff + params['epsilon'] * diff

    puvhat_known = np.zeros(len(known_anchors),)

    return puvhat_known, puvhat_sampled

def set_puvhat_noK_small(sel_label, sel_instance, agg_scores, queried, queried_labels, params):
    queried = np.array(queried)
    queried_labels = np.array(queried_labels)

    pos_queried = queried[queried_labels == 1]
    known_anchors = pos_queried

    pairs = []
    n_pairs = pos_queried.shape[0]

    for entry in pos_queried:
        pairs.append((sel_instance, entry))

    assert n_pairs == len(pairs)

    mn = min(agg_scores)
    mx = max(agg_scores)
    diff = expit(mn - mx)

    puvhat = np.ones(len(known_anchors),)*diff
    #diff = expit(agg_scores[sel_instance] - agg_scores[pos_queried])

    return pos_queried, pairs, puvhat

def set_puvhat(sel_label, top_instance, agg_score, anchors, queried, queried_labels, params):
    mn = np.min(agg_score)
    mx = np.max(agg_score)

    if(sel_label == 1):
        puvhat = expit(agg_score[top_instance] - agg_score[anchors] + params['epsilon'])
    else:
        puvhat = compute_sig(agg_score[anchors] - mn)
        
    if(params['fix_anchor']==True):
        for i in range(len(anchors)):
            a_point = anchors[i]
            if(a_point in queried):
                label_a_point = queried_labels[queried==int(a_point)]
                if(sel_label == 1):
                    if(label_a_point == 1):
                        puvhat[i] = 0.5
                    elif(label_a_point == 0):
                        puvhat[i] = params['pos_weight']
                elif(sel_label == 0):
                    if(label_a_point == 1):
                        puvhat[i] = params['neg_weight']
                    elif(label_a_point == 0):
                        puvhat[i] = 0.5
            
    return puvhat
        
def set_puvhat_pairwise(sel_label, sel_instance, agg_scores, known_anchors, sampled_anchors, params):
    if sel_label == 1:
        puvhat_known = np.ones(len(known_anchors),)
        puvhat_sampled = expit(agg_scores[sel_instance] - agg_scores[sampled_anchors]) + params['epsilon']
    elif sel_label == 0:
        puvhat_known =  np.zeros(len(known_anchors),)
        puvhat_sampled = expit(agg_scores[sel_instance] - agg_scores[sampled_anchors]) - params['epsilon']

    return puvhat_known, puvhat_sampled

def set_puvhat_negonly(sel_label, sel_instance, agg_scores, known_anchors, sampled_anchors, params):
    if sel_label == 0:
        mn = min(agg_scores)
        mx = max(agg_scores)

        diff = expit(mn-mx)
        puvhat_known =  np.ones(len(known_anchors),)*diff

        diff = expit(agg_scores[sel_instance] - agg_scores[sampled_anchors])
        puvhat_sampled = diff - params['epsilon'] * diff

    return puvhat_known, puvhat_sampled

def set_puvhat_posneg_known_NoSample(sel_label, sel_instance, agg_scores, known_anchors, sampled_anchors, params):
    mn = min(agg_scores)
    mx = max(agg_scores)

    if sel_label == 0:
        diff = expit(mn - mx)
        puvhat_known = np.ones(len(known_anchors), ) * diff
        puvhat_sampled = np.ones(len(sampled_anchors),) * diff
        return puvhat_known, puvhat_sampled

    if sel_label == 1:
        diff = expit(mx - mn)
        puvhat_known = np.ones(len(known_anchors), ) * diff

        diff = expit(agg_scores[sel_instance] - agg_scores[sampled_anchors])
        puvhat_sampled = diff + params['epsilon'] * diff

        return puvhat_known, puvhat_sampled

def set_puvhat_posneg_known(sel_label, sel_instance, agg_scores, known_anchors,
                                                         sampled_anchors, params):
    mn = min(agg_scores)
    mx = max(agg_scores)

    if sel_label == 0:

        diff = expit(mn - mx)
        puvhat_known = np.ones(len(known_anchors),) * diff

        diff = expit(agg_scores[sel_instance] - agg_scores[sampled_anchors])
        puvhat_sampled = diff - params['epsilon'] * diff

        return puvhat_known, puvhat_sampled

    if sel_label == 1:
        diff = expit(mx - mn)
        puvhat_known = np.ones(len(known_anchors),) * diff

        diff = expit(agg_scores[sel_instance] - agg_scores[sampled_anchors])
        puvhat_sampled = diff + params['epsilon'] * diff

        return puvhat_known, puvhat_sampled

def set_puvhat_negonly_sampling(sel_label, sel_instance, agg_scores, known_anchors, sampled_anchors, params):
    if sel_label == 0:
        diff = expit(agg_scores[sel_instance] - agg_scores[sampled_anchors])
        puvhat_sampled = diff - params['epsilon'] * diff
        puvhat_known = np.empty(len(known_anchors),)

    return puvhat_known, puvhat_sampled

'''
#### MORE ELABORATE STUFF BELOW
#### ALL FUNCTIONS THAT USED TO GENERATE VARIANTS ARE USED
'''

def get_sampling_prob(ios, sel_instance, sel_label, queried):
    # ios - Inverted overall score. Higher the point is on the list, lower is it's ios

    if(sel_label == 0):
        # We need to sample from top of the list, hence reverse the score.
        probs = 1./ios
        probs[sel_instance] = 0
        probs[queried] = 0
        # Do not choose from bottom half of the list.
        probs[int(0.5 * len(ios)):len(ios)] = 0
        probs = probs / np.sum(probs)

    if(sel_label == 1):
        probs = ios
        probs[sel_instance] = 0
        probs[queried] = 0
        # Do not choose from top half of the list.
        probs[0:int(0.5 * len(ios))] = 0
        probs = probs / np.sum(probs)

    return probs

def get_sampling_prob_NEW(ios, sel_instance, sel_label, queried):
    if (sel_label == 0):
        # We need to sample from top of the list, hence reverse the score.
        probs = 1. / ios
        probs[sel_instance] = 0
        probs[queried] = 0
        # Do not choose from bottom half of the list.
        probs[int(0.5 * len(ios)):len(ios)] = 0
        probs = probs / np.sum(probs)

    if (sel_label == 1):
        probs = ios
        probs[sel_instance] = 0
        probs[queried] = 0
        # Do not choose from top half of the list.
        probs[0:int(0.5 * len(ios))] = 0
        probs = probs / np.sum(probs)

    return probs

def get_random_sampling_prob(ios, sel_instance, sel_label, queried):
    probs = np.ones([ios.shape[0],])
    probs[sel_instance] = 0
    probs = probs/ np.sum(probs)

    return probs

def get_dumber_sampling_prob(ios, sel_instance, sel_label, queried):
    # ios - Inverted overall score. Higher the point is on the list, lower is it's ios

    if(sel_label == 0):
        # We need to sample from top of the list, hence reverse the score.
        probs = 1./(1+ios)
        probs[sel_instance] = 0
        probs = probs / np.sum(probs)

    if(sel_label == 1):
        probs = ios
        probs[sel_instance] = 0
        probs = probs / np.sum(probs)

    return probs


# Pos-Neg Known
def choose_posneg_anchors_known(sel_instance, sel_label, queried, queried_labels,
                                                               sorted_indexes, agg_scores, params):
    pos_known_neg = []
    neg_known_pos = []
    pos_sampled_neg = []
    neg_sampled_pos = []

    queried = np.array(queried)
    queried_labels = np.array(queried_labels)
    pos_queried = queried[queried_labels == 1]
    neg_queried = queried[queried_labels == 0]

    ios = 1. / (1 + agg_scores)
    if (np.max(ios) == np.min(ios)):
        ios = np.zeros(ios.shape)
    else:
        ios = (ios - np.min(ios)) / (np.max(ios) - np.min(ios))
    ios = exponentiate(ios, p=-0.99)

    if (sel_label == 0):
        for pt in pos_queried:
            neg_known_pos.append(int(pt))
        neg_known_pos = np.array(neg_known_pos)

        # Choose Remaining
        pts_to_sample = params['K'] - len(neg_known_pos)
        if (pts_to_sample > 0):
            probs = get_sampling_prob(ios, sel_instance, sel_label, queried)
            neg_sampled_pos = np.random.choice(range(agg_scores.shape[0]), p=probs, size=pts_to_sample, replace=False)

        return neg_known_pos, neg_sampled_pos


    if sel_label == 1:
        for pt in neg_queried:
            pos_known_neg.append(int(pt))
        pos_known_neg = np.array(pos_known_neg)

        pts_to_sample = params['K'] - len(pos_known_neg)
        if (pts_to_sample > 0):
            probs = get_sampling_prob(ios, sel_instance, sel_label, queried)
            pos_sampled_neg = np.random.choice(range(agg_scores.shape[0]), p=probs, size=pts_to_sample, replace=False)

        return pos_known_neg, pos_sampled_neg


def choose_posneg_anchors_known_NoSample2(sel_instance, sel_label, queried, queried_labels,
                                                               sorted_indexes, agg_scores, params):
    pos_known_neg = []
    neg_known_pos = []
    pos_sampled_neg = []
    neg_sampled_pos = []

    queried = np.array(queried)
    queried_labels = np.array(queried_labels)
    pos_queried = queried[queried_labels == 1]
    neg_queried = queried[queried_labels == 0]

    ios = 1. / (1 + agg_scores)
    if (np.max(ios) == np.min(ios)):
        ios = np.zeros(ios.shape)
    else:
        ios = (ios - np.min(ios)) / (np.max(ios) - np.min(ios))
    ios = exponentiate(ios, p=-0.99)

    if (sel_label == 0):
        for pt in pos_queried:
            neg_known_pos.append(int(pt))
        neg_known_pos = np.array(neg_known_pos)

        # Choose Remaining
        pts_to_sample = params['K'] - len(neg_known_pos)
        if (pts_to_sample > 0):
            probs = get_sampling_prob(ios, sel_instance, sel_label, queried)
            #neg_sampled_pos = np.random.choice(range(agg_scores.shape[0]), p=probs, size=pts_to_sample, replace=False)
            neg_sampled_pos = -1 * np.ones((pts_to_sample,))

        return neg_known_pos, neg_sampled_pos


    if sel_label == 1:
        for pt in neg_queried:
            pos_known_neg.append(int(pt))
        pos_known_neg = np.array(pos_known_neg)

        pts_to_sample = params['K'] - len(pos_known_neg)
        if (pts_to_sample > 0):
            probs = get_sampling_prob(ios, sel_instance, sel_label, queried)
            pos_sampled_neg = np.random.choice(range(agg_scores.shape[0]), p=probs, size=pts_to_sample, replace=False)

        return pos_known_neg, pos_sampled_neg


def set_puvhat_posneg_known(sel_label, sel_instance, agg_scores, known_anchors,
                                                         sampled_anchors, params):
    mn = min(agg_scores)
    mx = max(agg_scores)

    if sel_label == 0:
        diff = expit(mn - mx)
        puvhat_known = np.ones(len(known_anchors),) * diff

        diff = expit(agg_scores[sel_instance] - agg_scores[sampled_anchors])
        puvhat_sampled = diff - params['epsilon'] * diff

        return puvhat_known, puvhat_sampled

    if sel_label == 1:
        diff = expit(mx - mn)
        puvhat_known = np.ones(len(known_anchors),) * diff

        diff = expit(agg_scores[sel_instance] - agg_scores[sampled_anchors])
        puvhat_sampled = diff + params['epsilon'] * diff

        return puvhat_known, puvhat_sampled


# Pos-Neg Known with Dumber Sampling
def choose_posneg_anchors_known_dsample(sel_instance, sel_label, queried, queried_labels,
                                                                     sorted_indexes, agg_scores, params):
    pos_known_neg = []
    neg_known_pos = []
    pos_sampled_neg = []
    neg_sampled_pos = []

    known = []

    queried = np.array(queried)
    queried_labels = np.array(queried_labels)
    pos_queried = queried[queried_labels == 1]
    neg_queried = queried[queried_labels == 0]

    ios = 1. / (1 + agg_scores)
    if (np.max(ios) == np.min(ios)):
        ios = np.zeros(ios.shape)
    else:
        ios = (ios - np.min(ios)) / (np.max(ios) - np.min(ios))

    if (sel_label == 0):
        #for pt in pos_queried:
        #    neg_known_pos.append(int(pt))
        #neg_known_pos = np.array(neg_known_pos)
        for pt in queried:
            known.append(int(pt))
        # Choose Remaining
        #pts_to_sample = params['K'] - len(neg_known_pos)
        pts_to_sample = params['K'] - len(known)
        if (pts_to_sample > 0):
            probs = get_dumber_sampling_prob(ios, sel_instance, sel_label, queried)
            neg_sampled_pos = np.random.choice(range(agg_scores.shape[0]), p=probs, size=pts_to_sample, replace=False)

        #return neg_known_pos, neg_sampled_pos
        return known, neg_sampled_pos

    if sel_label == 1:
        #for pt in neg_queried:
        #    pos_known_neg.append(int(pt))
        #pos_known_neg = np.array(pos_known_neg)

        for pt in queried:
            known.append(int(pt))

        #pts_to_sample = params['K'] - len(pos_known_neg)
        pts_to_sample = params['K'] - len(known)
        if (pts_to_sample > 0):
            probs = get_dumber_sampling_prob(ios, sel_instance, sel_label, queried)
            pos_sampled_neg = np.random.choice(range(agg_scores.shape[0]), p=probs, size=pts_to_sample, replace=False)

        #return pos_known_neg, pos_sampled_neg
        return known, pos_sampled_neg

def choose_posneg_anchors_known_rsample(sel_instance, sel_label, queried, queried_labels,
                                                                     sorted_indexes, agg_scores, params):
    pos_known_neg = []
    neg_known_pos = []
    pos_sampled_neg = []
    neg_sampled_pos = []

    queried = np.array(queried)
    queried_labels = np.array(queried_labels)
    pos_queried = queried[queried_labels == 1]
    neg_queried = queried[queried_labels == 0]

    ios = 1. / (1 + agg_scores)
    if (np.max(ios) == np.min(ios)):
        ios = np.zeros(ios.shape)
    else:
        ios = (ios - np.min(ios)) / (np.max(ios) - np.min(ios))

    if (sel_label == 0):
        for pt in pos_queried:
            neg_known_pos.append(int(pt))
        neg_known_pos = np.array(neg_known_pos)

        # Choose Remaining
        pts_to_sample = params['K'] - len(neg_known_pos)
        if (pts_to_sample > 0):
            probs = get_random_sampling_prob(ios, sel_instance, sel_label, queried)
            neg_sampled_pos = np.random.choice(range(agg_scores.shape[0]), p=probs, size=pts_to_sample, replace=False)

        return neg_known_pos, neg_sampled_pos

    if sel_label == 1:
        for pt in neg_queried:
            pos_known_neg.append(int(pt))
        pos_known_neg = np.array(pos_known_neg)

        pts_to_sample = params['K'] - len(pos_known_neg)
        if (pts_to_sample > 0):
            probs = get_random_sampling_prob(ios, sel_instance, sel_label, queried)
            pos_sampled_neg = np.random.choice(range(agg_scores.shape[0]), p=probs, size=pts_to_sample, replace=False)

        return pos_known_neg, pos_sampled_neg

# Neg-Only Known with Good Sampling
def choose_negonly_anchors(sel_instance, sel_label, queried, queried_labels, sorted_indexes, agg_scores, params):
    pos_known_neg = []
    neg_known_pos = []
    pos_sampled_neg = []
    neg_sampled_pos = []

    ###First choose from anchor
    if (params['fix_anchor'] == True):
        queried = np.array(queried)
        queried_labels = np.array(queried_labels)
        pos_queried = queried[queried_labels == 1]
        neg_queried = queried[queried_labels == 0]

        neg_known_pos = []
        if (sel_label == 0):
            # Sampled from known pos
            # if(len(pos_queried)>0):
            #    neg_known_pos = np.random.choice(pos_queried, min(len(pos_queried), params['K']),replace=False)
            # Sample ALL Pairs
            for pt in pos_queried:
                neg_known_pos.append(int(pt))
        neg_known_pos = np.array(neg_known_pos)

    ios = 1. / (1 + agg_scores)
    if (np.max(ios) == np.min(ios)):
        ios = np.zeros(ios.shape)
    else:
        ios = (ios - np.min(ios)) / (np.max(ios) - np.min(ios))
    ios = exponentiate(ios, p=-0.99)
    # Choose Remaining
    if (sel_label == 0):
        pts_to_sample = params['K'] - len(neg_known_pos)
        if (pts_to_sample > 0):
            probs2 = 1. / (ios)
            probs2[sel_instance] = 0
            probs2[queried] = 0
            probs2[int(0.5 * len(ios)):len(ios)] = 0
            probs2 = probs2 / np.sum(probs2)

            neg_sampled_pos = np.random.choice(range(agg_scores.shape[0]), p=probs2, size=pts_to_sample, replace=False)

    print "Known Anchors=" + str(neg_known_pos)
    return neg_known_pos, neg_sampled_pos


def test_utils():
    score_file = "../../Results/Results_5/Scores_0"
    X,y = load_scores_file(score_file,"Original")
    w = np.ones(X.shape[1])
    w = w /np.sqrt(w.dot(w))
    scores = X.dot(w)
    sort_indexes = sort(scores)
    
    params={}
    params['epsilon']=0.1
    params['K']=20
    params['fix_anchor'] = True
    params['pos_weight'] = 0.85
    params['neg_weight'] = 0.2
    
    queried = []
    queried = [202, 203, 205, 210, 221, 230, 233, 207, 208, 209]
    queried_labels = [1,1,1,1,1,0,0,1,1,1]
    #aps = choose_anchor_points(201, queried, sort_indexes, scores, params)
    known, sampled = choose_pairwise_anchors(201, 1, queried, queried_labels, sort_indexes, scores, params)
    
    print known
    print sampled
    print np.sum(y[sampled])
    assert len(known)+len(sampled) ==  params['K']
    #assert y[known].all()==1
    #assert y[known].all()==0
    #assert len(known)>0
    
    
    #set_puvhat(1, 201, scores, aps, queried, queried_labels, params)
    puvhat_known, puvhat_sampled = set_puvhat_pairwise(1, 201, scores, known, sampled, params)
    print puvhat_known
    print puvhat_sampled
    
if __name__ == '__main__':
    test_utils()
    
    
    
