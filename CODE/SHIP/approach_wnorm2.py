from SHIP.anchor_utils import *
from SHIP.optimization_sgd import *
from SHIP.utils import *
from SHIP.weight_utils import *
from SHIP.coordinate_selector import *

'''
# TIMED of PROPOSED Approach
'''
def approach_PosNeg_Known_TIME(scores, labels, params, sgd_method):
    queried = []
    queried_labels = []
    stop_arrs = []
    n, m = scores.shape
    k = params['anchor_points']['K']
    sampled_pts_arr = []

    # initialize w
    w = init_weights(m, params['weight_initialize'])
    w_init = w.copy()

    # get scores
    agg_scores = scores.dot(w)
    # sort scores
    sorted_indexes = sort(agg_scores, decreasing=True)
    bt = 0
    time_per_update = []
    while (bt < params['budget']):
        # =====Get the top unlabelled instance=====
        start_time = time.time()
        topind = 0
        while (sorted_indexes[topind] in queried):
            topind += 1

        sel_instance = sorted_indexes[topind]
        sel_label = labels[sel_instance]

        queried.append(sel_instance)
        queried_labels.append(sel_label)

        known_anchors, sampled_anchors = choose_posneg_anchors_known(sel_instance, sel_label, queried, queried_labels,
                                                                     sorted_indexes, agg_scores,
                                                                     params['anchor_points'])

        known_puvhat, sampled_puvhat = set_puvhat_posneg_known(sel_label, sel_instance, agg_scores, known_anchors,
                                                               sampled_anchors, params['anchor_points'])

        anchors = np.concatenate((known_anchors, sampled_anchors), axis=0)
        puvhat = np.concatenate((known_puvhat, sampled_puvhat), axis=0)
        coordinate_type = params['cd']['coords']

        if coordinate_type == "UNION":
            coords = choose_diff_coordinates(scores[sel_instance], scores, known_anchors, sampled_anchors, params['cd'])
        else:
            coords = choose_coordinates_negonly(scores, sel_instance, known_anchors, sampled_anchors, params['cd'])

        w, stop_arr = optimize_posneg_sample_Momentum_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                             params['cd'], params['learning'], 0.75)

        end_time = time.time() - start_time
        time_per_update.append(end_time)
        stop_arrs.append([stop_arr, sel_label, labels[sampled_anchors]])
        bt += 1

        if (np.sum(queried_labels) >= np.sum(labels)):
            break

        agg_scores = scores.dot(w)
        sorted_indexes = sort(agg_scores, decreasing=True)
        print "So Far=" + str(np.sum(queried_labels)) + " Out Of " + str(bt)

    return queried, queried_labels, stop_arrs, time_per_update

'''
# NEW_Variant:1 Update on Mistake - Timed
'''
def approach_PosNeg_NegOnly_TIME(scores, labels, params, sgd_method):
    queried = []
    queried_labels = []
    stop_arrs = []
    n, m = scores.shape
    print n, m
    k = params['anchor_points']['K']

    # initialize w
    w = init_weights(m, params['weight_initialize'])
    w_init = w.copy()

    # get scores
    agg_scores = scores.dot(w)
    # sort scores
    sorted_indexes = sort(agg_scores, decreasing=True)
    bt = 0
    time_per_update = []
    while (bt < params['budget']):
        # =====Get the top unlabelled instance=====
        start_time = time.time()
        topind = 0
        while (sorted_indexes[topind] in queried):
            topind += 1

        sel_instance = sorted_indexes[topind]
        sel_label = labels[sel_instance]

        queried.append(sel_instance)
        queried_labels.append(sel_label)

        stop_arr = None
        if sel_label == 0:
            known_anchors, sampled_anchors = choose_negonly_anchors(sel_instance, sel_label, queried, queried_labels,
                                                                     sorted_indexes, agg_scores,
                                                                     params['anchor_points'])


            known_puvhat, sampled_puvhat = set_puvhat_negonly(sel_label, sel_instance, agg_scores, known_anchors,
                                                               sampled_anchors, params['anchor_points'])

            anchors = np.concatenate((known_anchors, sampled_anchors), axis=0)
            puvhat = np.concatenate((known_puvhat, sampled_puvhat), axis=0)

            coordinate_type = params['cd']['coords']

            if coordinate_type == "UNION":
                coords = choose_diff_coordinates(scores[sel_instance], scores, known_anchors, sampled_anchors, params['cd'])
            else:
                coords = choose_coordinates_negonly(scores, sel_instance, known_anchors, sampled_anchors, params['cd'])

            if sgd_method == "SGD":
                w, stop_arr = optimize_posneg_sample_SGD(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                     params['cd'], params['learning'])
            elif sgd_method == "Momentum":
                w, stop_arr = optimize_posneg_sample_Momentum(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                          params['cd'], params['learning'], 0.75)

            elif sgd_method == "NAG":
                w, stop_arr = optimize_posneg_sample_NAG(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                          params['cd'], params['learning'], 0.75)

            elif sgd_method == "SGD_Batch":
                w, stop_arr = optimize_posneg_sample_SGD_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                             params['cd'], params['learning'])

            elif sgd_method == "Momentum_Batch":
                w, stop_arr = optimize_posneg_sample_Momentum_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                             params['cd'], params['learning'], 0.75)
            elif sgd_method == "NAG_Batch":
                w, stop_arr = optimize_posneg_sample_NAG_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                            params['cd'], params['learning'], 0.75)

        end_time = time.time() - start_time
        time_per_update.append(end_time)
        stop_arrs.append(stop_arr)

        bt += 1
        if (np.sum(queried_labels) >= np.sum(labels)):
            break

        agg_scores = scores.dot(w)
        sorted_indexes = sort(agg_scores, decreasing=True)
        print "So Far=" + str(np.sum(queried_labels)) + " Out Of " + str(bt)

    return queried, queried_labels, stop_arrs, time_per_update

'''
# NEW_Variant:2 Dumber Sampling - ALL
'''
def approach_PosNeg_DumberSampling_TIME(scores, labels, params, sgd_method):
    queried = []
    queried_labels = []
    stop_arrs = []
    n, m = scores.shape
    print n, m
    k = params['anchor_points']['K']

    # initialize w
    w = init_weights(m, params['weight_initialize'])
    w_init = w.copy()

    # get scores
    agg_scores = scores.dot(w)
    # sort scores
    sorted_indexes = sort(agg_scores, decreasing=True)
    bt = 0
    time_per_update = []
    while (bt < params['budget']):
        # =====Get the top unlabelled instance=====
        start_time = time.time()
        topind = 0
        while (sorted_indexes[topind] in queried):
            topind += 1

        sel_instance = sorted_indexes[topind]
        sel_label = labels[sel_instance]

        queried.append(sel_instance)
        queried_labels.append(sel_label)

        known_anchors, sampled_anchors = choose_posneg_anchors_known_dsample(sel_instance, sel_label, queried,
                                                                             queried_labels,
                                                                             sorted_indexes, agg_scores,
                                                                             params['anchor_points'])

        known_puvhat, sampled_puvhat = set_puvhat_posneg_known(sel_label, sel_instance, agg_scores, known_anchors,
                                                               sampled_anchors, params['anchor_points'])

        anchors = np.concatenate((known_anchors, sampled_anchors), axis=0)
        puvhat = np.concatenate((known_puvhat, sampled_puvhat), axis=0)
        coordinate_type = params['cd']['coords']

        if coordinate_type == "UNION":
            coords = choose_diff_coordinates(scores[sel_instance], scores, known_anchors, sampled_anchors, params['cd'])
        else:
            coords = choose_coordinates_negonly(scores, sel_instance, known_anchors, sampled_anchors, params['cd'])

        if sgd_method == "SGD":
            w, stop_arr = optimize_posneg_sample_SGD(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                     params['cd'], params['learning'])
        elif sgd_method == "Momentum":
            w, stop_arr = optimize_posneg_sample_Momentum(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                          params['cd'], params['learning'], 0.75)

        elif sgd_method == "NAG":
            w, stop_arr = optimize_posneg_sample_NAG(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                          params['cd'], params['learning'], 0.75)

        elif sgd_method == "SGD_Batch":
            w, stop_arr = optimize_posneg_sample_SGD_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                             params['cd'], params['learning'])

        elif sgd_method == "Momentum_Batch":
            w, stop_arr = optimize_posneg_sample_Momentum_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                             params['cd'], params['learning'], 0.75)
        elif sgd_method == "NAG_Batch":
            w, stop_arr = optimize_posneg_sample_NAG_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                            params['cd'], params['learning'], 0.75)

        end_time = time.time() - start_time
        time_per_update.append(end_time)
        stop_arrs.append(stop_arr)
        bt += 1

        if (np.sum(queried_labels) >= np.sum(labels)):
            break

        agg_scores = scores.dot(w)
        sorted_indexes = sort(agg_scores, decreasing=True)
        print "So Far=" + str(np.sum(queried_labels)) + " Out Of " + str(bt)

    return queried, queried_labels, stop_arrs, time_per_update

'''
# Without Sampling - But same as our approach
'''
def approach_PosNegNoSampling_Known_TIME(scores, labels, params, sgd_method):
    queried = []
    queried_labels = []
    stop_arrs = []
    n, m = scores.shape
    print n, m
    k = params['anchor_points']['K']

    # initialize w
    w = init_weights(m, params['weight_initialize'])
    w_init = w.copy()

    # get scores
    agg_scores = scores.dot(w)
    # sort scores
    sorted_indexes = sort(agg_scores, decreasing=True)
    bt = 0
    time_per_update = []
    while (bt < params['budget']):
        # =====Get the top unlabelled instance=====
        start_time = time.time()
        topind = 0
        while (sorted_indexes[topind] in queried):
            topind += 1

        sel_instance = sorted_indexes[topind]
        sel_label = labels[sel_instance]

        queried.append(sel_instance)
        queried_labels.append(sel_label)

        known_anchors, sampled_anchors = choose_posneg_anchors_known(sel_instance, sel_label, queried, queried_labels,
                                                                     sorted_indexes, agg_scores,
                                                                     params['anchor_points'])

        known_puvhat, sampled_puvhat = set_puvhat_posneg_known_NoSample(sel_label, sel_instance, agg_scores, known_anchors,
                                                               sampled_anchors, params['anchor_points'])

        anchors = np.concatenate((known_anchors, sampled_anchors), axis=0)
        puvhat = np.concatenate((known_puvhat, sampled_puvhat), axis=0)
        coordinate_type = params['cd']['coords']

        if coordinate_type == "UNION":
            coords = choose_diff_coordinates(scores[sel_instance], scores, known_anchors, sampled_anchors, params['cd'])
        else:
            coords = choose_coordinates_negonly(scores, sel_instance, known_anchors, sampled_anchors, params['cd'])

        if sgd_method == "SGD":
            w, stop_arr = optimize_posneg_sample_SGD(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                     params['cd'], params['learning'])
        elif sgd_method == "Momentum":
            w, stop_arr = optimize_posneg_sample_Momentum(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                          params['cd'], params['learning'], 0.75)

        elif sgd_method == "NAG":
            w, stop_arr = optimize_posneg_sample_NAG(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                          params['cd'], params['learning'], 0.75)

        elif sgd_method == "SGD_Batch":
            w, stop_arr = optimize_posneg_sample_SGD_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                             params['cd'], params['learning'])

        elif sgd_method == "Momentum_Batch":
            w, stop_arr = optimize_posneg_sample_Momentum_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                             params['cd'], params['learning'], 0.75)
        elif sgd_method == "NAG_Batch":
            w, stop_arr = optimize_posneg_sample_NAG_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                            params['cd'], params['learning'], 0.75)

        end_time = time.time() - start_time
        time_per_update.append(end_time)
        stop_arrs.append(stop_arr)
        bt += 1

        if (np.sum(queried_labels) >= np.sum(labels)):
            break

        agg_scores = scores.dot(w)
        sorted_indexes = sort(agg_scores, decreasing=True)
        print "So Far=" + str(np.sum(queried_labels)) + " Out Of " + str(bt)

    return queried, queried_labels, stop_arrs, time_per_update

'''
# Without Sampling 2 - No xu - xv
'''
def approach_PosNegNoSampling2_Known_TIME(scores, labels, params, sgd_method):
    queried = []
    queried_labels = []
    stop_arrs = []
    n, m = scores.shape
    print n, m
    k = params['anchor_points']['K']

    # initialize w
    w = init_weights(m, params['weight_initialize'])
    w_init = w.copy()

    # get scores
    agg_scores = scores.dot(w)
    # sort scores
    sorted_indexes = sort(agg_scores, decreasing=True)
    bt = 0
    time_per_update = []
    while (bt < params['budget']):
        # =====Get the top unlabelled instance=====
        start_time = time.time()
        topind = 0
        while (sorted_indexes[topind] in queried):
            topind += 1

        sel_instance = sorted_indexes[topind]
        sel_label = labels[sel_instance]

        queried.append(sel_instance)
        queried_labels.append(sel_label)

        known_anchors, sampled_anchors = choose_posneg_anchors_known_NoSample2(sel_instance, sel_label, queried, queried_labels,
                                                                     sorted_indexes, agg_scores,
                                                                     params['anchor_points'])

        known_puvhat, sampled_puvhat = set_puvhat_posneg_known_NoSample(sel_label, sel_instance, agg_scores, known_anchors,
                                                               sampled_anchors, params['anchor_points'])

        anchors = np.concatenate((known_anchors, sampled_anchors), axis=0)
        puvhat = np.concatenate((known_puvhat, sampled_puvhat), axis=0)
        coordinate_type = params['cd']['coords']

        if coordinate_type == "UNION":
            coords = choose_diff_coordinates(scores[sel_instance], scores, known_anchors, sampled_anchors, params['cd'])
        else:
            coords = choose_coordinates_negonly(scores, sel_instance, known_anchors, sampled_anchors, params['cd'])

        if sgd_method == "SGD":
            w, stop_arr = optimize_posneg_sample_SGD(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                     params['cd'], params['learning'])
        elif sgd_method == "Momentum":
            w, stop_arr = optimize_posneg_sample_Momentum(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                          params['cd'], params['learning'], 0.75)

        elif sgd_method == "NAG":
            w, stop_arr = optimize_posneg_sample_NAG(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                          params['cd'], params['learning'], 0.75)

        elif sgd_method == "SGD_Batch":
            w, stop_arr = optimize_posneg_sample_SGD_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                             params['cd'], params['learning'])

        elif sgd_method == "Momentum_Batch":
            w, stop_arr = optimize_posneg_sample_Momentum_Lim100_NoS2(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                             params['cd'], params['learning'], 0.75)
        elif sgd_method == "NAG_Batch":
            w, stop_arr = optimize_posneg_sample_NAG_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                            params['cd'], params['learning'], 0.75)

        end_time = time.time() - start_time
        time_per_update.append(end_time)
        stop_arrs.append(stop_arr)
        bt += 1

        if (np.sum(queried_labels) >= np.sum(labels)):
            break

        agg_scores = scores.dot(w)
        sorted_indexes = sort(agg_scores, decreasing=True)
        print "So Far=" + str(np.sum(queried_labels)) + " Out Of " + str(bt)

    return queried, queried_labels, stop_arrs, time_per_update

'''
# Variant 1.
PosNeg - DumberSampling - ALL
'''
def approach_PosNeg_DumberSampling(scores, labels, params):
    queried = []
    queried_labels = []
    stop_arrs = []
    n, m = scores.shape
    print n, m
    k = params['anchor_points']['K']

    # initialize w
    w = init_weights(m, params['weight_initialize'])
    w_init = w.copy()

    # get scores
    agg_scores = scores.dot(w)
    # sort scores
    sorted_indexes = sort(agg_scores, decreasing=True)
    bt = 0

    while (bt < params['budget']):
        # =====Get the top unlabelled instance=====
        topind = 0
        while (sorted_indexes[topind] in queried):
            topind += 1

        sel_instance = sorted_indexes[topind]
        sel_label = labels[sel_instance]

        queried.append(sel_instance)
        queried_labels.append(sel_label)

        known_anchors, sampled_anchors = choose_posneg_anchors_known_dsample(sel_instance, sel_label, queried, queried_labels,
                                                                     sorted_indexes, agg_scores, params['anchor_points'])

        known_puvhat, sampled_puvhat = set_puvhat_posneg_known(sel_label, sel_instance, agg_scores, known_anchors,
                                                               sampled_anchors, params['anchor_points'])

        coordinate_type = params['cd']
        if coordinate_type == "UNION":
            coords = choose_diff_coordinates(scores[sel_instance], scores, known_anchors, sampled_anchors, params['cd'])
        else:
            coords = choose_coordinates_negonly(scores, sel_instance, known_anchors, sampled_anchors, params['cd'])

        w, stop_arr = optimize_posneg_sample(sel_label, sel_instance, w, scores, known_puvhat, sampled_puvhat,
                                             known_anchors, sampled_anchors, coords, params['cd'], params['learning'])

        stop_arrs.append(stop_arr)

        bt += 1

        if (np.sum(queried_labels) >= np.sum(labels)):
            break

        agg_scores = scores.dot(w)
        sorted_indexes = sort(agg_scores, decreasing=True)
        print "So Far=" + str(np.sum(queried_labels)) + " Out Of " + str(bt)

    return queried, queried_labels, stop_arrs

'''
# Variant 2.
NegOnly - Sampling - NonZero
'''
def approach_NegOnly_Known(scores, labels, params):
    queried = []
    queried_labels = []
    stop_arrs = []
    n, m = scores.shape
    print n, m
    k = params['anchor_points']['K']

    # initialize w
    w = init_weights(m, params['weight_initialize'])
    w_init = w.copy()

    # get scores
    agg_scores = scores.dot(w)
    # sort scores
    sorted_indexes = sort(agg_scores, decreasing=True)

    bt = 0
    while (bt < params['budget']):
        # =====Get the top unlabelled instance=====
        topind = 0
        while (sorted_indexes[topind] in queried):
            topind += 1

        sel_instance = sorted_indexes[topind]
        sel_label = labels[sel_instance]

        queried.append(sel_instance)
        queried_labels.append(sel_label)

        # print "Selected Label:"+str(sel_label)

        if sel_label == 0:
            known_anchors, sampled_anchors = choose_negonly_anchors(sel_instance, sel_label, queried, queried_labels,
                                                                    sorted_indexes, agg_scores, params['anchor_points'])

            puvhat = np.zeros([k, ])
            puv = np.zeros([k, ])

            known_puvhat, sampled_puvhat = set_puvhat_negonly(sel_label, sel_instance, agg_scores, known_anchors,
                                                              sampled_anchors, params['anchor_points'])

            coordinates = choose_coordinates_negonly(scores, sel_instance, known_anchors, sampled_anchors, params['cd'])

            w, stop_arr = optimize_negonly(sel_label, sel_instance, w, scores, known_puvhat, sampled_puvhat,
                                           known_anchors,
                                           sampled_anchors, coordinates, params['cd'], params['learning'])

            stop_arrs.append(stop_arr)

        bt += 1

        if (np.sum(queried_labels) >= np.sum(labels)):
            break

        agg_scores = scores.dot(w)
        sorted_indexes = sort(agg_scores, decreasing=True)
        print "So Far=" + str(np.sum(queried_labels)) + " Out Of " + str(bt)

    return queried, queried_labels, stop_arrs

'''
# Varaint 3.
PosNeg - RandomSampling - ALL 
'''
def approach_PosNeg_RandomSampling(scores, labels, params):
    queried = []
    queried_labels = []
    stop_arrs = []
    n, m = scores.shape
    print n, m
    k = params['anchor_points']['K']

    # initialize w
    w = init_weights(m, params['weight_initialize'])
    w_init = w.copy()

    # get scores
    agg_scores = scores.dot(w)
    # sort scores
    sorted_indexes = sort(agg_scores, decreasing=True)
    bt = 0

    num_sampled=0
    while (bt < params['budget']):
        # =====Get the top unlabelled instance=====
        topind = 0
        while (sorted_indexes[topind] in queried):
            topind += 1

        sel_instance = sorted_indexes[topind]
        sel_label = labels[sel_instance]

        queried.append(sel_instance)
        queried_labels.append(sel_label)

        known_anchors, sampled_anchors = choose_posneg_anchors_known_rsample(sel_instance, sel_label, queried, queried_labels,
                                                                     sorted_indexes, agg_scores, params['anchor_points'])

        known_puvhat, sampled_puvhat = set_puvhat_posneg_known(sel_label, sel_instance, agg_scores, known_anchors,
                                                               sampled_anchors, params['anchor_points'])

        if(len(sampled_anchors)>0):
            print "SelLabel="+str(sel_label)+" and sampled labels="+str(labels[np.array(sampled_anchors)])
            for s_lbl in labels[np.array(sampled_anchors)]:
                if s_lbl == sel_label:
                    num_sampled+=1

        coordinate_type = params['cd']
        if coordinate_type == "UNION":
            coords = choose_diff_coordinates(scores[sel_instance], scores, known_anchors, sampled_anchors, params['cd'])
        else:
            coords = choose_coordinates_negonly(scores, sel_instance, known_anchors, sampled_anchors, params['cd'])

        w, stop_arr = optimize_posneg_sample(sel_label, sel_instance, w, scores, known_puvhat, sampled_puvhat,
                                             known_anchors, sampled_anchors, coords, params['cd'], params['learning'])

        stop_arrs.append(stop_arr)

        bt += 1

        if (np.sum(queried_labels) >= np.sum(labels)):
            break

        agg_scores = scores.dot(w)
        sorted_indexes = sort(agg_scores, decreasing=True)
        print "So Far=" + str(np.sum(queried_labels)) + " Out Of " + str(bt)

    return queried, queried_labels, stop_arrs, num_sampled


'''
# Variant 4
PosNeg - Only Sampling - ALL
'''
def approach_PosNeg_Sampling(scores, labels, params):
    queried = []
    queried_labels = []
    stop_arrs = []
    n, m = scores.shape
    print n, m
    k = params['anchor_points']['K']

    # initialize w
    w = init_weights(m, params['weight_initialize'])
    w_init = w.copy()

    # get scores
    agg_scores = scores.dot(w)
    # sort scores
    sorted_indexes = sort(agg_scores, decreasing=True)

    bt = 0
    while (bt < params['budget']):
        # =====Get the top unlabelled instance=====
        topind = 0
        while (sorted_indexes[topind] in queried):
            topind += 1

        sel_instance = sorted_indexes[topind]
        sel_label = labels[sel_instance]

        queried.append(sel_instance)
        queried_labels.append(sel_label)

        sampled_anchors, known_anchors = choose_posneg_anchors(sel_instance, sel_label, queried, queried_labels,
                                                               sorted_indexes, agg_scores, params['anchor_points'])

        print "Chosen Anchors="+str(labels[sampled_anchors])
        puvhat = np.zeros([k,])
        puv = np.zeros([k,])

        known_puvhat, sampled_puvhat = set_puvhat_posneg(sel_label, sel_instance, agg_scores, known_anchors,
                                                         sampled_anchors, params['anchor_points'])

        coords = choose_coordinates_negonly(scores, sel_instance, known_anchors, sampled_anchors, params['cd'])

        w, stop_arr =  optimize_posneg_sample(sel_label, sel_instance, w, scores, known_puvhat, sampled_puvhat, known_anchors,
                                       sampled_anchors, coords, params['cd'], params['learning'])

        stop_arrs.append(stop_arr)

        bt+=1

        if(np.sum(queried_labels) >= np.sum(labels)):
            break

        agg_scores = scores.dot(w)
        sorted_indexes = sort(agg_scores, decreasing = True)
        print "So Far="+str(np.sum(queried_labels))+ " Out Of "+str(bt)

    return queried, queried_labels, stop_arrs


'''
# Variant 5
PosNeg Known SGD 
'''
def approach_PosNeg_Known_SGD(scores, labels, params, method_name):
    queried = []
    queried_labels = []
    stop_arrs = []
    n, m = scores.shape
    print n, m
    k = params['anchor_points']['K']

    # initialize w
    w = init_weights(m, params['weight_initialize'])
    w_init = w.copy()

    # get scores
    agg_scores = scores.dot(w)
    # sort scores
    sorted_indexes = sort(agg_scores, decreasing=True)
    bt = 0

    while (bt < params['budget']):
        # =====Get the top unlabelled instance=====
        topind = 0
        while (sorted_indexes[topind] in queried):
            topind += 1

        sel_instance = sorted_indexes[topind]
        sel_label = labels[sel_instance]

        queried.append(sel_instance)
        queried_labels.append(sel_label)

        known_anchors, sampled_anchors = choose_posneg_anchors_known(sel_instance, sel_label, queried, queried_labels,
                                                                     sorted_indexes, agg_scores,
                                                                     params['anchor_points'])

        known_puvhat, sampled_puvhat = set_puvhat_posneg_known(sel_label, sel_instance, agg_scores, known_anchors,
                                                               sampled_anchors, params['anchor_points'])

        coordinate_type = params['cd']['coords']
        if coordinate_type == "UNION":
            coords = choose_diff_coordinates(scores[sel_instance], scores, known_anchors, sampled_anchors, params['cd'])
        else:
            coords = choose_coordinates_negonly(scores, sel_instance, known_anchors, sampled_anchors, params['cd'])

        w, stop_arr = optimize_posneg_sample_NEW(sel_label, sel_instance, w, scores, known_puvhat, sampled_puvhat,
                                             known_anchors, sampled_anchors, coords, params['cd'], params['learning'],
                                             method_name)

        stop_arrs.append(stop_arr)

        bt += 1

        if (np.sum(queried_labels) >= np.sum(labels)):
            break

        agg_scores = scores.dot(w)
        sorted_indexes = sort(agg_scores, decreasing=True)
        print "So Far=" + str(np.sum(queried_labels)) + " Out Of " + str(bt)

    return queried, queried_labels, stop_arrs