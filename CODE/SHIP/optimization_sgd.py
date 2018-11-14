import numpy as np
from loss import *
from utils import *
from weight_utils import *
from coordinate_selector import *

crossEnt = lambda p, p_opt: np.sum(np.multiply(-p_opt, np.log(p)), axis=1)

def crossEnt_Loss(puvhat, puv):
    p_opt = np.concatenate((puvhat[:,None], (1-puvhat)[:,None]), axis=1)
    p = np.concatenate((puv[:,None], (1-puv)[:,None]), axis=1)
    #loss = np.mean(crossEnt(p, p_opt))
    loss = np.sum(crossEnt(p, p_opt))
    return loss

def get_gradient2(anchors, sel_instance, puv, puvhat, scores):
    gradient = np.zeros((anchors.shape[0], scores.shape[1]))
    for i in range(len(anchors)):
        if anchors[i] == -1:
            gradient[i,:] = (scores[sel_instance, :]) * (puv[i] - puvhat[i])
        else:
            gradient[i,:] = (scores[sel_instance, :] - scores[anchors[i], :]) * (puv[i] - puvhat[i])

    return gradient

def get_gradient(anchors, sel_instance, puv, puvhat, scores):
    gradient = (scores[sel_instance,:] - scores[anchors,:]) * (puv - puvhat)[:,np.newaxis]
    return gradient

def get_puv(scores, w, anchors, sel_instance):
    agg_scores = scores.dot(w)
    puv = expit(agg_scores[sel_instance] -  agg_scores[anchors])
    return puv

def optimize_posneg_sample_SGD(sel_label, sel_instance, w, scores, puvhat, anchors, coords, cd_params, learning_params):
    anchors = anchors.astype(int)
    learning_rate = learning_params['learning_rate']
    Tmax = learning_params['num_iters']
    stop_arr = []
    agg_scores = scores.dot(w)

    puv = expit(agg_scores[sel_instance] -  agg_scores[anchors])

    curr_loss = crossEnt_Loss(puvhat, puv)
    iter = 0

    while(True):
        w_new = w.copy()

        #Get Gradient
        diff = get_gradient(anchors, sel_instance, puv, puvhat, scores)

        # Updates
        for i in range(len(anchors)):
            w_index = np.array(coords[i])
            w_new[w_index] = w_new[w_index] - (learning_rate * diff[i, w_index])

        convergence_diff = np.linalg.norm(np.abs(w_new - w))
        w = w_new
        w = normalize_weights(w)
        puv_new = get_puv(scores, w, anchors, sel_instance)

        new_loss = crossEnt_Loss(puvhat, puv_new)
        loss_diff = curr_loss - new_loss
        curr_loss = new_loss
        puv = puv_new

        #print curr_loss
        iter += 1
        if (convergence_diff < 1e-6):
            stop_condition = 1
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (iter > Tmax):
            stop_condition = 2
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (loss_diff < 1e-8):
            stop_condition = 3
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break

    print "End Loss=" + str(curr_loss)
    return w, stop_arr

def optimize_posneg_sample_Momentum(sel_label, sel_instance, w, scores, puvhat, anchors, coords, cd_params, learning_params, gamma):
    anchors = anchors.astype(int)
    learning_rate = learning_params['learning_rate']
    Tmax = learning_params['num_iters']
    stop_arr = []
    agg_scores = scores.dot(w)

    puv = expit(agg_scores[sel_instance] -  agg_scores[anchors])

    curr_loss = crossEnt_Loss(puvhat, puv)
    iter = 0

    prev_update = None

    while(True):
        w_new = w.copy()

        #Get Gradient
        diff = get_gradient(anchors, sel_instance, puv, puvhat, scores)
        if prev_update is None:
            update = learning_rate * diff
        else:
            update = gamma * prev_update + learning_rate * diff

        # Updates
        for i in range(len(anchors)):
            w_index = np.array(coords[i])
            w_new[w_index] = w_new[w_index] - (update[i, w_index])


        prev_update = update

        convergence_diff = np.linalg.norm(np.abs(w_new - w))
        w = w_new
        w = normalize_weights(w)
        puv_new = get_puv(scores, w, anchors, sel_instance)
        puv = puv_new

        new_loss = crossEnt_Loss(puvhat, puv_new)
        loss_diff = curr_loss - new_loss
        curr_loss = new_loss

        #print curr_loss
        iter += 1
        if (convergence_diff < 1e-6):
            stop_condition = 1
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (iter > Tmax):
            stop_condition = 2
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (loss_diff < 1e-8):
            stop_condition = 3
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break

    print "End Loss="+str(curr_loss)
    return w, stop_arr

def optimize_posneg_sample_NAG(sel_label, sel_instance, w, scores, puvhat, anchors, coords, cd_params, learning_params, gamma):
    anchors = anchors.astype(int)
    learning_rate = learning_params['learning_rate']
    Tmax = learning_params['num_iters']
    stop_arr = []
    agg_scores = scores.dot(w)

    puv = expit(agg_scores[sel_instance] - agg_scores[anchors])

    curr_loss = crossEnt_Loss(puvhat, puv)
    iter = 0

    prev_update = None

    while(True):
        w_new = w.copy()

        if prev_update is None:
            diff = get_gradient(anchors, sel_instance, puv, puvhat, scores)
            update = learning_rate * diff
        else:
            w_diff = w.copy()

            for i in range(len(anchors)):
                w_index = np.array(coords[i])
                w_diff[w_index] = w_diff[w_index] - (gamma * prev_update[i, w_index])

            puv_diff = get_puv(scores, w_diff, anchors, sel_instance)
            diff = get_gradient(anchors, sel_instance, puv_diff, puvhat, scores)

            update = gamma * prev_update + learning_rate * diff

        # Updates
        for i in range(len(anchors)):
            w_index = np.array(coords[i])
            w_new[w_index] = w_new[w_index] - (update[i, w_index])

        prev_update = update

        convergence_diff = np.linalg.norm(np.abs(w_new - w))
        w = w_new
        w = normalize_weights(w)
        puv_new = get_puv(scores, w, anchors, sel_instance)
        puv = puv_new

        new_loss = crossEnt_Loss(puvhat, puv_new)
        loss_diff = curr_loss - new_loss
        curr_loss = new_loss

        #print "Loss="+str(curr_loss)
        iter += 1
        if (convergence_diff < 1e-6):
            stop_condition = 1
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (iter > Tmax):
            stop_condition = 2
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (loss_diff < 1e-8):
            stop_condition = 3
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break

    return w, stop_arr

def optimize_posneg_sample_AG2(sel_label, sel_instance, w, scores, puvhat, anchors, coords, cd_params, learning_params):
    anchors = anchors.astype(int)
    learning_rate = learning_params['learning_rate']
    Tmax = learning_params['num_iters']
    stop_arr = []
    agg_scores = scores.dot(w)

    puv = expit(agg_scores[sel_instance] - agg_scores[anchors])

    curr_loss = crossEnt_Loss(puvhat, puv)
    iter = 0
    ys = [w]
    xs = [w]

    while(True):
        y = ys[-1]
        x = xs[-1]

        puv_y = get_puv(scores, y, anchors, sel_instance)
        g = get_gradient(anchors, sel_instance, puv_y, puvhat, scores)
        x_update = learning_rate * g

        x_plus = y.copy()
        # Updates
        for i in range(len(anchors)):
            w_index = np.array(coords[i])
            x_plus[w_index] = x_plus[w_index] - x_update[i, w_index]

        # Step A COMPLETED.
        convergence_diff = np.linalg.norm(np.abs(x_plus - x))
        x_check = normalize_weights(x_plus)
        puv_new = get_puv(scores, x_check, anchors, sel_instance)
        new_loss = crossEnt_Loss(puvhat, puv_new)
        loss_diff = curr_loss - new_loss
        curr_loss = new_loss
        print "Loss=" + str(curr_loss)

        iter += 1
        if (convergence_diff < 1e-6):
            stop_condition = 1
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (iter > Tmax):
            stop_condition = 2
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (loss_diff < 1e-8):
            stop_condition = 3
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break

        x_plus = x_check
        # Step B COMPLETED
        # IF NOT Converged, make y_plus
        y_plus = x_plus.copy()
        for i in range(len(anchors)):
            w_index = np.array(coords[i])
            y_plus[w_index] = y_plus[w_index] + ((iter-1)/float(iter+2)) * (x_plus - x)[w_index]

        y_plus = normalize_weights(y_plus)
        ys.append(y_plus)
        xs.append(x_plus)

    print "Loss="+str(curr_loss)
    return x_plus, stop_arr

class BacktrackingLineSearch(object):
    def __init__(self):
        self.alpha = 1.0
        self.beta = 0.9

    def f(self, puvhat, puv, coords, w, g, learning_rate):
        if g is None:
            return crossEnt_Loss(puvhat, puv)
        else:
            w_diff = w.copy()
            for i in range(len(coords)):
                w_index = np.array(coords[i])
                w_diff[w_index] = w_diff[w_index] - (learning_rate * g[i, w_index])

            w_diff = normalize_weights(w_diff)
            puv_diff = get_puv(scores, w_diff, anchors, sel_instance)

            return crossEnt_Loss(puvhat, puv_diff)


    def __call__(self, g, w, anchors, puv, puvhat, coords):
        a = self.alpha
        f = self.f

        g_sum = np.sum(g, axis=0)
        while(f(puvhat, puv, coords, w, g, a) > f(puvhat, puv, coords, w, None, a) - 0.5 * a * (g_sum.dot(g_sum))):
            a = a * self.beta
            if a < 1e-4:
                return a

        return a

def optimize_posneg_sample_AG2_BT(sel_label, sel_instance, w, scores, puvhat, anchors, coords, cd_params, learning_params):
    anchors = anchors.astype(int)
    learning_rate = learning_params['learning_rate']
    Tmax = learning_params['num_iters']
    stop_arr = []
    agg_scores = scores.dot(w)

    puv = expit(agg_scores[sel_instance] - agg_scores[anchors])

    curr_loss = crossEnt_Loss(puvhat, puv)
    iter = 0
    ys = [w]
    xs = [w]

    btfunc = BacktrackingLineSearch()

    while (True):
        y = ys[-1]
        x = xs[-1]

        puv_y = get_puv(scores, y, anchors, sel_instance)
        g = get_gradient(anchors, sel_instance, puv_y, puvhat, scores)

        lr = btfunc(g, w, anchors, puv_y, puvhat, coords)
        x_update = lr * g

        x_plus = y.copy()
        # Updates
        for i in range(len(anchors)):
            w_index = np.array(coords[i])
            x_plus[w_index] = x_plus[w_index] - x_update[i, w_index]

        # Step A COMPLETED.
        convergence_diff = np.linalg.norm(np.abs(x_plus - x))
        x_check = normalize_weights(x_plus)
        puv_new = get_puv(scores, x_check, anchors, sel_instance)
        new_loss = crossEnt_Loss(puvhat, puv_new)
        loss_diff = curr_loss - new_loss
        curr_loss = new_loss
        print "Loss=" + str(curr_loss)

        iter += 1
        if (convergence_diff < 1e-6):
            stop_condition = 1
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (iter > Tmax):
            stop_condition = 2
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (loss_diff < 1e-8):
            stop_condition = 3
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break

        x_plus = x_check
        # Step B COMPLETED
        # IF NOT Converged, make y_plus
        y_plus = x_plus.copy()
        for i in range(len(anchors)):
            w_index = np.array(coords[i])
            y_plus[w_index] = y_plus[w_index] + ((iter - 1) / float(iter + 2)) * (x_plus - x)[w_index]

        y_plus = normalize_weights(y_plus)
        ys.append(y_plus)
        xs.append(x_plus)

    print "Loss=" + str(curr_loss)
    return x_plus, stop_arr

def optimize_posneg_sample_SGD_Backtracking(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                            cd_params, learning_params, beta):
    anchors = anchors.astype(int)
    learning_rate = learning_params['learning_rate']
    learning_rate = 1.0
    Tmax = learning_params['num_iters']
    stop_arr = []
    agg_scores = scores.dot(w)

    puv = expit(agg_scores[sel_instance] -  agg_scores[anchors])

    curr_loss = crossEnt_Loss(puvhat, puv)
    iter = 0

    while(True):
        w_new = w.copy()

        ###Get Ideal Learning Rate:
        lr = learning_rate
        # Get new loss
        diff = get_gradient(anchors, sel_instance, puv, puvhat, scores)
        grad_norm2 = np.linalg.norm(diff, 2)
        w_diff = w.copy()
        for i in range(len(anchors)):
            w_index = np.array(coords[i])
            w_diff[w_index] = w_diff[w_index] - (lr * diff[i, w_index])

        w_diff = normalize_weights(w_diff)
        puv_diff = get_puv(scores, w_diff, anchors, sel_instance)
        new_loss_diff = crossEnt_Loss(puvhat, puv_diff)

        # Check the conditions
        while(new_loss_diff > curr_loss - 0.5 * lr * grad_norm2**2):
            lr = beta * lr
            w_diff = w.copy()
            for i in range(len(anchors)):
                w_index = np.array(coords[i])
                w_diff[w_index] = w_diff[w_index] - (lr * diff[i, w_index])

            w_diff = normalize_weights(w_diff)
            puv_diff = get_puv(scores, w_diff, anchors, sel_instance)
            new_loss_diff = crossEnt_Loss(puvhat, puv_diff)

        diff = get_gradient(anchors, sel_instance, puv, puvhat, scores)
        for i in range(len(anchors)):
            w_index = np.array(coords[i])
            w_new[w_index] = w_new[w_index] - (lr * diff[i, w_index])

        convergence_diff = np.linalg.norm(np.abs(w_new - w))
        w = w_new
        w = normalize_weights(w)

        puv_new = get_puv(scores, w, anchors, sel_instance)
        new_loss = crossEnt_Loss(puvhat, puv_new)
        puv = puv_new

        loss_diff = curr_loss - new_loss
        curr_loss = new_loss

        print "Ideal LR="+str(lr)+ " and Loss="+str(new_loss)+" and Loss diff="+str(loss_diff)
        # print curr_loss
        iter += 1
        if (convergence_diff < 1e-6):
            stop_condition = 1
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (iter > Tmax):
            stop_condition = 2
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (loss_diff < 1e-8):
            stop_condition = 3
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break

    print "End Loss=" + str(curr_loss)
    return w, stop_arr

def optimize_posneg_sample_NAG_Backtracking(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                            cd_params, learning_params, gamma):
    anchors = anchors.astype(int)
    learning_rate = learning_params['learning_rate']
    Tmax = learning_params['num_iters']
    stop_arr = []
    agg_scores = scores.dot(w)

    puv = expit(agg_scores[sel_instance] - agg_scores[anchors])

    curr_loss = crossEnt_Loss(puvhat, puv)
    iter = 0

    prev_update = None
    btfunc = BacktrackingLineSearch()

    while (True):
        w_new = w.copy()

        diff = get_gradient(anchors, sel_instance, puv, puvhat, scores)
        learning_rate = btfunc(diff, w, anchors, puv, puvhat, coords)

        if prev_update is None:
            update = learning_rate * diff
        else:
            w_diff = w.copy()

            for i in range(len(anchors)):
                w_index = np.array(coords[i])
                w_diff[w_index] = w_diff[w_index] - (gamma * prev_update[i, w_index])

            puv_diff = get_puv(scores, w_diff, anchors, sel_instance)
            diff = get_gradient(anchors, sel_instance, puv_diff, puvhat, scores)

            update = gamma * prev_update + learning_rate * diff

        # Updates
        for i in range(len(anchors)):
            w_index = np.array(coords[i])
            w_new[w_index] = w_new[w_index] - (update[i, w_index])

        prev_update = update

        convergence_diff = np.linalg.norm(np.abs(w_new - w))
        w = w_new
        w = normalize_weights(w)
        puv_new = get_puv(scores, w, anchors, sel_instance)
        puv = puv_new

        new_loss = crossEnt_Loss(puvhat, puv_new)
        loss_diff = curr_loss - new_loss
        curr_loss = new_loss

        print "Loss="+str(curr_loss)
        iter += 1
        if (convergence_diff < 1e-6):
            stop_condition = 1
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (iter > Tmax):
            stop_condition = 2
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (loss_diff < 1e-8):
            stop_condition = 3
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break

    return w, stop_arr

def optimize_posneg_sample_SGD_1(sel_label, sel_instance, w, scores, puvhat, anchors, coords, learning_params):
    anchors = anchors.astype(int)
    learning_rate = learning_params['learning_rate']
    Tmax = learning_params['num_iters']
    stop_arr = []
    agg_scores = scores.dot(w)

    puv = expit(agg_scores[sel_instance] - agg_scores[anchors])

    curr_loss = crossEnt_Loss(puvhat, puv)
    iter = 0

    w_new = w.copy()

    for i in range(len(anchors)):
        diff =  (scores[sel_instance,:] - scores[anchors[i],:]) * (puv[i] - puvhat[i])
        w_index = np.array(coords[i])
        w_new[w_index] = w_new[w_index] - diff[w_index]

    w = w_new
    w = normalize_weights(w)

    return w, stop_arr

def create_batches(size_samples, size_perBatch):
    batches = []
    numBatches = int(np.ceil(size_samples * 1.0 / size_perBatch))
    for i in range(numBatches):
        batches.append(np.arange(i*size_perBatch, min((i+1)*size_perBatch,size_samples)))

    return batches

def optimize_posneg_sample_SGD_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords, cd_params, learning_params):
    coords=np.array(coords)
    anchors = anchors.astype(int)
    learning_rate = learning_params['learning_rate']
    Tmax = learning_params['num_iters']
    stop_arr = []
    agg_scores = scores.dot(w)
    puv = expit(agg_scores[sel_instance] - agg_scores[anchors])

    cur_loss = crossEnt_Loss(puvhat, puv)

    iter = 0

    batches = create_batches(len(anchors), 100)

    while (True):
        w_new = w.copy()

        batch_index = iter%len(batches)
        batch_anchors = anchors[batches[batch_index]]
        batch_puv = puv[batches[batch_index]]
        batch_puvhat = puvhat[batches[batch_index]]
        batch_coords = coords[batches[batch_index]]

        for i in range(len(batch_anchors)):
            w_index = np.array(batch_coords[i])
            diff = (scores[sel_instance, w_index] - scores[batch_anchors[i].astype(int), w_index]) * (batch_puv[i] - batch_puvhat[i])
            w_new[w_index] = w_new[w_index] - (learning_rate * diff)

        convergence_diff = np.linalg.norm(np.abs(w_new - w))
        w = w_new
        w = normalize_weights(w)

        puv = get_puv(scores, w, anchors, sel_instance)

        new_loss = crossEnt_Loss(puvhat, puv)
        loss_diff = cur_loss - new_loss
        cur_loss = new_loss

        iter += 1
        if (convergence_diff < 1e-6):
            stop_condition = 1
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (iter > Tmax):
            stop_condition = 2
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (loss_diff < 1e-8):
            stop_condition = 3
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break

    print "End Loss="+str(cur_loss)
    print stop_arr
    return w, stop_arr

def optimize_posneg_sample_Momentum_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords, cd_params, learning_params, gamma):
    anchors = anchors.astype(int)
    coords = np.array(coords)
    learning_rate = learning_params['learning_rate']
    Tmax = learning_params['num_iters']
    stop_arr = []
    agg_scores = scores.dot(w)
    puv = expit(agg_scores[sel_instance] -  agg_scores[anchors])

    curr_loss = crossEnt_Loss(puvhat, puv)
    iter = 0
    batches = create_batches(len(anchors), 100)
    prev_update = None

    while(True):
        w_new = w.copy()

        batch_index = iter % len(batches)
        batch_anchors = anchors[batches[batch_index]]
        batch_puv = puv[batches[batch_index]]
        batch_puvhat = puvhat[batches[batch_index]]
        batch_coords = coords[batches[batch_index]]

        #Get Gradient
        diff = get_gradient(batch_anchors, sel_instance, batch_puv, batch_puvhat, scores)

        prev_update_overall = np.sum(prev_update, axis=0)
        prev_update_overall = np.tile(prev_update_overall, (len(batch_anchors), 1))

        if prev_update is None:
            update = learning_rate * diff
        else:
            update = gamma * prev_update_overall + learning_rate * diff

        # Updates
        for i in range(len(batch_anchors)):
            w_index = np.array(batch_coords[i])
            w_new[w_index] = w_new[w_index] - (update[i, w_index])

        prev_update = update

        convergence_diff = np.linalg.norm(np.abs(w_new - w))
        w = w_new
        w = normalize_weights(w)
        puv_new = get_puv(scores, w, anchors, sel_instance)
        puv = puv_new

        new_loss = crossEnt_Loss(puvhat, puv_new)
        loss_diff = curr_loss - new_loss
        curr_loss = new_loss

        #print curr_loss
        iter += 1
        if (convergence_diff < 1e-6):
            stop_condition = 1
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (iter > Tmax):
            stop_condition = 2
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (loss_diff < 1e-8):
            stop_condition = 3
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break

    print "End Loss="+str(curr_loss)
    return w, stop_arr


def optimize_posneg_sample_Momentum_Lim100_NoS2(sel_label, sel_instance, w, scores, puvhat, anchors, coords, cd_params, learning_params, gamma):
    anchors = anchors.astype(int)
    coords = np.array(coords)
    learning_rate = learning_params['learning_rate']
    Tmax = learning_params['num_iters']
    stop_arr = []
    agg_scores = scores.dot(w)
    puv = expit(agg_scores[sel_instance] -  agg_scores[anchors])

    curr_loss = crossEnt_Loss(puvhat, puv)
    iter = 0
    batches = create_batches(len(anchors), 100)
    prev_update = None

    while(True):
        w_new = w.copy()

        batch_index = iter % len(batches)
        batch_anchors = anchors[batches[batch_index]]
        batch_puv = puv[batches[batch_index]]
        batch_puvhat = puvhat[batches[batch_index]]
        batch_coords = coords[batches[batch_index]]

        #Get Gradient
        diff = get_gradient2(batch_anchors, sel_instance, batch_puv, batch_puvhat, scores)

        prev_update_overall = np.sum(prev_update, axis=0)
        prev_update_overall = np.tile(prev_update_overall, (len(batch_anchors), 1))

        if prev_update is None:
            update = learning_rate * diff
        else:
            update = gamma * prev_update_overall + learning_rate * diff

        # Updates
        for i in range(len(batch_anchors)):
            w_index = np.array(batch_coords[i])
            w_new[w_index] = w_new[w_index] - (update[i, w_index])

        prev_update = update

        convergence_diff = np.linalg.norm(np.abs(w_new - w))
        w = w_new
        w = normalize_weights(w)
        puv_new = get_puv(scores, w, anchors, sel_instance)
        puv = puv_new

        new_loss = crossEnt_Loss(puvhat, puv_new)
        loss_diff = curr_loss - new_loss
        curr_loss = new_loss

        #print curr_loss
        iter += 1
        if (convergence_diff < 1e-6):
            stop_condition = 1
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (iter > Tmax):
            stop_condition = 2
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (loss_diff < 1e-8):
            stop_condition = 3
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break

    print "End Loss="+str(curr_loss)
    return w, stop_arr

def optimize_posneg_sample_NAG_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords, cd_params, learning_params, gamma):
    anchors = anchors.astype(int)
    coords = np.array(coords)
    learning_rate = learning_params['learning_rate']
    Tmax = learning_params['num_iters']
    stop_arr = []
    agg_scores = scores.dot(w)

    puv = expit(agg_scores[sel_instance] - agg_scores[anchors])

    curr_loss = crossEnt_Loss(puvhat, puv)
    iter = 0
    batches = create_batches(len(anchors), 100)

    prev_update = None

    while(True):
        w_new = w.copy()

        batch_index = iter % len(batches)
        batch_anchors = anchors[batches[batch_index]]
        batch_puv = puv[batches[batch_index]]
        batch_puvhat = puvhat[batches[batch_index]]
        batch_coords = coords[batches[batch_index]]

        if prev_update is None:
            diff = get_gradient(batch_anchors, sel_instance, batch_puv, batch_puvhat, scores)
            update = learning_rate * diff
        else:
            w_diff = w.copy()
            prev_update_overall = np.sum(prev_update, axis=0)
            prev_update_overall = np.tile(prev_update_overall,(len(batch_anchors),1))

            for i in range(len(batch_anchors)):
                w_index = np.array(batch_coords[i])
                w_diff[w_index] = w_diff[w_index] - (gamma * prev_update_overall[i, w_index])

            puv_diff = get_puv(scores, w_diff, batch_anchors, sel_instance)
            diff = get_gradient(batch_anchors, sel_instance, puv_diff, batch_puvhat, scores)

            update = gamma * prev_update_overall + learning_rate * diff

        # Updates
        for i in range(len(batch_anchors)):
            w_index = np.array(batch_coords[i])
            w_new[w_index] = w_new[w_index] - (update[i, w_index])

        prev_update = update

        convergence_diff = np.linalg.norm(np.abs(w_new - w))
        w = w_new
        w = normalize_weights(w)
        puv_new = get_puv(scores, w, anchors, sel_instance)
        puv = puv_new

        new_loss = crossEnt_Loss(puvhat, puv_new)
        loss_diff = curr_loss - new_loss
        curr_loss = new_loss

        #print "Loss="+str(curr_loss)
        iter += 1
        if (convergence_diff < 1e-6):
            stop_condition = 1
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (iter > Tmax):
            stop_condition = 2
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break
        if (loss_diff < 1e-8):
            stop_condition = 3
            stop_arr.append((stop_condition, convergence_diff, loss_diff, iter))
            break

    print "End Loss="+str(curr_loss)
    return w, stop_arr


if __name__ == '__main__':
    # Write testing code here:
    # Read the input file
    data_file = "/Users/hemanklamba/Documents/Experiments/Interactive_Outliers/Dataset/Feb_DS/ann.csv"
    X = np.loadtxt(data_file)
    print "Orig data read=" + str(X.shape)
    y = X[:, X.shape[1] - 1]
    X = X[:, 0:X.shape[1] - 1]
    # Get scores vector
    score_file = "/Users/hemanklamba/Documents/Experiments/Interactive_Outliers/Scores/Feb_DS/ann/ann_0_Sample_256TREE.csv"
    scores, y = load_scores_file(score_file, "Original")

    # Set weight vector
    w = np.ones(scores.shape[1])
    w = w / np.sqrt(w.dot(w))
    print "Shape of X=" + str(X.shape)
    print "Shape of w=" + str(w.shape)

    # Get an arbitrary sel_label and sel_instance
    sel_label = 1
    sel_instance = 201

    # Set puvhat
    known_puvhat = np.array([1.0, 1.0])
    sampled_puvhat = np.array(
        [0.62005177, 0.62005177, 0.61995821, 0.61877928, 0.61954655, 0.61982723, 0.62005177, 0.62005177, 0.62005177,
         0.61986465, 0.61964011, 0.61995821, 0.61995821, 0.61973367, 0.62014532, 0.61967753, 0.61995821, 0.61973367])
    # Set anchors
    known_anchors = np.array([233, 230])
    sampled_anchors = np.array([180, 154, 0, 40, 190, 99, 198, 72, 93, 47, 181, 176, 43, 18, 75, 79, 186, 77])

    # Set cd_params
    pos_coords = range(X.shape[1])
    neg_coords = range(X.shape[1])
    cd_params = {}
    cd_params['pos_coords'] = 'UNION'
    cd_params['neg_coords'] = 'UNION'
    cd_params['coords'] = 'UNION'

    #Set learning_params
    learning_params = {}
    learning_params['learning_rate'] = 0.1
    learning_params['num_iters'] = 5000
    learning_params['reg_constant'] = 0.1

    coords = choose_diff_coordinates(scores[sel_instance], scores, known_anchors, sampled_anchors, 'UNION')

    anchors = np.concatenate((known_anchors, sampled_anchors), axis=0)
    puv = get_puv(scores, w, anchors, sel_instance)
    puvhat = np.concatenate((known_puvhat, sampled_puvhat), axis=0)

    '''
    start_time = time.time()
    wsimp, sarr_simp = optimize_posneg_sample_SGD(sel_label, sel_instance, w, scores, puvhat, anchors, coords, cd_params,
                                                  learning_params)

    simple_time = time.time() - start_time
    print "Time taken=" + str(simple_time)
    print sarr_simp
    print wsimp
    #print np.argsort(wsimp)

    agg_scores_wsimp = scores.dot(wsimp)
    sorted_indexes_wsimp = sort(agg_scores_wsimp, decreasing=True)
    
    start_time = time.time()
    ws_mom, sarr_mom = optimize_posneg_sample_Momentum(sel_label, sel_instance, w, scores, puvhat, anchors, coords, cd_params,
                                    learning_params, 0.75)
    mom_time = time.time() - start_time
    print "Time taken="+str(mom_time)
    print sarr_mom
    print ws_mom
    #print np.argsort(ws_mom)

    agg_scores_wsmom =  scores.dot(ws_mom)
    sorted_indexes_wsmom = sort(agg_scores_wsmom, decreasing=True)

    print sorted_indexes_wsimp
    print sorted_indexes_wsmom
    print np.sum(sorted_indexes_wsimp - sorted_indexes_wsmom)
    
    
    start_time = time.time()
    ws_nag, sarr_nag = optimize_posneg_sample_NAG(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                       cd_params,
                                                       learning_params, 0.75)
    nag_time = time.time() - start_time
    print "Time taken=" + str(nag_time)
    print sarr_nag
    print ws_nag

    agg_scores_wsnag = scores.dot(ws_nag)
    sorted_indexes_wsnag = sort(agg_scores_wsnag, decreasing=True)

    print sorted_indexes_wsnag
    '''
    start_time = time.time()
    ws_nag_bt, sarr_nag_bt = optimize_posneg_sample_SGD_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                  cd_params, learning_params)
    nag_bt_time = time.time() - start_time
    print "Time taken=" + str(nag_bt_time)
    print ws_nag_bt
    print sarr_nag_bt

    start_time = time.time()
    ws_nag_bt, sarr_nag_bt = optimize_posneg_sample_Momentum_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors,
                                                               coords,
                                                               cd_params, learning_params, 0.75)
    nag_bt_time = time.time() - start_time
    print "Time taken=" + str(nag_bt_time)
    print ws_nag_bt
    print sarr_nag_bt

    start_time = time.time()
    ws_nag_bt, sarr_nag_bt = optimize_posneg_sample_NAG_Lim100(sel_label, sel_instance, w, scores, puvhat, anchors,
                                                                    coords,
                                                                    cd_params, learning_params, 0.75)
    nag_bt_time = time.time() - start_time
    print "Time taken=" + str(nag_bt_time)
    print ws_nag_bt
    print sarr_nag_bt


    '''
    start_time = time.time()
    ws_ag2, sarr_ag2 = optimize_posneg_sample_AG2(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                  cd_params, learning_params)
    ag2_time = time.time() - start_time
    print "Time taken="+str(ag2_time)
    print ws_ag2
    print sarr_ag2

    start_time = time.time()
    ws_ag2_BT, sarr_ag2_BT = optimize_posneg_sample_AG2_BT(sel_label, sel_instance, w, scores, puvhat, anchors, coords,
                                                  cd_params, learning_params)
    ag2_BT_time = time.time() - start_time
    print "Time taken=" + str(ag2_BT_time)
    print ws_ag2_BT
    print sarr_ag2_BT
    '''
