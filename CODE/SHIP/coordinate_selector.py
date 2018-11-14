from SHIP.utils import *
import numpy as np

def get_pos_coords(top_score):
    mod_top_score = np.abs(top_score)
    sorted_indices =  np.where(mod_top_score>0)[0]
    return sorted_indices

def get_top_coords(top_score, method):
    mod_top_score = np.abs(top_score)
    m = method.split("-")[1]
    m = int(m)
    sorted_indices = sort(mod_top_score, decreasing=True)
    return sorted_indices[0:m]
    
def get_prob_coords(top_score, num_iters):
    mod_score = np.abs(top_score)
    pos_indexes = mod_score>0
    mod_score2 = exponentiate(mod_score[pos_indexes])
    probs = np.zeros(top_score.shape)
    probs[pos_indexes] = mod_score2/np.sum(mod_score2)
    sel_index = np.random.choice(range(probs.shape[0]), p = probs, size=num_iters, replace=True)
    return sel_index

def choose_noK_diff_coordinates(scores, pairs, cd_params):
    '''
    :return: Array of co-ordiantes to update in order of [pairs,]
    '''
    coordinates = []
    for pair in pairs:
        coordinates_one = set(get_pos_coords(scores[pair[0]]))
        coordinates_two = set(get_pos_coords(scores[pair[1]]))
        union_coordinates = list(coordinates_one.union(coordinates_two))
        coordinates.append(union_coordinates)

    return coordinates

def choose_noK_small_coordinates(top_score, scores, anchors, cd_params):
    '''
    :return: Array of co-ordinates to update.
    '''
    coordinates  =[]
    selPoint_coordinates = get_pos_coords(top_score)
    for anchor_point in anchors:
        aPoint_coordinates = set(get_pos_coords(scores[anchor_point]))
        union_coordinates = list(set(selPoint_coordinates).union(aPoint_coordinates))
        coordinates.append(union_coordinates)

    return coordinates

def choose_diff_coordinates(top_score, scores, known_anchors, sampled_anchors, cd_params):
    '''
    :return: Array of co-ordinates to update in order of [known,sampled-anchors]
    '''
    coordinates=[]
    selPoint_coordinates = get_pos_coords(top_score)
    for anchor_point in known_anchors:
        aPoint_coordinates = set(get_pos_coords(scores[anchor_point]))
        union_coordinates = list(set(selPoint_coordinates).union(aPoint_coordinates))
        coordinates.append(union_coordinates)

    for anchor_point in sampled_anchors:
        coordinates.append(selPoint_coordinates)

    return coordinates

def choose_coordinates_negonly(scores, sel_instance, known_anchors, sampled_anchors, cd_params):
    known_anchors = np.array(known_anchors)
    sampled_anchors = np.array(sampled_anchors)

    if (cd_params['coords'] == "ALL"):
        coordinates = np.array([range(scores.shape[1])] * (known_anchors.shape[0] + sampled_anchors.shape[0]))
        return coordinates

    elif (cd_params['coords'] == "NonZero"):
        coordinates = []
        for anchor in known_anchors:
            coordinates.append(get_pos_coords(scores[sel_instance]))
        for anchor in sampled_anchors:
            coordinates.append(get_pos_coords(scores[sel_instance]))

        return coordinates

def choose_coordinates(top_score, cd_params, num_iters):
    if((cd_params['pos_coords'] == "ALL") or (cd_params['pos_coords']==None)):
        pos_coords = range(top_score.shape[0])
    if((cd_params['neg_coords'] == "ALL") or (cd_params['neg_coords']==None)):
        neg_coords = range(top_score.shape[0])
        
    if((cd_params['pos_coords'] == "Top-100") or (cd_params['neg_coords'] == "Top-100")):
        top_coords = get_top_coords(top_score, "Top-100")
        
        if(cd_params['pos_coords'] == "Top-100"):
            pos_coords = top_coords
        if(cd_params['neg_coords'] == "Top-100"):
            neg_coords = top_coords
            
    if((cd_params['pos_coords'] == "ProbExp") or (cd_params['neg_coords'] == "ProbExp")):
        prob_coords = get_prob_coords(top_score, num_iters)
        
        if(cd_params['pos_coords'] == "ProbExp"):
            pos_coords = prob_coords
        if(cd_params['neg_coords'] == "ProbExp"):
            neg_coords = prob_coords
            
    if((cd_params['pos_coords'] == "NonZero") or (cd_params['neg_coords'] == "NonZero")):
        top_coords = get_pos_coords(top_score)
        
        if(cd_params['pos_coords'] == "NonZero"):
            pos_coords = top_coords
        if(cd_params['neg_coords'] == "NonZero"):
            neg_coords = top_coords


    return pos_coords, neg_coords

def test_coordinate_selector():
    dims = 1000
    top_score =  np.zeros(1000,)
    sel_coords = np.random.choice(range(1000), p=np.ones(1000,)/1000, size=120, replace=False)
    for i in sel_coords:
        top_score[i] += np.random.random()
    
    cd_params = {}
    cd_params['pos_coords'] = 'NonZero'
    cd_params['neg_coords'] = 'NonZero'
    num_iters = 1000
    
    pos_coords, neg_coords  =  choose_coordinates(top_score, cd_params, num_iters)
    print pos_coords
    print "====="
    print neg_coords
    print "===="
    print len(set(pos_coords))
    print "===="
    print len(set(pos_coords).intersection(set(sel_coords)))/100.0
    print "===="
    print len(set(neg_coords).intersection(set(sel_coords)))/100.0
    print "====topscore"
    print np.argsort(-top_score)
    print "====topscore"
    
    
    
if __name__ == '__main__':
    test_coordinate_selector()
    
