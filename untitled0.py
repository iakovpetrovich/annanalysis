# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 18:46:53 2020

@author: jasap
"""
from pyflann import *
para = []
for tp in [0.3,0.6,0.7,0.85]:
    
    flann = FLANN()
    set_distance_type('euclidean')

    
    startClock = time.clock()
    startTime = process_time()
    flannparams = flann.build_index(train, algorithm ='autotuned', target_precision=tp, build_weight=0 ,memory_weight=0, sample_fraction=0.05)
    #params = flann.build_index(train, algorithm ='', trees = 30)
    end_time = process_time()
    constructionTime = end_time - startTime
    endClock = time.clock()
    constructionClock= endClock - startClock
    
    startClock = time.clock()
    startTime = process_time()
    result, dist = flann.nn_index(query, 100)
    end_time = process_time()
    searchTime = end_time - startTime
    endClock = time.clock()
    searchClock= endClock - startClock
    
    result = hp.fillIfNotAllAreFound(result)    
    
    result = np.asanyarray(result)
    
    rkdDflannRecall = hp.returnRecall(result, groundTruth)
    avgDist = np.mean(np.sqrt(dist))
    para.append(flannparams)
 
    reacll.append(rkdDflannRecall)
    algorithm.append(flannparams['algorithm']+'-flann-build005')
    construciotnTimes.append(constructionTime)
    searchTimes.append(searchTime)
    avgdistances.append(avgDist)
    searchClocks.append(searchClock)
    constructionClocks.append(constructionClock)
    del flann
    gc.collect()
    