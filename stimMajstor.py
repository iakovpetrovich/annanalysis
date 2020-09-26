# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:30:37 2020

@author: jasap
"""
#import random
#import sys 
#import os
#sys.path.append("C:/Users/jasap/.spyder-py3/annanalysis/")
#import helper as hp
#from annoy import AnnoyIndex
#import pandas as pd
#from sklearn import preprocessing
#import numpy as np
#sys.path.append("C:/Users/jasap/.spyder-py3/annanalysis/")
#import helper as hp
#import time
#from time import process_time
#from itertools import chain  
#
#trainPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_base.fvecs'
#queryPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_query.fvecs'
#groundPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_groundtruth.ivecs'
#
##trainPath = 'C:/Users/jasap/Downloads/sift.tar/sift/sift_base.fvecs'
##queryPath = 'C:/Users/jasap/Downloads/sift.tar/sift/sift_query.fvecs'
##groundPath = 'C:/Users/jasap/Downloads/sift.tar/sift/sift_groundtruth.ivecs'
#
#train = hp.read_fvecs(trainPath)
##there is 100 querry ponts
#query = hp.read_fvecs(queryPath)
##there is index number of 100 nearset n. for each querry point
#groundTruth = hp.read_ivecs(groundPath)
#from time import perf_counter

algorithm = []
construciotnTimes=[]
searchTimes=[]
reacll = []
k = 100
avgdistances = []

#bruteForce
from sklearn.neighbors import NearestNeighbors

startTime = time.perf_counter()
nbrs = NearestNeighbors(n_neighbors=100, algorithm='brute').fit(train)
end_time = time.perf_counter()
constructionTime = end_time - startTime

startTime = time.perf_counter()
dist,result = nbrs.kneighbors(query,return_distance=True)
end_time = time.perf_counter()
searchTime = end_time - startTime

bruteRecall = hp.returnRecall(result, groundTruth)
avgDist = np.mean(np.mean(dist,axis=1))

reacll.append(bruteRecall)
algorithm.append('brute force')
construciotnTimes.append(constructionTime)
searchTimes.append(searchTime)
avgdistances.append(avgDist)

#KDTree test scikit learn
from sklearn.neighbors import KDTree

for lfs in [1000]:
    startTime = time.perf_counter()
    kdt = KDTree(train, metric='euclidean', leaf_size = lfs)
    end_time = time.perf_counter()
    constructionTime = end_time - startTime
    
    startTime = time.perf_counter()
    dist, result = kdt.query(query, k=100, return_distance=True)
    end_time = time.perf_counter()
    searchTime = end_time - startTime
    
    kdTreeRecall = hp.returnRecall(result, groundTruth)
    avgDist = np.mean(dist)
    ktreeparams = kdt.get_tree_stats()
    reacll.append(kdTreeRecall)
    algorithm.append('k-D')
    construciotnTimes.append(constructionTime)
    searchTimes.append(searchTime)
    avgdistances.append(avgDist)


#BallTree
from sklearn.neighbors import BallTree

for lfs in [100]:
    startTime = time.perf_counter()
    bt = BallTree(train, metric='euclidean',leaf_size = lfs)
    end_time = time.perf_counter()
    constructionTime = end_time - startTime
    
    startTime = time.perf_counter()
    dist, result = bt.query(query, k=100, return_distance=True)
    end_time = time.perf_counter()
    searchTime = end_time - startTime
    
    ballTreeRecall = hp.returnRecall(result, groundTruth)
    avgDist = np.mean(dist)
    ballTreeparams = bt.get_tree_stats()
    reacll.append(ballTreeRecall)
    algorithm.append('ball-tree')
    construciotnTimes.append(constructionTime)
    searchTimes.append(searchTime)
    avgdistances.append(avgDist)
    
    
compareResults = pd.DataFrame({ 'algorithm':algorithm, 'constructionTime':construciotnTimes, 'searchTime':searchTimes,'recall':reacll,'avgDistance':avgdistances})


#Annoy
#from annoy import AnnoyIndex
#
#for trees in [12]:
#    f = train.shape[1]
#    t = AnnoyIndex(f, 'euclidean')
#
#
#    startTime = time.perf_counter()
#    for i in range(train.shape[0]):
#        t.add_item(i,train[i])
#    t.build(trees)
#    end_time = time.perf_counter()
#    constructionTime = end_time - startTime
#    
#    rez = []
#    dist = []
#    startTime = time.perf_counter()
#    for q in query:
#        res,d = t.get_nns_by_vector(q, 100, include_distances=True)
#        rez.append(res)
#        dist.append(d)
#        #result.append(t.get_nns_by_vector(q, 100, include_distances=True))
#    end_time = time.perf_counter()
#    searchTime = end_time - startTime
#    
#        
#    result = hp.fillIfNotAllAreFound(rez)
#    
#    result = np.asanyarray(result)
#    annoyRecall = hp.returnRecall(result, groundTruth)  
#    avgDist = np.mean(list(chain.from_iterable(dist)))
#    
#    reacll.append(annoyRecall)
#    algorithm.append('Annoy')
#    construciotnTimes.append(constructionTime)
#    searchTimes.append(searchTime)
#    avgdistances.append(avgDist)
#    
#    rez = []
#    dist = []
#    startTime = time.perf_counter()
#    for q in query:
#        res,d = t.get_nns_by_vector(q, 100, include_distances=True, search_k = 2000)
#        rez.append(res)
#        dist.append(d)
#        #result.append(t.get_nns_by_vector(q, 100, include_distances=True))
#    end_time = time.perf_counter()
#    searchTime = end_time - startTime
#    
#        
#    result = hp.fillIfNotAllAreFound(rez)
#    
#    result = np.asanyarray(result)
#    annoyRecall = hp.returnRecall(result, groundTruth)  
#    avgDist = np.mean(list(chain.from_iterable(dist)))
#    
#    reacll.append(annoyRecall)
#    algorithm.append('Annoy')
#    construciotnTimes.append(constructionTime)
#    searchTimes.append(searchTime)
#    avgdistances.append(avgDist)
from annoy import AnnoyIndex

for trees in [80]:
    numTrees = trees
    
    f = train.shape[1]
    t = AnnoyIndex(f, 'euclidean')
    
    startClock= time.perf_counter()
    startTime = time.perf_counter()
    for i in range(train.shape[0]):
        t.add_item(i,train[i])
    t.build(numTrees)
    end_time = time.perf_counter()
    constructionTime = end_time - startTime
    endClock = time.perf_counter()
    constructionClock= endClock - startClock

    for search in [0.6,0.8,1,1.2,1.4,2]:
        
        
        searchK = int(search * k * numTrees)
    
    

        rez = []
        dist = []
        #startClock = time.perf_counter()
        startTime = time.perf_counter()
        for q in query:
            res,d = t.get_nns_by_vector(q, 100, search_k = searchK, include_distances=True )
            rez.append(res)
            dist.append(d)
            #result.append(t.get_nns_by_vector(q, 100, include_distances=True))
        end_time = time.perf_counter()
        searchTime = end_time - startTime
        #endClock = time.perf_counter()
        #searchClock= endClock - startClock
        
            
        result = hp.fillIfNotAllAreFound(rez)
        
        result = np.asanyarray(result)
        annoyRecall = hp.returnRecall(result, groundTruth)  
        avgDist = np.mean(list(chain.from_iterable(dist)))
        
        reacll.append(annoyRecall)
        algorithm.append('Annoy-trees-'+str(numTrees))
        construciotnTimes.append(constructionTime)
        searchTimes.append(searchTime)
        avgdistances.append(avgDist)



#mrpt multi RP-tree
import mrpt

for a in [(0.5,5),(0.6,6),(0.8,8),(0.9,10)]:
    m = a[1]
    target_recall = a[0]    

    startTime = time.perf_counter()
    index = mrpt.MRPTIndex(train.astype(np.float32))
    index.build_autotune_sample( 0.65, k,trees_max=10)
    end_time = time.perf_counter()
    constructionTime = end_time - startTime
    
    mrtpquery = query.astype(np.float32)
    
    rez = []
    dist = []
    startTime = time.perf_counter()

    for q in mrtpquery:
        res, d = index.ann(q, return_distances=True)
        rez.append(res)
        dist.append(d)
    end_time = time.perf_counter()
    searchTime = end_time - startTime
    mrptparams = index.parameters()    
    result = hp.fillIfNotAllAreFound(rez)   
    
    result = np.asanyarray(result)
    mrptreeRecall = hp.returnRecall(result, groundTruth)
    avgDist = np.mean(list(chain.from_iterable(dist)))
    
    reacll.append(mrptreeRecall)
    algorithm.append('mrpt')
    construciotnTimes.append(constructionTime)
    searchTimes.append(searchTime)
    avgdistances.append(avgDist)


"""
NMSLIB
They have implemented a few algs
Emphasizing HNSW as most successful 
Modifications of VP and NAPP are introduced
hnsw, sw-graph, vp-tree, napp, simple_invindx,brute_force 
"""
#
##HNSW
#import nmslib
#
#
#for m in [2,6,12]:
#    hnsw = nmslib.init(method='hnsw', space='l2')
#    startTime = time.perf_counter()
#    hnsw.addDataPointBatch(train)
#    #hnsw.createIndex({'delaunay_type':1})
#    hnsw.createIndex({'delaunay_type':1, 'M':m})
#    end_time = time.perf_counter()
#    constructionTime = end_time - startTime
#    
#    # get all nearest neighbours for all the datapoint
#    # using a pool of 4 threads to compute
#    startTime = time.perf_counter()
#    neighbours = hnsw.knnQueryBatch(query, k=100, num_threads=2)
#    end_time = time.perf_counter()
#    searchTime = end_time - startTime
#    
#    rez =[]
#    dist =[]
#    for i in neighbours:
#        rez.append(list(i[0]))
#        dist.append(list(i[1]))
#    
#    result = hp.fillIfNotAllAreFound(rez)
#      
#    result = np.array(rez)
#    hnswRecall = hp.returnRecall(result, groundTruth)
#    avgDist = np.mean(np.sqrt(list(chain.from_iterable(dist))))
#    
#    reacll.append(hnswRecall)
#    algorithm.append('HNSW'+str(m))
#    construciotnTimes.append(constructionTime)
#    searchTimes.append(searchTime)
#    avgdistances.append(avgDist)
#    
#    
#    hnsw.setQueryTimeParams({'efSearch': 50})
#    startTime = time.perf_counter()
#    neighbours = hnsw.knnQueryBatch(query, k=100, num_threads=2)
#    end_time = time.perf_counter()
#    searchTime = end_time - startTime
#    
#    rez =[]
#    dist =[]
#    for i in neighbours:
#        rez.append(list(i[0]))
#        dist.append(list(i[1]))
#    
#    result = hp.fillIfNotAllAreFound(rez)
#      
#    result = np.array(rez)
#    hnswRecall = hp.returnRecall(result, groundTruth)
#    avgDist = np.mean(np.sqrt(list(chain.from_iterable(dist))))
#    
#    reacll.append(hnswRecall)
#    algorithm.append('HNSW')
#    construciotnTimes.append(constructionTime)
#    searchTimes.append(searchTime)
#    avgdistances.append(avgDist)
#
#    
#compareResults = pd.DataFrame({ 'algorithm':algorithm, 'constructionTime':construciotnTimes, 'searchTime':searchTimes,'recall':reacll,'avgDistance':avgdistances})
#
Mparams = []
efParams = []
Mresults = []

import nmslib
for MMAX in [3]:
    hnsw = nmslib.init(method='hnsw', space='l2')
    

    startClock = time.perf_counter()
    startTime = time.perf_counter()
    hnsw.addDataPointBatch(train)
    hnsw.createIndex({'delaunay_type':1, 'M':MMAX})
    end_time = time.perf_counter()
    constructionTime = end_time - startTime
    endClock = time.perf_counter()
    constructionClock= endClock - startClock

    
    
    
    for efParam in [40,70,90,110,150]:
        # get all nearest neighbours for all the datapoint
        # using a pool of 4 threads to compute
        startClock = time.perf_counter()
        startTime = time.perf_counter()
        hnsw.setQueryTimeParams({'efSearch': efParam})
        neighbours = hnsw.knnQueryBatch(query, k=100)
        end_time = time.perf_counter()
        searchTime = end_time - startTime
        endClock = time.perf_counter()
        searchClock= endClock - startClock

        rez =[]
        dist =[]
        for i in neighbours:
            rez.append(list(i[0]))
            dist.append(list(i[1]))

        result = hp.fillIfNotAllAreFound(rez)

        result = np.array(rez)
        hnswRecall = hp.returnRecall(result, groundTruth)
        avgDist = np.mean(np.sqrt(list(chain.from_iterable(dist))))

        reacll.append(hnswRecall)
        algorithm.append('HNSW-M-'+str(MMAX))
        construciotnTimes.append(constructionTime)
        searchTimes.append(searchTime)
        avgdistances.append(avgDist)
        
        


##vp-tree
#import nmslib
#vptree = nmslib.init(method='vptree', space='l2')
#
# 
#startTime = time.perf_counter()
#vptree.addDataPointBatch(train)
#vptree.createIndex({'bucketSize' : 200,'selectPivotAttempts':10})
#end_time = time.perf_counter()
#constructionTime = end_time - startTime
#
#
#
#for maxL in [2,5,10,25,40]:
#    vptree.setQueryTimeParams({'maxLeavesToVisit':maxL,'alphaLeft':1.1,'alphaRight':1.1})
#    
#    # get all nearest neighbours for all the datapoint
#    # using a pool of 4 threads to compute
#    startTime = time.perf_counter()
#    neighbours = vptree.knnQueryBatch(query, k=100, num_threads=2)
#    end_time = time.perf_counter()
#    searchTime = end_time - startTime
#    
#    rez =[]
#    dist = []
#    for i in neighbours:
#        rez.append(list(i[0]))
#        dist.append(list(i[1]))
#        
#    rez = hp.fillIfNotAllAreFound(rez)    
#    
#    result = np.asanyarray(rez)
#    vptreeR = result.copy()
#    vpdist = dist.copy()
#    
#    vptreeRecall = hp.returnRecall(result, groundTruth)
#    avgDist = np.mean(list(chain.from_iterable(dist)))
#    
#    reacll.append(vptreeRecall)
#    algorithm.append('vp-Tree')
#    construciotnTimes.append(constructionTime)
#    searchTimes.append(searchTime)
#    avgdistances.append(avgDist)
import nmslib
vptree = nmslib.init(method='vptree', space='l2')

startTime = time.perf_counter()
vptree.addDataPointBatch(train)
vptree.createIndex({'bucketSize' : 100,'selectPivotAttempts':10})
end_time = time.perf_counter()
constructionTime = end_time - startTime

# get all nearest neighbours for all the datapoint
# using a pool of 4 threads to compute
startTime = time.perf_counter()
neighbours = vptree.knnQueryBatch(query, k=100, num_threads=2)
end_time = time.perf_counter()
searchTime = end_time - startTime

rez =[]
dist = []
for i in neighbours:
    rez.append(list(i[0]))
    dist.append(list(i[1]))
    
rez = hp.fillIfNotAllAreFound(rez)    

result = np.asanyarray(rez)
vptreeR = result.copy()
vpdist = dist.copy()

vptreeRecall = hp.returnRecall(result, groundTruth)
avgDist = np.mean(list(chain.from_iterable(dist)))

reacll.append(vptreeRecall)
algorithm.append('vp-Tree')
construciotnTimes.append(constructionTime)
searchTimes.append(searchTime)
avgdistances.append(avgDist)



#____________________________OVO__________________


 
 


from pyflann import *
flann = FLANN()
set_distance_type('euclidean')


startTime = time.perf_counter()
flannparamsKmeans  =  flann.build_index(train, algorithm ='kmeans', 
                                 branching = 2,
                                 iterrations = -1,
                                 #cb_index = 0.2,
                                 #leaf_max_size = 10,
                                 #trees = 15
                                 leaf_max_size= 40
                                 )
        
end_time = time.perf_counter()
constructionTime = end_time - startTime
#flannparamsKmeans =  flann.build_index(train, algorithm='kmeans', branching=2, centersinit = 'gonzales')

for c in [10,80,100,120,130]:

    
    startTime = time.perf_counter()
    result, dist = flann.nn_index(query, 100, checks=c)
    end_time = time.perf_counter()
    searchTime = end_time - startTime
    
    result = hp.fillIfNotAllAreFound(result)    
    
    result = np.asanyarray(result)
    
    
    rkdDflannRecall = hp.returnRecall(result, groundTruth)
    avgDist = np.mean(np.sqrt(dist))
    
    reacll.append(rkdDflannRecall)
    algorithm.append(flannparamsKmeans['algorithm']+'-flann')
    construciotnTimes.append(constructionTime)
    searchTimes.append(searchTime)
    avgdistances.append(avgDist)


compareResults = pd.DataFrame({ 'algorithm':algorithm, 'constructionTime':construciotnTimes, 'searchTime':searchTimes,'recall':reacll,'avgDistance':avgdistances})

 #_____________________________________________________#   



 
 #__________________#______________________#



for numT in [1,2,3,4,8,16,]:
    flann = FLANN()
    set_distance_type('euclidean')
    startTime = time.perf_counter()
    flannparamsKdtree =  flann.build_index(train, algorithm='kdtree', trees=numT, leaf_max_size= 100)
    
    end_time = time.perf_counter()
    constructionTime = end_time - startTime
    
    startTime = time.perf_counter()
    result, dist = flann.nn_index(query, 100, checks = 400)
    end_time = time.perf_counter()
    searchTime = end_time - startTime
    
    result = hp.fillIfNotAllAreFound(result)    
    
    result = np.asanyarray(result)
    
    
    rkdDflannRecall = hp.returnRecall(result, groundTruth)
    avgDist = np.mean(np.sqrt(dist))
    
    reacll.append(rkdDflannRecall)
    algorithm.append(flannparamsKdtree['algorithm']+'-flann')
    construciotnTimes.append(constructionTime)
    searchTimes.append(searchTime)
    avgdistances.append(avgDist)


compareResults = pd.DataFrame({ 'algorithm':algorithm, 'constructionTime':construciotnTimes, 'searchTime':searchTimes,'recall':reacll})

 

#compareResults.to_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/siftSmall.csv', sep='\t' )


