# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:30:37 2020

@author: jasap
"""
import random
import sys 
import os
sys.path.append("C:/Users/jasap/.spyder-py3/annanalysis/")
import helper as hp
from annoy import AnnoyIndex
import pandas as pd
from sklearn import preprocessing
import numpy as np
sys.path.append("C:/Users/jasap/.spyder-py3/annanalysis/")
import helper as hp
import time
from time import process_time
from itertools import chain  

trainPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_base.fvecs'
queryPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_query.fvecs'
groundPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_groundtruth.ivecs'

#trainPath = 'C:/Users/jasap/Downloads/sift.tar/sift/sift_base.fvecs'
#queryPath = 'C:/Users/jasap/Downloads/sift.tar/sift/sift_query.fvecs'
#groundPath = 'C:/Users/jasap/Downloads/sift.tar/sift/sift_groundtruth.ivecs'

train = hp.read_fvecs(trainPath)
#there is 100 querry ponts
query = hp.read_fvecs(queryPath)
#there is index number of 100 nearset n. for each querry point
groundTruth = hp.read_ivecs(groundPath)


algorithm = []
construciotnTimes=[]
searchTimes=[]
reacll = []
k = 100
avgdistances = []

#bruteForce
from sklearn.neighbors import NearestNeighbors

startTime = process_time() 
nbrs = NearestNeighbors(n_neighbors=100, algorithm='brute').fit(train)
end_time = process_time()
constructionTime = end_time - startTime

startTime = process_time() 
dist,result = nbrs.kneighbors(query,return_distance=True)
end_time = process_time()
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

startTime = process_time()
kdt = KDTree(train, metric='euclidean', leaf_size = 2)
end_time = process_time()
constructionTime = end_time - startTime

startTime = process_time()
dist, result = kdt.query(query, k=100, return_distance=True)
end_time = process_time()
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

startTime = process_time()
bt = BallTree(train, metric='euclidean')
end_time = process_time()
constructionTime = end_time - startTime

startTime = process_time()
dist, result = bt.query(query, k=100, return_distance=True)
end_time = process_time()
searchTime = end_time - startTime

ballTreeRecall = hp.returnRecall(result, groundTruth)
avgDist = np.mean(dist)
ballTreeparams = bt.get_tree_stats()
reacll.append(ballTreeRecall)
algorithm.append('ball-tree')
construciotnTimes.append(constructionTime)
searchTimes.append(searchTime)
avgdistances.append(avgDist)

#Annoy
from annoy import AnnoyIndex
f = train.shape[1]
t = AnnoyIndex(f, 'euclidean')

startTime = process_time()
for i in range(train.shape[0]):
    t.add_item(i,train[i])
t.build(15)
end_time = process_time()
constructionTime = end_time - startTime

rez = []
dist = []
startTime = process_time()
for q in query:
    res,d = t.get_nns_by_vector(q, 100, include_distances=True)
    rez.append(res)
    dist.append(d)
    #result.append(t.get_nns_by_vector(q, 100, include_distances=True))
end_time = process_time()
searchTime = end_time - startTime

    
result = hp.fillIfNotAllAreFound(rez)

result = np.asanyarray(result)
annoyRecall = hp.returnRecall(result, groundTruth)  
avgDist = np.mean(list(chain.from_iterable(dist)))

reacll.append(annoyRecall)
algorithm.append('Annoy')
construciotnTimes.append(constructionTime)
searchTimes.append(searchTime)
avgdistances.append(avgDist)


#mrpt multi RP-tree
import mrpt

target_recall = 0.91

startTime = process_time()
index = mrpt.MRPTIndex(train.astype(np.float32))
index.build_autotune_sample(target_recall, k)
end_time = process_time()
constructionTime = end_time - startTime

mrtpquery = query.astype(np.float32)

rez = []
dist = []
startTime = process_time()
for q in mrtpquery:
    res, d = index.ann(q, return_distances=True)
    rez.append(res)
    dist.append(d)
end_time = process_time()
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

#HNSW
import nmslib

hnsw = nmslib.init(method='hnsw', space='l2')

startTime = process_time()
hnsw.addDataPointBatch(train)
#hnsw.createIndex({'delaunay_type':1})
hnsw.createIndex({'delaunay_type':2, 'M':20})
end_time = process_time()
constructionTime = end_time - startTime

# get all nearest neighbours for all the datapoint
# using a pool of 4 threads to compute
startTime = process_time()
neighbours = hnsw.knnQueryBatch(query, k=100, num_threads=2)
end_time = process_time()
searchTime = end_time - startTime

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
algorithm.append('HNSW')
construciotnTimes.append(constructionTime)
searchTimes.append(searchTime)
avgdistances.append(avgDist)



#vp-tree
import nmslib
vptree = nmslib.init(method='vptree', space='l2')

startTime = process_time()
vptree.addDataPointBatch(train)
vptree.createIndex()
end_time = process_time()
constructionTime = end_time - startTime

# get all nearest neighbours for all the datapoint
# using a pool of 4 threads to compute
startTime = process_time()
neighbours = vptree.knnQueryBatch(query, k=100, num_threads=2)
end_time = process_time()
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


#flann
from pyflann import *

flann = FLANN()
set_distance_type('euclidean')

startTime = process_time()
flannparams =  flann.build_index(train, algorithm ='autotuned', target_precision=0.8, build_weight=0.02,memory_weight=0.01,sample_fraction=0.01)
#flann.build_index(train, algorithm='kdtree', trees=15)
end_time = process_time()
constructionTime = end_time - startTime

startTime = process_time()
result, dist = flann.nn_index(query, 100)
end_time = process_time()
searchTime = end_time - startTime

result = hp.fillIfNotAllAreFound(result)    

result = np.asanyarray(result)


rkdDflannRecall = hp.returnRecall(result, groundTruth)
avgDist = np.mean(np.sqrt(dist))

reacll.append(rkdDflannRecall)
algorithm.append(flannparams['algorithm']+'-flann')
construciotnTimes.append(constructionTime)
searchTimes.append(searchTime)
avgdistances.append(avgDist)

flann = FLANN()
set_distance_type('euclidean')

startTime = process_time()
flannparamsKmeans =  flann.build_index(train, algorithm='kmeans', branching=2, centersinit = 'gonzales')

end_time = process_time()
constructionTime = end_time - startTime

startTime = process_time()
result, dist = flann.nn_index(query, 100)
end_time = process_time()
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


flann = FLANN()
set_distance_type('euclidean')

startTime = process_time()
flannparamsKdtree =  flann.build_index(train, algorithm='kdtree', trees=15)

end_time = process_time()
constructionTime = end_time - startTime

startTime = process_time()
result, dist = flann.nn_index(query, 100)
end_time = process_time()
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



compareResults = pd.DataFrame({ 'algorithm':algorithm, 'constructionTime':construciotnTimes, 'searchTime':searchTimes,'recall':reacll,'avgDistance':avgdistances})

compareResults.to_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/siftSmall.csv', sep='\t' )
