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

trainPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_base.fvecs'
queryPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_query.fvecs'
groundPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_groundtruth.ivecs'

train = hp.read_fvecs(trainPath)
#there is 100 querry ponts
query = hp.read_fvecs(queryPath)
#there is index number of 100 nearset n. for each querry point
groundTruth = hp.read_ivecs(groundPath)

def returnRecAll(result, test):
    numOfTrueNeighbours = []
    #for every result vector we check how many right neighbours were identified
    for i in range(result.shape[0]):
        numTN = len(set(result[i].tolist()) & set(groundTruth[i].tolist()))
        numOfTrueNeighbours.append(numTN)
        recall = sum(numOfTrueNeighbours) /result.size
    return recall

#bruteForce
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=100, algorithm='brute').fit(train)
end_time = process_time()


startTime = process_time() 
result = nbrs.kneighbors(query,return_distance=False)
end_time = process_time()
end_time - startTime
bruteRecall = returnRecAll(result, groundTruth)


#KDTree test scikit learn
from sklearn.neighbors import KDTree
kdt = KDTree(train, leaf_size=30, metric='euclidean')
startTime = process_time()
result = kdt.query(query, k=100, return_distance=False)
end_time = process_time()
end_time - startTime
kdTreeRecall = returnRecAll(result, groundTruth)


#BallTree
from sklearn.neighbors import BallTree
bt = BallTree(train, leaf_size=4, metric='euclidean')
startTime = process_time()
result = bt.query(query, k=100, return_distance=False)
end_time = process_time()
end_time - startTime


result = []
startTime = process_time()
for q in query:
    result.append(bt.query([q], k=100, return_distance=False))
end_time = process_time()
end_time - startTime

rez =[]
for i in result:
    rez.append(list(i[0]))
result = np.asanyarray(rez)

ballTreeRecall = returnRecAll(result, groundTruth)

#Annoy
from annoy import AnnoyIndex
f = train.shape[1]
t = AnnoyIndex(f, 'euclidean')

for i in range(train.shape[0]):
    t.add_item(i,train[i])

t.build(100)

result = []
startTime = process_time()
for q in query:
    result.append(t.get_nns_by_vector(q, 100, search_k=-1, include_distances=False))
end_time = process_time()
end_time - startTime

result = np.asanyarray(result)
annoyRecall = returnRecAll(result, groundTruth)    

#SomeOtherApproach















