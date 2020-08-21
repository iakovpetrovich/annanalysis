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
        numTN = len(set(result[i].tolist()) & set(test[i].tolist()))
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
kdTreeRecall = hp.returnRecAll(result, groundTruth)

#Testing K-D tree capabilities in different dimensions
#Trying to produce "CURSE of dimensionality" but it is only linear growth
results = pd.DataFrame({'Dimensions':[0], 'constructionTime':[0.0], 'searchTime':[0.0]})

dimensionsList=[]
cinstruciotnTimes=[]
searchTimes=[]
for dimensions in range(5,126,5):
    
    for i in range(0,5):
        
        testRandomMatrix = []
        for i in range(10000):
            vector = [random.gauss(0, 1) for z in range(dimensions)]
            testRandomMatrix.append(vector)
            
        testQuery = random.sample(testRandomMatrix, 100)
       
            
        startTime = process_time()       
        kdt = KDTree(np.array(testRandomMatrix), leaf_size=30, metric='euclidean')
        end_time = process_time()
        constructionTime = end_time - startTime
            
        startTime = process_time()
        result = kdt.query( np.array(testQuery), k=100, return_distance=False)
        end_time = process_time()
        searchTime = end_time - startTime
        
        dimensionsList.append(dimensions)
        cinstruciotnTimes.append(constructionTime)
        searchTimes.append(searchTime)

 results3= pd.DataFrame({'Dimensions':dimensionsList, 'constructionTime':cinstruciotnTimes, 'searchTime':searchTimes})  
results = pd.DataFrame({'Dimensions':dimensionsList, 'constructionTime':cinstruciotnTimes, 'searchTime':searchTimes})
results2 = pd.DataFrame({'Dimensions':dimensionsList, 'constructionTime':cinstruciotnTimes, 'searchTime':searchTimes}) 
 
   
rezultati = results.deep_copy()

totalResults = results.append(results2)

totalResults.groupby(["Dimensions"]).agg({
        
        'Dimensions':'mean',
        'constructionTime':'mean',
        'searchTime':'mean'
        
        })


import seaborn as sns

import matplotlib.pyplot as plt

ax = sns.lineplot(x="Dimensions", y="searchTime",  dashes=True, data=totalResults)

totalResults.to_csv('C:/Users/jasap/.spyder-py3/annanalysis/TryingToProduceCurseOfDimensionality.csv')

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


def returnTrueNeighbours(result, test):
    TrueNeighbours = []
    for i in range(result.shape[0]):
        TN = list(set(result[i].tolist()) & set(test[i].tolist()))
        TrueNeighbours.append(TN)
    return TrueNeighbours

tn = returnTrueNeighbours(result, groundTruth)
len(tn[43])

"""
NMSLIP 
They have implemented a few algs
Emphasizing HNSW as most successful 
Modifications of VP and NAPP are introduced
hnsw, sw-graph, vp-tree, napp, simple_invindx,brute_force 
"""

#HNSW
import nmslib

hnsw = nmslib.init(method='hnsw', space='l2')
hnsw.addDataPointBatch(train)

startTime = process_time()
hnsw.createIndex({'post': 2}, print_progress=True)
end_time = process_time()
end_time - startTime

# query for the nearest neighbours of the first datapoint
ids, distances = index.knnQuery(data[0], k=10 )

# get all nearest neighbours for all the datapoint
# using a pool of 4 threads to compute
startTime = process_time()
neighbours = hnsw.knnQueryBatch(query, k=100, num_threads=2)
end_time = process_time()
end_time - startTime


rez =[]
for i in neighbours:
    print(list(i[0]))
    rez.append(list(i))
result = np.asanyarray(rez)
len(neighbours[0][1])
len(neighbours)
returnRecAll(result, groundTruth)


result = []
startTime = process_time()
for q in query:
    result.append((hnsw.knnQuery(q, k=100))[0])
end_time = process_time()
end_time - startTime


rez = []
for i in result:
    rez.append(list(i))

result = np.asanyarray(rez)


returnRecAll(result, groundTruth)









