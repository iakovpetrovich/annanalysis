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

#trainPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_base.fvecs'
#queryPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_query.fvecs'
#groundPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_groundtruth.ivecs'

trainPath = 'C:/Users/jasap/Downloads/sift.tar/sift/sift_base.fvecs'
queryPath = 'C:/Users/jasap/Downloads/sift.tar/sift/sift_query.fvecs'
groundPath = 'C:/Users/jasap/Downloads/sift.tar/sift/sift_groundtruth.ivecs'

train = read_fvecs(trainPath)
#there is 100 querry ponts
query = read_fvecs(queryPath)
#there is index number of 100 nearset n. for each querry point
groundTruth = read_ivecs(groundPath)

import gc
gc.collect()


algorithm = []
construciotnTimes=[]
searchTimes=[]
reacll = []
k = 100
avgdistances = []

constructionClocks = []
searchClocks = []
clockAlg = []
##vp-tree

import nmslib
vptree = nmslib.init(method='vptree', space='l2')

startTime = process_time()
vptree.addDataPointBatch(train)
vptree.createIndex({'bucketSize' : 10000,'selectPivotAttempts':10})
end_time = process_time()
constructionTime = end_time - startTime

# get all nearest neighbours for all the datapoint
# using a pool of 4 threads to compute
for maxLeave in [30]:#[2,10,15,20,25]:
  
    vptree.setQueryTimeParams({'maxLeavesToVisit':maxLeave,'alphaLeft':1.1,'alphaRight':1.1})
    startTime = process_time()
    neighbours = vptree.knnQueryBatch(query,k=100, num_threads=2 )
    end_time = process_time()
    searchTime = end_time - startTime
    
    rez =[]
    dist = []
    for i in neighbours:
        rez.append(list(i[0]))
        dist.append(list(i[1]))
        
    rez = fillIfNotAllAreFound(rez)    
    
    result = np.asanyarray(rez)
    
    vptreeRecall = returnRecall(result, groundTruth)
    avgDist = np.mean(list(chain.from_iterable(dist)))
    
    reacll.append(vptreeRecall)
    algorithm.append('vp-Tree-10k-mL'+str(maxLeave))
    #algorithm.append('vp-Tree-maxLeaves'+str(maxLeaves))
    construciotnTimes.append(constructionTime)
    searchTimes.append(searchTime)
    avgdistances.append(avgDist)
    del rez
    del dist
    del result
    gc.collect()

#vptree.saveIndex('vptreeIndex.ann')    
del vptree
gc.collect()


#HNSW da bi islo po redu
#______________________________________________#

for MMAX in [40]:#[2,5,8,15,30]:
    hnsw = nmslib.init(method='hnsw', space='l2')
    
    startClock = time.clock()
    startTime = process_time()
    hnsw.addDataPointBatch(train)
    hnsw.createIndex({'delaunay_type':0, 'M':MMAX})
    end_time = process_time()
    constructionTime = end_time - startTime
    endClock = time.clock()
    constructionClock= endClock - startClock
    
    
    
    # get all nearest neighbours for all the datapoint
    # using a pool of 4 threads to compute
    startClock = time.clock()
    startTime = process_time()
    neighbours = hnsw.knnQueryBatch(query, k=100, num_threads=2)
    end_time = process_time()
    searchTime = end_time - startTime
    endClock = time.clock()
    searchClock= endClock - startClock
    
    rez =[]
    dist =[]
    for i in neighbours:
        rez.append(list(i[0]))
        dist.append(list(i[1]))
    
    result = fillIfNotAllAreFound(rez)
      
    result = np.array(rez)
    hnswRecall = returnRecall(result, groundTruth)
    avgDist = np.mean(np.sqrt(list(chain.from_iterable(dist))))
    
    reacll.append(hnswRecall)
    algorithm.append('HNSW-M-'+str(MMAX))
    construciotnTimes.append(constructionTime)
    searchTimes.append(searchTime)
    avgdistances.append(avgDist)
    constructionClocks.append(constructionClock)
    searchClocks.append(searchClock)
    clockAlg.append('HNSW-M-'+str(MMAX))
    hnsw.saveIndex('hnswIndex40.ann')
    del hnsw
    del rez
    del dist
    del result
    del neighbours
    gc.collect()

#______________________________________________#    
#annoy
#Annoy
from annoy import AnnoyIndex
for trs in [90]:#[5,15,30,45,60,80]:
    
    f = train.shape[1]
    t = AnnoyIndex(f, 'euclidean')
    
    startClock= time.clock()
    startTime = process_time()
    for i in range(train.shape[0]):
        t.add_item(i,train[i])
    t.build(trs)
    end_time = process_time()
    constructionTime = end_time - startTime
    endClock = time.clock()
    constructionClock= endClock - startClock
    
    
    rez = []
    dist = []
    startClock = time.clock()
    startTime = process_time()
    for q in query:
        res,d = t.get_nns_by_vector(q, 100, include_distances=True)
        rez.append(res)
        dist.append(d)
        #result.append(t.get_nns_by_vector(q, 100, include_distances=True))
    end_time = process_time()
    searchTime = end_time - startTime
    endClock = time.clock()
    searchClock= endClock - startClock
    
        
    result = fillIfNotAllAreFound(rez)
    
    result = np.asanyarray(result)
    annoyRecall = returnRecall(result, groundTruth)  
    avgDist = np.mean(list(chain.from_iterable(dist)))
    
    reacll.append(annoyRecall)
    algorithm.append('Annoy-trees-'+str(trs))
    construciotnTimes.append(constructionTime)
    searchTimes.append(searchTime)
    avgdistances.append(avgDist)
    searchClocks.append(searchClock)
    constructionClocks.append(constructionClock)
    clockAlg.append('Annoy-trees-'+str(trs))
    t.save('annoyIndex90.ann')
    del t
    del rez
    del dist
    del result
    gc.collect()
#______________________________________________#
nesrecni mrpt

import mrpt

params = []
for target_recall,mt in [(0.9,650)]:#[(0.4,30),(0.65,100),(0.8,300),(0.9,500)]:

    
    startTime = process_time()
    index = mrpt.MRPTIndex(train.astype(np.float32))
    index.build_autotune_sample(target_recall, k, trees_max=mt)
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
        
    result = fillIfNotAllAreFound(rez)   
    
    result = np.asanyarray(result)
    mrptreeRecall = returnRecall(result, groundTruth)
    avgDist = np.mean(list(chain.from_iterable(dist)))
    
    reacll.append(mrptreeRecall)
    algorithm.append('mrpt-'+str(target_recall)+'mt'+str(mt))
    construciotnTimes.append(constructionTime)
    searchTimes.append(searchTime)
    avgdistances.append(avgDist)
    params.append(index.parameters())
    #index.save('mrpt700.ann')
    del index
    del rez
    del dist
    del result
    gc.collect()


     

    
    

#flann
from pyflann import *
para = []
for tp in [0.6,0.8,0.9]:
    
    flann = FLANN()
    set_distance_type('euclidean')
    gc.collect()
    
    startClock = time.clock()
    startTime = process_time()
    flannparams = flann.build_index(train, algorithm ='autotuned', target_precision=tp, build_weight=0.005,memory_weight=0,sample_fraction=0.001)
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
    
    result = fillIfNotAllAreFound(result)    
    
    result = np.asanyarray(result)
    
    rkdDflannRecall = returnRecall(result, groundTruth)
    avgDist = np.mean(np.sqrt(dist))
    para.append(flannparams)
 
    reacll.append(rkdDflannRecall)
    algorithm.append(flannparams['algorithm']+'-flann-build005')
    construciotnTimes.append(constructionTime)
    searchTimes.append(searchTime)
    avgdistances.append(avgDist)
    searchClocks.append(searchClock)
    constructionClocks.append(constructionClock)
    clockAlg.append(flannparams['algorithm']+'-flann-build005')
    

compareResults = pd.DataFrame({ 'algorithm':algorithm, 'constructionTime':construciotnTimes, 'searchTime':searchTimes,'recall':reacll,'avgDistance':avgdistances})

compareResults.to_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/sift1MilionToJeTo.csv', sep='\t' )

clocktimes = pd.DataFrame({ 'algorithm':clockAlg, 'ClockConstructionTime':constructionClocks, 'clockSearchTime':searchClocks})
clocktimes.to_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/sift1MilionClockTimeZaNajbolje.csv', sep='\t' )
#
#def load_mnist(path, kind='train'):
#    import os
#    import gzip
#    import numpy as np
#
#    """Load MNIST data from `path`"""
#    labels_path = os.path.join(path,
#                               '%s-labels-idx1-ubyte.gz'
#                               % kind)
#    images_path = os.path.join(path,
#                               '%s-images-idx3-ubyte.gz'
#                               % kind)
#
#    with gzip.open(labels_path, 'rb') as lbpath:
#        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
#                               offset=8)
#
#    with gzip.open(images_path, 'rb') as imgpath:
#        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
#                               offset=16).reshape(len(labels), 784)
#
#    return images, labels
#
#X_train, y_train = load_mnist('C:/Users/jasap/Downloads', kind='train')
#X_test, y_test = load_mnist('C:/Users/jasap/Downloads', kind='t10k')
#
#del X_train
#del X_test
#gc.collect()
#del X_train
#del X_train

