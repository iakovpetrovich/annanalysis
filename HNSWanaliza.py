# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 22:05:27 2020

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
import math 
import gc

trainPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_base.fvecs'
queryPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_query.fvecs'
groundPath = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_groundtruth.ivecs'


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
MMAXparam = []
dgraphParam = []
constructionClocks = []
searchClocks = []
clockAlg = []

import nmslib

for example in  [(dgraph, MMAX) for dgraph in [0,1,2,3] for MMAX in [2,4,6,8,10,12]]:
    
    hnsw = nmslib.init(method='hnsw', space='l2')
    
    dgraph = example[0]
    MMAX = example[1]
    
    MMAXparam.append(example[1])
    dgraphParam.append(example[0])
    
    startClock = time.clock()
    startTime = process_time()
    hnsw.addDataPointBatch(train)
    hnsw.createIndex({'delaunay_type':dgraph, 'M':MMAX})
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
    
    result = hp.fillIfNotAllAreFound(rez)
      
    result = np.array(rez)
    hnswRecall = hp.returnRecall(result, groundTruth)
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
    
    
    

hr = pd.DataFrame({ 'algorithm':algorithm, 
                               'constructionTime':construciotnTimes, 
                               'searchTime':searchTimes,
                               'recall':reacll,
                               'avgDistance':avgdistances,
                               'MMAXparam':MMAXparam,
                               'dgraphParam':dgraphParam,
                               'constructionClocks':constructionClocks,
                               'searchClocks':searchClocks,
                               })

    
    

a1 = hr[hr.dgraphParam == 0]
a5 = hr[hr.dgraphParam == 1]
a10 = hr[hr.dgraphParam == 2]
a30 = hr[hr.dgraphParam == 3]

import matplotlib.pyplot as plt

plt.xlabel('M-param')
plt.ylabel('R')

plt.plot(list(a1.MMAXparam),list(a1.recall), 
         label = 'delaunay_type 0', color = 'gold', marker = 'o' )
plt.plot(list(a5.MMAXparam),list(a5.recall),
         label = 'delaunay_type 1', color = 'goldenrod', marker = 'o' )
plt.plot(list(a10.MMAXparam),list(a10.recall), 
         label = 'delaunay_type 2', color = 'orange', marker = 'o' )
plt.plot(list(a30.MMAXparam),list(a30.recall), 
         label = 'delaunay_type 3', color = 'darkorange', marker = 'o' )
#plt.xticks(visual['Dimensions'].unique())
plt.legend()

plt.show()

hr.to_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/HNSWAnaliza.csv', sep='\t' )