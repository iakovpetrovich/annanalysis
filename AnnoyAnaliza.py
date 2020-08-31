# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 20:56:19 2020

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
searchKparam = []
numTreesParam = []
constructionClocks = []
searchClocks = []
clockAlg = []

for example in [(trees, search) for trees in [1,5,10,30] for search in [0.8,0.9,1.1,1.2]]:
    numTrees = example[0]
    searchK = int(example[1] * k * numTrees)
    
    
    numTreesParam.append(numTrees)
    searchKparam.append(searchK)
    
    f = train.shape[1]
    t = AnnoyIndex(f, 'euclidean')
    
    startClock= time.clock()
    startTime = process_time()
    for i in range(train.shape[0]):
        t.add_item(i,train[i])
    t.build(numTrees)
    end_time = process_time()
    constructionTime = end_time - startTime
    endClock = time.clock()
    constructionClock= endClock - startClock
    
    
    rez = []
    dist = []
    startClock = time.clock()
    startTime = process_time()
    for q in query:
        res,d = t.get_nns_by_vector(q, 100, search_k = searchK, include_distances=True)
        rez.append(res)
        dist.append(d)
        #result.append(t.get_nns_by_vector(q, 100, include_distances=True))
    end_time = process_time()
    searchTime = end_time - startTime
    endClock = time.clock()
    searchClock= endClock - startClock
    
        
    result = hp.fillIfNotAllAreFound(rez)
    
    result = np.asanyarray(result)
    annoyRecall = hp.returnRecall(result, groundTruth)  
    avgDist = np.mean(list(chain.from_iterable(dist)))
    
    reacll.append(annoyRecall)
    algorithm.append('Annoy-trees-'+str(numTrees))
    construciotnTimes.append(constructionTime)
    searchTimes.append(searchTime)
    avgdistances.append(avgDist)
    searchClocks.append(searchClock)
    constructionClocks.append(constructionClock)
    clockAlg.append('Annoy-trees-'+str(numTrees))
    t.save('annoyIndex90.ann')
    del t
    del rez
    del dist
    del result
    gc.collect()
    
    

ar = pd.DataFrame({ 'algorithm':algorithm, 
                               'constructionTime':construciotnTimes, 
                               'searchTime':searchTimes,
                               'recall':reacll,
                               'avgDistance':avgdistances,
                               'numTreesParam':numTreesParam,
                               'searchKparam':searchKparam,
                               'constructionClocks':constructionClocks,
                               'searchClocks':searchClocks,
                               })


a1 = ar[ar.numTreesParam == 1]
a5 = ar[ar.numTreesParam == 5]
a10 = ar[ar.numTreesParam == 10]
a30 = ar[ar.numTreesParam == 30]

import matplotlib.pyplot as plt

plt.xlabel('searck_k')
plt.ylabel('R')

plt.plot(list(a1.searchKparam),list(a1.recall), 
         label = 'Annoy 1', color = 'blue', marker = 'd' )
plt.plot(list(a5.searchKparam),list(a5.recall),
         label = 'Annoy 5', color = 'mediumblue', marker = 'd' )
plt.plot(list(a10.searchKparam),list(a10.recall), 
         label = 'Annoy 10', color = 'darkblue', marker = 'd' )
plt.plot(list(a30.searchKparam),list(a30.recall), 
         label = 'Annoy 30', color = 'navy', marker = 'd' )
#plt.xticks(visual['Dimensions'].unique())
plt.legend()

plt.show()


#
#import matplotlib.pyplot as plt
#
#
#plt.xlabel('R')
#plt.ylabel('Ts')
#plt.legend()
#plt.plot(list(a1.recall),list(a1.searchTime), 
#         label = 'Annoy 1', color = 'blue', marker = 'd' )
#plt.plot(list(a5.recall),list(a5.searchTime),
#         label = 'Annoy 5', color = 'mediumblue', marker = 'd' )
#plt.plot(list(a10.recall),list(a10.searchTime), 
#         label = 'Annoy 10', color = 'darkblue', marker = 'd' )
#plt.plot(list(a30.recall),list(a30.searchTime), 
#         label = 'Annoy 15', color = 'navy', marker = 'd' )
##plt.xticks(visual['Dimensions'].unique())
#
#plt.show()
#
#
#
#
#import matplotlib.pyplot as plt
#
#plt.xlabel('R')
#plt.ylabel('Ts')
#
#plt.plot(list(a1.recall),list(a1.searchClocks), 
#         label = 'Annoy 1', color = 'blue', marker = 'd' )
#plt.plot(list(a5.recall),list(a5.searchClocks),
#         label = 'Annoy 5', color = 'mediumblue', marker = 'd' )
#plt.plot(list(a10.recall),list(a10.searchClocks), 
#         label = 'Annoy 10', color = 'darkblue', marker = 'd' )
#plt.plot(list(a30.recall),list(a30.searchClocks), 
#         label = 'Annoy 15', color = 'navy', marker = 'd' )
#plt.legend()
#
#plt.show()
#
#
#
#
#import matplotlib.pyplot as plt
#
#plt.xlabel('R')
#plt.ylabel('Ts')
#plt.legend()
#plt.plot(list(a1.searchClocks),list(a1.searchKparam), 
#         label = 'Annoy 1', color = 'blue', marker = 'd' )
#plt.plot(list(a5.searchClocks),list(a5.searchKparam),
#         label = 'Annoy 5', color = 'mediumblue', marker = 'd' )
#plt.plot(list(a10.searchClocks),list(a10.searchKparam), 
#         label = 'Annoy 10', color = 'darkblue', marker = 'd' )
#plt.plot(list(a30.searchClocks),list(a30.searchKparam), 
#         label = 'Annoy 15', color = 'navy', marker = 'd' )
##plt.xticks(visual['Dimensions'].unique())
#plt.legend()
#
#plt.show()
#
#
#ar.to_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/AnnoyAnaliza.csv', sep='\t' )
#
#
#
#
#






