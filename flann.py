# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 00:15:36 2020

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



from pyflann import *
para = []

for tp in [0.6,0.8,0.9]:
    
    flann = FLANN()
    set_distance_type('euclidean')
    gc.collect()
    
    startClock = time.clock()
    startTime = process_time()
    #flannparams = flann.build_index(train, algorithm ='kmeans', build_weight=1,memory_weight=0.1, centers_init = 'random',sample_fraction=0.1)
    #flannparams = flann.build_index(train, algorithm ='autotuned', target_precision=tp, build_weight=0 ,memory_weight=0, sample_fraction=0.05)
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
    clockAlg.append(flannparams['algorithm']+'-flann-build005')
    