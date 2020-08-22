# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 15:45:02 2020

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
from time import process_time 

#bruteForce
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=100, algorithm='brute').fit(train)
end_time = process_time()


startTime = process_time() 
result = nbrs.kneighbors(query,return_distance=False)
end_time = process_time()
end_time - startTime
bruteRecall = returnRecAll(result, groundTruth)

from sklearn.neighbors import KDTree




def createTestAndQuery(dimensions, trainPoints, queryPoints):
    testRandomMatrix = []
        
    for i in range(trainPoints):
            vector = [random.gauss(0, 1) for z in range(dimensions)]
            testRandomMatrix.append(vector)
            
    testQuery = random.sample(testRandomMatrix, 100)
    
    return testRandomMatrix, testQuery
    

dimensionsList=[]
cinstruciotnTimes=[]
searchTimes=[]
algorithm = []
for dimensions in range(2,33,2):
    
    for i in range(0,5):
        
        testRandomMatrix, testQuery = createTestAndQuery(dimensions,100000, 100)
       
        ###K-D tree 
        #Construction
        startTime = process_time()       
        kdt = KDTree(np.array(testRandomMatrix), leaf_size=30, metric='euclidean')
        end_time = process_time()
        constructionTime = end_time - startTime
        
        #query
        startTime = process_time()
        result = kdt.query( np.array(testQuery), k=100, return_distance=False)
        end_time = process_time()
        searchTime = end_time - startTime
        
        #saveResults
        dimensionsList.append(dimensions)
        cinstruciotnTimes.append(constructionTime)
        searchTimes.append(searchTime)
        algorithm.append('k-D')
        
        ###brute force
        #Construction
        startTime = process_time()   
        nbrs = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='euclidean').fit(np.array(testRandomMatrix))
        end_time = process_time()
        
        #query
        startTime = process_time()
        resultnbrs = nbrs.kneighbors( np.array(testQuery),return_distance=False)
        end_time = process_time()
        searchTime = end_time - startTime
        
        #saveResults
        dimensionsList.append(dimensions)
        cinstruciotnTimes.append(constructionTime)
        searchTimes.append(searchTime)
        algorithm.append('brute_force')



compareResults = pd.DataFrame({'Dimensions':dimensionsList, 'constructionTime':cinstruciotnTimes, 'searchTime':searchTimes, 'alg':algorithm})

grouppedResults = compareResults.groupby(["alg","Dimensions"]).agg({
        
        'Dimensions':'mean',
        'constructionTime':'mean',
        'searchTime':'mean'
        
        })

grouppedResults = grouppedResults.reset_index('alg')


import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.plot(visual[visual['alg'] == 'k-D'].searchTime, 
         label = 'k-D', color = 'tab:orange', marker = '.' )
plt.plot(visual[visual['alg'] != 'k-D'].searchTime, 
         label = 'KNN', color = 'tab:blue',marker = '.' )
plt.xticks(visual['Dimensions'].unique())
plt.xlabel('Dimenzije')
plt.ylabel('Vreme pretrage (s)')
plt.xlim([0,32])
plt.ylim([0,1.75])
plt.legend()
#plt.show()
plt.savefig('C:/Users/jasap/.spyder-py3/annanalysis/graphs/KDBruteFCurseOfDim.png',format='png')

grouppedResults.to_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/KDBruteFCurseOfDim.csv', sep='\t' )


