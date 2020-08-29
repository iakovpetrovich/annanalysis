# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 20:50:08 2020

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



import pandas as pd

df = pd.read_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/siftSmall.csv', sep='\t')


x = [6,3,8,4,5,10,9,2]
y = [2,5,4,3,7,1,6,7]

import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0,11])
plt.ylim([0,11])
plt.xticks(np.arange(0,12,1))
plt.yticks(np.arange(0,12,1))
for i in range(len(x)):
    #print('('+str(x[i])+','+str(y[i])+')')
    plt.annotate('('+str(x[i])+','+str(y[i])+')',(x[i],y[i]))

plt.show()

#________________
import matplotlib.pyplot as plt
import numpy as np
x = df['recall']
y = df['searchTime']

plt.style.use('seaborn-whitegrid')
plt.plot(color = 'tab:blue')
plt.scatter(x,y)
plt.xlabel('R')
plt.ylabel('Ts')
#plt.xlim([0,11])
#plt.ylim([0,11])
#plt.xticks(np.arange(0,12,1))
#plt.yticks(np.arange(0,12,1))
for i in [1,2,3,4,5,7,8,9]:
    #print('('+str(x[i])+','+str(y[i])+')')
    plt.annotate(df.loc[i]['algorithm'],(x[i],y[i]))
#annotate brute and vp
plt.annotate(' ',(df.loc[0]['recall'],df.loc[0]['searchTime']))
plt.annotate('vp-tree/scikit BF',(df.loc[6]['recall'],df.loc[6]['searchTime']))

plt.show()
#______________________________#

import matplotlib.pyplot as plt
import numpy as np
x = df['recall']
y = df['searchTime']

plt.style.use('seaborn-whitegrid')
plt.plot(color = 'tab:blue')
plt.scatter(x,y)
plt.xlabel('R')
plt.ylabel('Ts')
#plt.xlim([0,11])
#plt.ylim([0,11])
#plt.xticks(np.arange(0,12,1))
#plt.yticks(np.arange(0,12,1))
for i in [1,2,3,4,5,7,8,9]:
    #print('('+str(x[i])+','+str(y[i])+')')
    plt.annotate(df.loc[i]['algorithm'],(x[i],y[i]))
#annotate brute and vp
plt.annotate(' ',(df.loc[0]['recall'],df.loc[0]['searchTime']))
plt.annotate('vp-tree/scikit BF',(df.loc[6]['recall'],df.loc[6]['searchTime']))

plt.show()

#___
x = df['recall']
y = df['constructionTime']

plt.style.use('seaborn-whitegrid')
plt.plot(color = 'tab:blue')
plt.scatter(x,y)
plt.xlabel('R')
plt.ylabel('Tp')
#plt.xlim([0,11])
#plt.ylim([0,11])
#plt.xticks(np.arange(0,12,1))
#plt.yticks(np.arange(0,12,1))
for i in range(len(x)):
    #print('('+str(x[i])+','+str(y[i])+')')
    plt.annotate(df.loc[i]['algorithm'],(x[i],y[i]))
#annotate brute and vp


plt.show()



dfs = pd.read_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/sift1mBetterResults.csv', sep='\t')

Annoy = dfs[dfs['algorithm'].str.contains('Annoy')][['recall','searchTime']]
hnsw = dfs[dfs['algorithm'].str.contains('HNSW')][['recall','searchTime']]
mrpt = dfs[dfs['algorithm'].str.contains('mrpt')][['recall','searchTime']]
vp = dfs[dfs['algorithm'].str.contains('vp-')][['recall','searchTime']]

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.plot(list(Annoy.recall),list(Annoy.searchTime), 
         label = 'Annoy', color = 'tab:blue', marker = 'd' )
plt.plot(list(hnsw.recall),list(hnsw.searchTime),
         label = 'HNSW', color = 'tab:orange', marker = 'o' )
plt.plot(list(mrpt.recall),list(mrpt.searchTime), 
         label = 'mrpt', color = 'tab:red', marker = 'v' )
plt.plot(list(vp.recall),list(vp.searchTime), 
         label = 'vp-Tree', color = 'tab:purple', marker = 's' )
#plt.xticks(visual['Dimensions'].unique())
plt.xlabel('R')
plt.ylabel('Ts')

plt.legend()
plt.show()

