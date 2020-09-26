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



dfs = pd.read_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/sift1MilionToJeTo.csv', sep='\t')
#dfs.drop(axis=0,labels=19,inplace=True)

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
plt.ylim([0,250])
plt.legend()
plt.show()




#_________DATABRICKS__________#

dfs = pd.read_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/Sift1MDataBricks.csv')

Annoy = dfs[dfs['algorithm'].str.contains('Annoy')]
hnsw = dfs[dfs['algorithm'].str.contains('HNSW')]
vp = dfs[dfs['algorithm'].str.contains('vp-')]
linear = dfs[dfs['algorithm'].str.contains('linear')]
kd = dfs[dfs['algorithm'].str.contains('k-D')]
flann = dfs[dfs['algorithm'].str.contains('flann')]
lsh = dfs[dfs['algorithm'].str.contains('lsh')]


import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.plot(list(Annoy.recall),list(Annoy.searchTime), 
         label = 'Annoy', color = 'tab:blue', marker = 'd' )
plt.plot(list(hnsw.recall),list(hnsw.searchTime),
         label = 'HNSW', color = 'tab:orange', marker = 'o' )
plt.plot(list(vp.recall),list(vp.searchTime), 
         label = 'vp-Tree', color = 'tab:purple', marker = 's' )
plt.plot(list(linear.recall),list(linear.searchTime), 
         label = 'linear', color = 'black', marker = '^' )
plt.plot(list(kd.recall),list(kd.searchTime), 
         label = 'k-D', color = 'gray', marker = 'v' )
plt.plot(list(flann.recall),list(flann.searchTime), 
         label = 'kmeans-flann', color = 'yellow', marker = '+' )
plt.plot(list(lsh.recall),list(lsh.searchTime), 
         label = 'lsh', color = 'green', marker = '*' )

plt.legend()
plt.xlabel('R')
plt.ylabel('Ts')

plt.show()
#____________________________#
dfs['Qsec'] = 10000/ dfs.searchTime

Annoy = dfs[dfs['algorithm'].str.contains('Annoy')]
hnsw = dfs[dfs['algorithm'].str.contains('HNSW')]
vp = dfs[dfs['algorithm'].str.contains('vp-')]
linear = dfs[dfs['algorithm'].str.contains('linear')]
kd = dfs[dfs['algorithm'].str.contains('k-D')]
flann = dfs[dfs['algorithm'].str.contains('flann')]
lsh = dfs[dfs['algorithm'].str.contains('lsh-l11k3')]

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.plot(list(Annoy.recall),list(Annoy.Qsec), 
         label = 'Annoy', color = 'tab:blue', marker = 'd' )
plt.plot(list(hnsw.recall),list(hnsw.Qsec),
         label = 'HNSW', color = 'tab:orange', marker = 'o' )
plt.plot(list(vp.recall),list(vp.Qsec), 
         label = 'vp-Tree', color = 'tab:purple', marker = 's' )
plt.plot(list(flann.recall),list(flann.Qsec), 
         label = 'kmeans-flann', color = 'yellow', marker = '+' )
plt.plot(list(lsh.recall),list(lsh.Qsec), 
         label = 'lsh', color = 'green', marker = '*' )
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
plt.xlabel('R')
plt.ylabel('Upit/sec')

plt.show()



http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz



#_____DATABRICKS popravljen LSH_______________

dfs = pd.read_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/Sift1MDataBricksPopravljenLSH.csv')

Annoy = dfs[dfs['algorithm'].str.contains('Annoy')]
hnsw = dfs[dfs['algorithm'].str.contains('HNSW')]
vp = dfs[dfs['algorithm'].str.contains('vp-')]
linear = dfs[dfs['algorithm'].str.contains('linear')]
kd = dfs[dfs['algorithm'].str.contains('k-D')]
flann = dfs[dfs['algorithm'].str.contains('flann')]
lsh = dfs[dfs['algorithm'].str.contains('lsh')]


import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.plot(list(Annoy.recall),list(Annoy.searchTime), 
         label = 'Annoy', color = 'tab:blue', marker = 'd' )
plt.plot(list(hnsw.recall),list(hnsw.searchTime),
         label = 'HNSW', color = 'tab:orange', marker = 'o' )
plt.plot(list(vp.recall),list(vp.searchTime), 
         label = 'vp-Tree', color = 'tab:purple', marker = 's' )
plt.plot(list(linear.recall),list(linear.searchTime), 
         label = 'linear', color = 'black', marker = '^' )

plt.plot(list(flann.recall),list(flann.searchTime), 
         label = 'kmeans-flann', color = 'yellow', marker = '+' )
plt.plot(list(lsh.recall),list(lsh.searchTime), 
         label = 'lsh', color = 'green', marker = '*' )

plt.legend()
plt.xlabel('R')
plt.ylabel('Ts')

plt.show()

dfs.recall = dfs.recall.round(4)
dfs.searchTime = dfs.searchTime.round(4)
dfs.constructionTime = dfs.constructionTime.round(2)
dfs.avgDistance = dfs.avgDistance.round(2)

dfs.to_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/excel/Sift1MDataBricksPopravljenLSH.csv', sep = '\t')






dfs = compareResults

Annoy = dfs[dfs['algorithm'].str.contains('Annoy')]
hnsw = dfs[dfs['algorithm'].str.contains('HNSW')]
vp = dfs[dfs['algorithm'].str.contains('vp-')]
linear = dfs[dfs['algorithm'].str.contains('brute')]
kd = dfs[dfs['algorithm'].str.contains('k-D')]
flann = dfs[dfs['algorithm'].str.contains('kmeans')]
kdtree = dfs[dfs['algorithm'].str.contains('kdtree')]
mrpt = dfs[dfs['algorithm'].str.contains('mrpt')]
ball = dfs[dfs['algorithm'].str.contains('ball')]

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.plot(list(Annoy.recall),list(Annoy.searchTime), 
         label = 'Annoy', color = 'tab:blue', marker = 'd' )
plt.plot(list(hnsw.recall),list(hnsw.searchTime),
         label = 'HNSW', color = 'tab:orange', marker = 'o' )
plt.plot(list(vp.recall),list(vp.searchTime), 
         label = 'vp-tree', color = 'tab:purple', marker = 's' )
plt.plot(list(flann.recall),list(flann.searchTime), 
         label = 'kmeans-flann', color = 'yellow', marker = '+' )
plt.plot(list(kd.recall),list(kd.searchTime), 
         label = 'k-D', color = 'gray', marker = 'v' )
plt.plot(list(kdtree.recall),list(kdtree.searchTime), 
         label = 'rkd-flann', color = 'gray', marker = '.' )
plt.plot(list(linear.recall),list(linear.searchTime), 
         label = 'linear', color = 'black', marker = '^' )
plt.plot(list(mrpt.recall),list(mrpt.searchTime), 
         label = 'mrpt', color = 'tab:red', marker = 'v' )
plt.plot(list(ball.recall),list(ball.searchTime), 
         label = 'ball-tree', color = 'tab:blue', marker = 'o' )
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
plt.xlabel('R')
plt.ylabel('Ts')

plt.show()



import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.plot(Annoy.recall[8],Annoy.constructionTime[8], 
         label = 'Annoy', color = 'tab:blue', marker = 'd' )
plt.annotate('Annoy',(Annoy.recall[8],Annoy.constructionTime[8]))

plt.plot(hnsw.recall[17],hnsw.constructionTime[17],
         label = 'HNSW', color = 'tab:orange', marker = 'o' )
plt.annotate('HNSW',(hnsw.recall[17],hnsw.constructionTime[17]))

plt.plot(vp.recall,vp.constructionTime, 
         label = 'vp-tree', color = 'tab:purple', marker = 's' )
plt.annotate('vp-tree',(vp.recall,vp.constructionTime))


plt.plot(flann.recall[23],flann.constructionTime[23], 
         label = 'kmeans-flann', color = 'yellow', marker = '+' )
plt.annotate('kmeans-flann',(flann.recall[23],flann.constructionTime[23]))


plt.plot(kd.recall,kd.constructionTime, 
         label = 'k-D', color = 'gray', marker = 'v' )
plt.annotate('k-D',(kd.recall,kd.constructionTime))


plt.plot(kdtree.recall[29],kdtree.constructionTime[29], 
         label = 'rkd-flann', color = 'gray', marker = '.' )
plt.annotate('rkd-flann',(kdtree.recall[29],kdtree.constructionTime[29]))


plt.plot(linear.recall,linear.constructionTime, 
         label = 'linear', color = 'black', marker = '^' )
plt.annotate('linear',(linear.recall,linear.constructionTime))


plt.plot(mrpt.recall[12],mrpt.constructionTime[12], 
         label = 'mrpt', color = 'tab:red', marker = 'v' )
plt.annotate('mrpt',(mrpt.recall[12],mrpt.constructionTime[12]))


plt.plot(ball.recall,ball.constructionTime, 
         label = 'ball-tree', color = 'tab:blue', marker = 'o' )
plt.annotate('ball-tree',(ball.recall,ball.constructionTime))


#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.xlabel('R')
plt.ylabel('Tp')

plt.show()


for i in range(len(x)):
    #print('('+str(x[i])+','+str(y[i])+')')


dfs.to_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/siftSmallNeo.csv', sep='\t' )

 
