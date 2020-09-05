# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:54:34 2020

@author: jasap
"""

#________LSHANALYSIS_____#
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

ar = pd.read_csv('C:/Users/jasap/.spyder-py3/annanalysis/resultCsv/LshAnaliza.csv')


a103 = ar[(ar.tables == 10) & (ar.ks == 3)]
a104 = ar[(ar.tables == 10) & (ar.ks == 4)]
a153 = ar[(ar.tables == 15) & (ar.ks == 3)]
a154 = ar[(ar.tables == 15) & (ar.ks == 4)]
a203 = ar[(ar.tables == 20) & (ar.ks == 3)]
a204 = ar[(ar.tables == 20) & (ar.ks == 4)]


from itertools import cycle
colors = ['blue','purple', 'green','yellow','orange', 'red']
colors = ['tab:red','tab:blue']
markers = ['d','d','o','o','+','+']
sets = [a103,a104,a153,a154,a203,a204]

colorC = cycle(colors)
amrkersC = cycle(markers)

#_____________searchK 
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.xlabel('Ts')
plt.ylabel('R')
for s in sets:
    plt.plot(list(s.searchTime),list(s.recall), 
         label = 'LSH t-'+str(s.tables.iloc[0])+' k-'+str(s.ks.iloc[0]), color = next(colorC), marker = next(amrkersC))

plt.legend()
plt.show()    


import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

p1 = mlines.Line2D([], [], color='tab:red', marker='d',
                          markersize=15, label='Blue stars')

plt.legend(handles=[p1])


plt.show()

p2 = mlines.Line2D([], [], color='tab:blue', marker='d',
                          markersize=15, label='Blue stars')

plt.legend([(p1, p2)], ['LSH 10'], numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})

plt.show()
