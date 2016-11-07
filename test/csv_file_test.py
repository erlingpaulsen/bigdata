# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:34:07 2016

@author: ERLING
"""
from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import filehandler as fh
from sklearn.cluster import KMeans

csv_path = "../data/unzipped/KMFILES/"
ex = "*.csv"

files = fh.filesInDir(csv_path,ex)

def loadData(path, filename):
        df = pd.read_csv(path+filename, sep=',', nrows=1000)
        return df

def toFloat(x, notFloat = np.nan):
    try:
        float(x)
        return float(x)
    except ValueError:
        return notFloat
        
def findFrequency(col):
    col = np.array(map(toFloat, col))
    col2 = col[~np.isnan(col)]
    return len(col2)/len(col)

def frequencyCols(df):
    freqs = []    
    for c in df.columns:
        freqs.append(findFrequency(df[c]))
    return np.array(freqs)

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)

fig2 = plt.figure(figsize=(20,10))
ax2 = fig2.add_subplot(111)

color = ['b', 'r', 'g', 'm', 'y']
a = 1
n = 1
bar_width = 1/(n+1)


for i in range(n):
    df = loadData(csv_path,files[i])
    freqs = frequencyCols(df)
    index = np.arange(len(freqs))
    ax.bar(index + i*bar_width, freqs, width=bar_width, color=color[i], alpha=a)
    ax2.bar(index, np.sort(freqs), width=1, color=color[i], alpha=a)
    
ax.legend(['1', '2', '3', '4', '5'])