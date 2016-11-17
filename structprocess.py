# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:25:02 2016

@author: ERLING
"""

import pandas as pd
import numpy as np
import filehandler as fh
import datastruct as ds
import timeit

struct_path = "data/struct/"
rounded_path = 'data/struct_rounded/'
freq_path = 'data/struct_freq_only/'
ex = 'txt'

tag = 'RG012C/Meas1/PRIM'

files = fh.filesInDir(rounded_path,ex)
n = len(files)
print 'Number of files: '+str(n)


#f1 = files[0]
#f2 = files[1]
#f3 = files[2]
#f4 = files[3]
#
#struct1 = ds.Data.load(rounded_path+f1)
#struct2 = ds.Data.load(rounded_path+f2)
#struct3 = ds.Data.load(rounded_path+f3)
#struct4 = ds.Data.load(rounded_path+f4)
#fh.saveColumn(struct1.variables[tag].timestamps, "struct1ts")
#fh.saveColumn(struct2.variables[tag].timestamps, "struct2ts")
#fh.saveColumn(struct3.variables[tag].timestamps, "struct3ts")
#fh.saveColumn(struct4.variables[tag].timestamps, "struct4ts")
#
#
#struct5 = ds.concatStructs([struct1, struct2, struct3, struct4])
#fh.saveColumn(struct5.variables[tag].timestamps, "struct5ts")

#plt.plot(struct5.variables[tag].timestamps, np.arange(len(struct5.variables[tag].timestamps)))

#starttime = timeit.default_timer()
#struct = ds.Data.load(freq_path+'freq_every_20th_file.txt')
#
#dfs = []
#for k in struct.variables.keys():
#    p10 = np.percentile(struct.variables[k].frequencies, 10)  
#    p25 = np.percentile(struct.variables[k].frequencies, 25) 
#    p50 = np.percentile(struct.variables[k].frequencies, 50) 
#    p75 = np.percentile(struct.variables[k].frequencies, 75) 
#    p90 = np.percentile(struct.variables[k].frequencies, 90) 
#    mean = np.mean(struct.variables[k].frequencies)
#    dfs.append(pd.DataFrame(data={k:[p10, p25, p50, p75, p90, mean]}))
#df_freq = pd.concat(dfs, axis=1)
#stoptime = timeit.default_timer()
#print 'Used '+ str(int(stoptime-starttime))+' seconds in total to convert to table.'
#def constructFreqDf():
#    init = True
#    for i,f in enumerate(files):
#        struct = ds.Data.load(struct_path+f)
#        keys = struct.variables.keys()
#        if init:
#            df = pd.DataFrame(index = keys)
#            init = False
#        
#        d25 = []
#        d50 = []
#        d75 = []
#        dmean = []
#        for k in keys:
#            d25.append(struct.variables[k].frequencies[0])
#            d50.append(struct.variables[k].frequencies[1])
#            d75.append(struct.variables[k].frequencies[2])
#            dmean.append(struct.variables[k].frequencies[3])
#        
#        df_temp = pd.DataFrame(data={'File '+str(i)+': f25':np.array(d25),
#                                     'File '+str(i)+': f50':np.array(d50),
#                                     'File '+str(i)+': f75':np.array(d75),
#                                     'File '+str(i)+': fmean':np.array(dmean)},
#                                     index = keys)
#        
#        df = pd.concat([df,df_temp],axis=1)
#        
#    del d25, d50, d75, dmean, init, df_temp

#X = preprocessing.scale(df.values)
#
#kmeans = KMeans(n_clusters=3).fit(X)
#dbscan = DBSCAN().fit(X)