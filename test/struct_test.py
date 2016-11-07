# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:04:07 2016

@author: ERLING
"""

import numpy as np
import pandas as pd
import filehandler as fh
import datahandler as dh
import matplotlib.dates as pltd
import pickle


tag_map = {'Start':'.01', 'Stop':'.02', 'OutRunning':'.03', 'Meas1':'', 'B4':'', 'Out':'', 'ProMeas':'', 'OutMeas':'', 'OutPosition':'.01'}
sl = fh.loadDataH5('../data/', 'signal_list.h5') 

class Variable:
    """
    Class for storing each variable and its attributes
    """
    
    def __init__(self, col):
        self.values = col.values[~np.isnan(col.values)]
        self.timestamps = pltd.date2num(col.index[~np.isnan(col.values)].values)
 
        (tagno, name, system) = getVarNames(col)
        self.tagno = str(tagno)        
        self.name = str(name)
        self.system = str(system)
    

class Data:
    
    def __init__(self, df):
        self.variables = {col : Variable(df[col]) for col in df.columns}
        
    def save(self, filename):
        f = file(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    
    @staticmethod
    def load(filename):
        with file(filename, 'rb') as f:
            return pickle.load(f)
            

def getVarNames(col):
    tag = col.name.split('/',2)
    if len(tag)>1:
        tag = tag[0]+tag_map[tag[1]]
    elif tag[0] == 'nan':
        tag = np.nan
    tagno = tag
    name = tag
    try:
        name = sl.loc[tag, 'RDS Name']
        system = sl.loc[tag, 'RDS Location']
    except KeyError:
        system = 'N/A'
    return (tagno, name, system)

def loadcsv():
    csv_path = "../data/unzipped/KMFILES/"
    ex = "csv"
    files = fh.filesInDir(csv_path,ex)
    df = fh.loadDataCSV(csv_path, files[0])
    obcols = dh.getObjectCols(df)
    df[obcols] = dh.replaceBoolWithInt(df[obcols])
    return df

df = loadcsv()
  
#coli = ['first', 'second', 'third']
#col1 = [np.nan,4,6]
#col2 = [100,200,300]
#df = pd.DataFrame(data={'Col1':col1, 'CA005/OutRunning/PRIM':col2}, index=coli)

datai = Data(df)
datai.save('mydata.txt')
#data2 = Data.load('mydata.txt')


#floats = col[~np.isnan(col)]