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
sl = fh.loadDataH5('data/', 'signal_list.h5') 

class Variable:
    """
    Class for storing each variable and its attributes
    """
    
    def __init__(self, col):
        self.values = col.values[~np.isnan(col.values)]
        self.timestamps = col.index[~np.isnan(col.values)]
 
        (tagno, name, system, unit) = getVarNames(col)
        (f25, f50, f75, fmean) = getFrequencies(self.timestamps)
        self.tagno = str(tagno)        
        self.name = str(name)
        self.system = str(system)
        self.unit = str(unit)
        self.frequencies = np.array([f25, f50, f75, fmean])
    

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
    unit = '-'
    try:
        name = sl.loc[tag, 'RDS Name']
        system = sl.loc[tag, 'RDS Location']
        unit = sl.loc[tag, 'Unit']
    except KeyError:
        system = 'N/A'
    return (tagno, name, system, unit)


def getFrequencies(timestamps):
    n = len(timestamps)
    if n < 2:
        return (0,0,0,0)
    else:
        dT = np.subtract(timestamps[1:n], timestamps[0:n-1])
        dTs = 1 / (dT.seconds + dT.microseconds*(1e-06))
        return (np.percentile(dTs, 25), np.percentile(dTs, 50), np.percentile(dTs, 75), np.mean(dTs))



