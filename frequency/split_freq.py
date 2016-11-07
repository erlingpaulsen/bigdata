# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 12:33:07 2016

@author: ERLING
"""

import pandas as pd
import numpy as np 
import timeit

# To import files from another folder
import sys
sys.path.append('../')
import filehandler as fh
import datahandler as dh
import datastruct as ds
from datastruct import Data


# Getting filenames
path = '../data/struct/'
rounded_path = '../data/struct_rounded/'
#freq_path = '../data/struct_freq_only/'
ex = 'txt'

files = fh.filesInDir(rounded_path,ex)
n = len(files)
print 'Number of files: '+str(n)

# Loading structs
starttime = timeit.default_timer()

structs = []
for f in files:
    structs.append(ds.Data.load(rounded_path+f))
    
stoptime = timeit.default_timer()
print 'Used '+ str(int(stoptime-starttime))+' seconds to load all structs.'

# Concatenating structs
starttime = timeit.default_timer()

struct_full = ds.concatStructs(structs)

stoptime = timeit.default_timer()
print 'Used '+ str(int(stoptime-starttime))+' seconds to concatenate all structs.'

# Saving struct
struct_full.save('FULL_PERIOD_ROUNDED.txt')

#struct_full = Data.load('FULL_PERIOD_ROUNDED.txt')