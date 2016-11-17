# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 12:33:07 2016

@author: ERLING
"""

#import pandas as pd
#import numpy as np 
#import timeit

# To import files from another folder
import sys
sys.path.append('../')
import filehandler as fh
#import datahandler as dh
import datastruct as ds
from datastruct import Data


# Getting filenames
path = '../data/struct/'
rounded_path = '../data/struct_rounded/'
#freq_path = '../data/struct_freq_only/'
ex = 'txt'

#files = fh.filesInDir(rounded_path,ex)
#n = len(files)
#print 'Number of files: '+str(n)
#
## Loading structs
#starttime = timeit.default_timer()
#
#structs = []
#for f in files:
#    structs.append(ds.Data.load(rounded_path+f))
#    
#stoptime = timeit.default_timer()
#print 'Used '+ str(int(stoptime-starttime))+' seconds to load all structs.'
#
## Concatenating structs
#starttime = timeit.default_timer()
#
#struct_full = ds.concatStructs(structs)
#
#stoptime = timeit.default_timer()
#print 'Used '+ str(int(stoptime-starttime))+' seconds to concatenate all structs.'

# Saving struct
struct_full = Data.load('../data/FULL_PERIOD_ROUNDED.txt')
grp1_l = fh.loadColumn('../frequency/group1_log')
grp2_l = fh.loadColumn('../frequency/group2_log')
grp3_l = fh.loadColumn('../frequency/group3_log')
grp4_l = fh.loadColumn('../frequency/group4_log')
struct_grp4_l = ds.deleteVariables(struct_full, grp1_l+grp2_l+grp3_l)
struct_grp4_l.save('../data/FULL_PERIOD_ROUNDED_GRP4_LOG.txt')

#df_full = ds.toTable(struct_full)

#struct_full = Data.load('FULL_PERIOD_ROUNDED.txt')