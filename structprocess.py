# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:25:02 2016

@author: ERLING
"""

import pandas as pd
import numpy as np
import filehandler as fh
import datahandler as dh
import datastruct as ds
import matplotlib.pyplot as plt
from datastruct import Data

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

struct_path = "data/struct/"
rounded_path = 'data/struct_rounded/'
freq_path = 'data/struct_freq_only/'
ex = 'txt'
winddir = 'WIND REL DIRECTION  [deg]'
windspeed = 'WIND SPEED  [kts]'
cargo4 = 'CARGO LEVEL - TANK4  [m]'
#files = fh.filesInDir(rounded_path,ex)
#n = len(files)
#print 'Number of files: '+str(n)
#
#varlist = fh.loadColumn('selected_vars')
#print 'Number of selected variables: '+str(len(varlist))
#
#structs = []
#for i,f in enumerate(files):
#    print 'Loading file '+str(i)
#    struct = Data.load(rounded_path+f)
#    struct = ds.keepVariables(struct, varlist)
#    structs.append(struct)
#
#print 'Concatenating structs'
#struct_full = ds.concatStructs(structs)
#print 'Converting to DataFrame'
#df_full = ds.toTable(struct_full)
#
#print 'Saving struct and DataFrame'
#struct_full.save('FULL_PERIOD_ROUNDED_SELECTED_VARS.txt')
#df_full.to_hdf('FULL_PERIOD_ROUNDED_SELECTED_VARS.h5', 'table')
#del struct_full

# Loading dataframe
#df = pd.read_hdf('FULL_PERIOD_ROUNDED_SELECTED_VARS.h5', 'table')

# Filtering wind speed
#df = dh.windSpeedFilter(df)

# Filtering speed over ground
#df = dh.speedOverGroundFilter(df)

# Filtering cargo level
#df = dh.cargoLevelFilter(df)
#df.to_hdf('FULL_PERIOD_ROUNDED_SELECTED_VARS_FILTERED.h5', 'table')
#df = pd.read_hdf('FULL_PERIOD_ROUNDED_SELECTED_VARS_FILTERED.h5', 'table')
# Splitting wind direction in cos and sin
#df = dh.splitWind(df)

# Interpolating dataframe
#df = dh.interpolateDataFrame(df)

# Recomputing wind direction
#df = dh.recomputeWind(df)

# Make wind direction symmetric about starboard and port
#df = dh.fixWindDir(df)
#df.to_hdf('FULL_PERIOD_ROUNDED_SELECTED_VARS_FILTERED_INTERP.h5', 'table')
# Summing some variables
#df_summed = dh.sumVars(df)

#mru = pd.read_hdf('SELECTED_PERIOD_MRU.h5', 'table')
df = pd.read_hdf('data/FULL_PERIOD_ROUNDED_SELECTED_VARS_FILTERED_INTERP_SUM.h5', 'table')

#mru = dh.interpolateDataFrame(mru)

#df_mru = df.loc[mru.index]
#df_mru = pd.concat([df_mru, mru], axis=1)
#df_mru.to_hdf('SELECTED_LADED_PERIOD_ROUNDED_SELECTED_VARS_FILTERED_INTERP_MRU.h5', 'table')
#df_mru_5 = dh.interpolateDataFrame(df_mru, resamp=1, resampT='5T')
#df_mru_5.to_hdf('SELECTED_LADEN_PERIOD_ROUNDED_SELECTED_VARS_FILTERED_INTERP_MRU_5MIN.h5', 'table')
#df_mru_5.to_csv('SELECTED_LADED_PERIOD_ROUNDED_SELECTED_VARS_FILTERED_INTERP_MRU_5MIN.csv', sep=',')


#print 'Interpolating and saving'
df_interp = dh.interpolateDataFrame(df, resamp=1, resampT='1T')
df_interp.to_hdf('FULL_PERIOD_ROUNDED_SELECTED_VARS_FILTERED_INTERP_SUM_1MIN.h5', 'table')
#
#print 'Resampling every minute, interpolating and saving'
#df_interp = dh.interpolateDataFrame(df_sum, resamp=1, resampT='1T')
#df_interp.to_hdf('FULL_PERIOD_ROUNDED_SELECTED_VARS_SUMMED_INTERP_1MIN.h5', 'table')
#
#print 'Resampling every 5 minutes, interpolating and saving'
#df_interp = dh.interpolateDataFrame(df_sum, resamp=1, resampT='5T')
#df_interp.to_hdf('FULL_PERIOD_ROUNDED_SELECTED_VARS_SUMMED_INTERP_5MIN.h5', 'table')
#
#print 'Resampling every 10 minutes, interpolating and saving'
#df_interp = dh.interpolateDataFrame(df_sum, resamp=1, resampT='10T')
#df_interp.to_hdf('FULL_PERIOD_ROUNDED_SELECTED_VARS_SUMMED_INTERP_10MIN.h5', 'table')   