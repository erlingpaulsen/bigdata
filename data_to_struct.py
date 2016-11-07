# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:24:13 2016

@author: ERLING

This script unzips all the files, loads the CSV files, converst them to a 
custom data structure and rounds off every timestamp to the nearest second,
averaging the values for duplicate timestamps.
"""

import numpy as np
import pandas as pd
import filehandler as fh
import datahandler as dh
import datastruct as ds
import timeit

zip_path = "data/zipped/"
zip_ex = "zip"
unzip_path = "data/unzipped/"
csv_path = "data/unzipped/KMFILES/"
csv_ex = "csv"

files = fh.filesInDir(zip_path,zip_ex)
print 'Number of files: '+str(len(files))

start = 0
stop = len(files)

starttime = timeit.default_timer()

for i in np.arange(start, stop):
    print 'Processing file: '+files[i][37:-4]
    # Unzipping the zip file
    fh.unzipFiles([files[i]], zip_path, unzip_path)
    
    # Loading the CSV file as DataFrame
    df = fh.loadDataCSV(csv_path, files[i][0:-3]+csv_ex)
    
    # Replacing columns with True and False with 1 and 0
    obcols = dh.getObjectCols(df)
    df[obcols] = dh.replaceBoolWithInt(df[obcols])
    
    # Creating a custom data structure of the data
    struct = ds.Data(df)
    
    # Rounding off every timestamp to seconds
    struct = ds.roundTimestamps(struct)
    
    # Removing duplicate timestamps and replace the value by the average
    struct = ds.avgDuplicates(struct)
    
    # Saving the struct
    struct.save('data/struct/'+files[i][37:-3]+'txt')
    fh.deleteFile(files[i][0:-3]+csv_ex, csv_path)
    del df
    del struct

stoptime = timeit.default_timer()
print 'Time used loading, preprocessing and saving ' +str(stop-start)+' file(s) as struct: '+ str(int(stoptime-starttime))+' seconds'