# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:24:13 2016

@author: ERLING
"""

import numpy as np
import pandas as pd
import filehandler as fh
import datahandler as dh
import datastruct as ds
import timeit

zip_path = "data/zipped2/"
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
    fh.unzipFiles([files[i]], zip_path, unzip_path) 
    df = fh.loadDataCSV(csv_path, files[i][0:-3]+csv_ex)
    obcols = dh.getObjectCols(df)
    df[obcols] = dh.replaceBoolWithInt(df[obcols])
    struct = ds.Data(df)
    struct.save('data/struct/'+files[i][37:-3]+'txt')
    fh.deleteFile(files[i][0:-3]+csv_ex, csv_path)
    del df
    del struct

stoptime = timeit.default_timer()
print 'Time used loading and saving ' +str(stop-start)+' file(s) as struct: '+ str(int(stoptime-starttime))+' seconds'