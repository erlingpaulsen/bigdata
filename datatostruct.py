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

zip_path = "data/zipped/"
zip_ex = "zip"
csv_path = "data/unzipped/KMFILES/"
csv_ex = "csv"

files = fh.filesInDir(zip_path,zip_ex)
print 'Number of files: '+str(len(files))

start = 0
stop = 10

for i in np.arange(start, stop):
    fh.unzipFiles([files[i]], zip_path, csv_path) 
    df = fh.loadDataCSV(csv_path, files[i])
    obcols = dh.getObjectCols(df)
    df[obcols] = dh.replaceBoolWithInt(df[obcols])
