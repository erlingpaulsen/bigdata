# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:29:34 2016

@author: ERLING

This script is used to unzip all the zipped datafiles located in 
'data/zipped/' to the location 'data/unzipped/KMFILES/'
"""

import filehandler as fh
import timeit

zip_path = 'data/zipped/'
unzip_path = 'data/unzipped/'
ext = 'zip'

# Loading files
files = fh.filesInDir(zip_path, ext)
print str(len(files))+' files to unzip.'

#Unzipping the files and timing the process
start = timeit.default_timer()
fh.unzipFiles(files, zip_path, unzip_path)
stop = timeit.default_timer()
print ('Time used unzipping ' +str(len(files))+' file(s): '+ str(stop - start))+' seconds'