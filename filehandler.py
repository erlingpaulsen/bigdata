# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:14:02 2016

@author: ERLING
"""

import glob, os, zipfile
import numpy as np
import pandas as pd


def printFilesInDir(directory, extension):
    """
    printFilesInDir(directory, extension) prints all the files in a directory
    with a given extension
    Input:
        - directory: Pathname to a directory
        - extension: File extension
    """
    owd = os.getcwd()
    os.chdir(directory)
    print "\nFiles in "+str(os.getcwd())+" with file extension "+extension+":"
    count = 0
    for file in glob.glob(extension):
        print "\t"+file
        count += 1
    print "Number of file(s): "+str(count)+"\n"
    os.chdir(owd)

def filesInDir(directory, extension):
    """
    filesInDir(directory, extension) returns a list with all the files in
    a directory with a given file extension
    
    Input:
        - directory: Pathname to a directory
        - extension: File extension
    Output:
        - files: List of files
    """
    files = []
    owd = os.getcwd()
    os.chdir(directory)
    for file in glob.glob('*.'+extension):
        files.append(file)
    os.chdir(owd)
        
    return files

def unzipFiles(files, filedir, unzipdir):
    """
    unzipFiles(files, filedir, unzipdir) unzips all the files with filename
    given by files existing in the direction filedir. The files are unzipped
    to the directory given by unzipdir.
    
    Input:
        - files: List of files
        - filedir: Pathname to the directory containing the files
        - unzipdir: Pathname do a directory where you want the unzipped files
    """
    owd = os.getcwd()
    os.chdir(unzipdir)
    n = len(files)
    print "\nUnzipping "+str(n)+" file(s) to "+str(os.getcwd())+"..."
    os.chdir(owd)
    for f in files:    
        zip_ref = zipfile.ZipFile(filedir+f, 'r')
        zip_ref.extractall(unzipdir)
        zip_ref.close()
    print "Done unzipping\n"

def deleteFile(f, filedir):
    try:
        os.remove(filedir+f)
    except OSError:
        print 'Could not delete '+filedir+f

def loadDataCSV(directory, filename, rows=-1):
    """
    loadDataCSV(directory, filename, rows=-1) loads data from a csv file into a 
    pandas dataframe.
    
    Input:
        - path: Pathname to a directory
        - filename: Filename, ending with .csv
        - rows: Number of rows to be loaded. If not specified, all rows are loaded.
    Output:
        - df: Pandas DataFrame
    """
    if rows == -1:
        df = pd.read_csv(directory+filename, sep=',', na_values=' ')
    else:
        df = pd.read_csv(directory+filename, sep=',', nrows=rows, na_values=' ')
    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
    df.index=df['Time']
    df.drop('Time', axis=1, inplace=True)
    return df
        
def loadDataH5(directory, filename):
    """
    loadDataH5(directory, filename, rows=-1) loads data from a h5 file into a 
    pandas dataframe.
    
    Input:
        - path: Pathname to a directory
        - filename: Filename, ending with .h5
    Output:
        - df: Pandas DataFrame
    """
    df = pd.read_hdf(directory+filename, 'table')
    return df
    
def saveSeries(series, filename):
    """
    Saves pandas series to a txt file as CSV format.
    
    Input:
        series: Series you want to save
        filename: filename of the .txt file, do not include file format (.txt)
    """
    
    #converting col from series to array and iterate through it
    #used for debugging mainly
    with open(filename + ".txt", "w") as text_file:
        for i in series.index:  
            text_file.write("%s,%s\n" % (i,series[i]))

###################################
#           Example               #
###################################
#
#path = "../data/zipped/"
#ex = "*.zip"
#unzipdir = "../data/unzipped/"
#printFilesInDir(path, ex)
#files = filesInDir(path, ex)
#unzipFiles(files, path, unzipdir)