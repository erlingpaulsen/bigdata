# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:34:07 2016

@author: ERLING
"""
from __future__ import division
import pandas as pd
import numpy as np


def toFloat(x, notFloat=np.nan):
    """
    toFloat(x, notFloat=np.nan) tries to convert x to float. If conversion 
    is not possible, NaN is returned. If possible, float(x) is returned.
    
    Input:
        - x: Data to be converted.
        - notFloat: Value that will be returned if converion is not possible.
                    By default, NaN is returned.
    Output:
        - float(x) if possible, notFloat else.
    """
    try:
        float(x)
        return float(x)
    except ValueError:
        return notFloat

def toInt(x):
    """
    toBool(x) tries to convert x to int. If conversion is not possible, x is
    returned. If possible int(x) is returned.
    
    Input:
        - x: Data to be converted.
    Output: 
        - int(x) is possible, x else.
    """
    try:
        int(x)
        return int(x)
    except ValueError:
        return x

def colToFloat(col, notFloat=np.nan):
    """
    colToFloat(col, notFloat=np.nan) converts a column to float using 
    toFloat(x, notFloat=np.nan).
    
    Input:
        - col: Column of data to be converted.
    Output:
        - float(col) where conversion is possible, notFloat else.
    """
    return np.array(map(toFloat, col))

def colBoolToInt(col):
    """
    colBoolToInt(col) converts a column with boolean values to integers using
    toInt(x).
    
    Input:
        - col: Column of data to be converted.
    Output:
        - int(col) where conversion is possible, x else.
    """
    return np.array(map(toInt, col.values))
    
def replaceBoolWithInt(df):
    """
    replaceBoolWithInt(df) converts boolean values in a dataframe with
    integers using colBoolToInt on all columns.
    
    Input:
        - df: Dataframe
    Output:
        - int(df) where conversion is possible, df else.
    """
    return df.apply(colBoolToInt, axis=0)
   
def getFloatEntries(col):
    """
    getFloatEntries(col) converts a column to float and returns the number
    of float entries and their indices.
    
    Input:
        - col: Column of data.
    Output:
        - (n, indices): Tuple consisting of the number of float entries and
                        their indices.
    """
    floats = np.where(~np.isnan(col))
    return (len(floats[0]), floats[0])

def countFloatEntries(col):
    """
    countFloatEntries(col) converts a column to float and returns the number
    of float entriess.
    
    Input:
        - col: Column of data.
    Output:
        - n: Number of float entries.
    """
    floats = col[~np.isnan(col)]
    return len(floats)

def getColFreqs(df):
    """
    getColFreqs(df) computes the frequency content of the columns in the 
    dataframe df and stores it in a new dataframe.
    
    Input:
        - df: Dataframe.
    Output:
        - (freqdf, dTs): A tuple consisting of a new dataframe freqdf with the
                         columns of df as indices, and their frequencies and 
                         number of measurements as columns, and the time 
                         interval of df dTs in seconds.
    """
    ncols = df.shape[0]
    dT = df.index[ncols-1]-df.index[0]
    dTs = float(dT.seconds)+float(dT.microseconds*(1e-06))
    hits = np.array(df.apply(countFloatEntries, axis=0))
    freqs = np.divide(hits,dTs)
    freqdf = pd.DataFrame(data={'Frequency':freqs, 'Number of measurements':hits}, index=df.columns)
    return (freqdf, dTs)


def getObjectCols(df):
    """
    getObjectCols(df) returns the columns from a dataframe with datatype object.
    
    Input:
        - df: Dataframe.
    Output:
        - Columns with datatype object in the dataframe.
    """
    return df.columns[df.dtypes=='object']