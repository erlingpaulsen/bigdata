# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:34:07 2016

@author: ERLING

This file contains some relevant methods for working with and manipulating the
Pandas DataFrames.
"""
from __future__ import division
import pandas as pd
import numpy as np
import datastruct as ds


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

  
def interpolateDataFrame(df, resamp=0, resampT='1S'):
    """
    Interpolates missing values linearly based on the time, based on pandas 
    interpolate method.
    Has the possibility to resample the dataframe if necessary based on pandas
    resample method, if two measurements fall within the same resample period, 
    resample averages the values together.
    'xS' for x seconds
    'xT' for x minutes
    
    Inputs:
        df: Pandas DataFrame
        resamp: Default 0, change to 1 if you wish to resample
        resampT: Default set to 1 second, change if you wish to resample
                 on an other time interval
    
    Outputs:
        interpolated dataframe
    """
    if resamp:
        return df.resample(resampT).mean().interpolate(method='time', limit_direction='both')
    
    return df.interpolate(method='time', limit_direction='both')
    
def sumCargo(df, byName=1):
    clvls = ['CT030/Meas1/PRIM', 'CT031/Meas1/PRIM', 'CT032/Meas1/PRIM', 'CT033/Meas1/PRIM']
    if byName:    
        clvls = [ds.tag2name(c) for c in clvls]
#    ctot = df[clvls].sum(axis=1) 
#    ctot.name = 'Total Cargo Level [m]'
    df['TOTAL CARGO LEVEL [m]'] = df[clvls].sum(axis=1)
    df.drop(clvls, axis=1, inplace=True)
#    df = pd.concat([df,ctot], axis=1)
    return df
    
def sumFuelFlow(df, byName=1):
    fgs = ['GH075.01/Meas1/PRIM', 'GH071.01/Meas1/PRIM', 'GH067.01/Meas1/PRIM', 'GH063.01/Meas1/PRIM']
    if byName:    
        fgs = [ds.tag2name(fg) for fg in fgs]
#    fgtot = df[fgs].sum(axis=1)
#    fgtot.name = 'Total Fuel Gas Flow to MGE [kg/h]'
    df['TOTAL FUEL GAS FLOW TO MGE [kg/h]'] = df[fgs].sum(axis=1)
    df.drop(fgs,axis=1, inplace=True)
#    df = pd.concat([df,fgtot], axis=1)
    return df
    
def sumGasFlow(df, byName=1):
    ffs = ['MG1514/Meas1/PRIM', 'MG2514/Meas1/PRIM', 'MG3514/Meas1/PRIM', 'MG4514/Meas1/PRIM']
    if byName:    
        ffs = [ds.tag2name(ff) for ff in ffs]
#    fftot = df[ffs].sum(axis=1)
#    fftot.name = 'Total Fuel Oil Flow to MGE [l/h]'
    df['TOTAL FUEL OIL FLOW TO MGE [l/h]'] = df[ffs].sum(axis=1)
    df.drop(ffs,axis=1, inplace=True)
#    df = pd.concat([df,fftot], axis=1)
    return df

def sumEngineSpeed(df, byName=1):
    speeds = ['MG019/Meas1/PRIM', 'MG119/Meas1/PRIM', 'MG219/Meas1/PRIM', 'MG319/Meas1/PRIM']
    if byName:    
        speeds = [ds.tag2name(s) for s in speeds]
#    stot = df[speeds].sum(axis=1)
#    stot.name = 'Total Engine Speed [rpm]'
    df['TOTAL ENGINE SPEED [rpm]'] = df[speeds].sum(axis=1)
    df.drop(speeds, axis=1, inplace=True)
#    df = pd.concat([df,stot], axis=1)
    return df

def sumEnginePower(df, byName=1):
    pows = ['PM100.07/Meas1/PRIM', 'PM110.07/Meas1/PRIM', 'PM120.07/Meas1/PRIM', 'PM130.07/Meas1/PRIM']
    if byName:    
        pows = [ds.tag2name(p) for p in pows]
#    ptot = df[pows].sum(axis=1)
#    ptot.name = 'Total Engine Power [kW]'
    df['TOTAL ENGINE POWER [kW]'] = df[pows].sum(axis=1)
    df.drop(pows, axis=1, inplace=True)
#    df = pd.concat([df,ptot], axis=1)
    return df

def sumShaft(df, byName=1):
    pows = ['SP051/Meas1/PRIM', 'SP055/Meas1/PRIM']
    torqs = ['SP052/Meas1/PRIM','SP056/Meas1/PRIM']
    thrusts = ['SP053/Meas1/PRIM','SP057/Meas1/PRIM']
    speeds = ['SP054/Meas1/PRIM', 'SP058/Meas1/PRIM']
    
    if byName:    
        pows = [ds.tag2name(p) for p in pows]  
        torqs = [ds.tag2name(t) for t in torqs]
        thrusts = [ds.tag2name(th) for th in thrusts]
        speeds = [ds.tag2name(s) for s in speeds]
    
#    ptot = df[pows].abs().sum(axis=1)
#    ptot.name = 'Total Shaft Power [kW]'
    df['TOTAL SHAFT POWER [kW]'] = df[pows].sum(axis=1)
#    tqtot = df[torqs].abs().sum(axis=1)
#    tqtot.name = 'Total Shaft Torque [kNm]'
    df['TOTAL SHAFT TORQUE [kNm]'] = df[torqs].abs().sum(axis=1)
#    thtot = df[thrusts].abs().sum(axis=1)
#    thtot.name = 'Total Shaft Thrust [kNm]'
    df['TOTAL SHAFT THRUST [kN]'] = df[thrusts].abs().sum(axis=1)
#    stot = df[speeds].abs().sum(axis=1)
#    stot.name = 'Total Shaft Speed [rpm]'
    df['TOTAL SHAFT SPEED [rpm]'] = df[speeds].abs().sum(axis=1)
    
    df.drop(pows + torqs + thrusts + speeds, axis=1, inplace=True)
#    df = pd.concat([df, ptot, tqtot, thtot, stot], axis=1)
    return df
    
def sumVars(df):
    df = sumCargo(df)
    df = sumFuelFlow(df)
    df = sumGasFlow(df)
    df= sumEngineSpeed(df)
    df = sumEnginePower(df)
    df = sumShaft(df)
    return df


def cargoLevelFilter(df, byName=1):
    clvls = ['CT030/Meas1/PRIM', 'CT031/Meas1/PRIM', 'CT032/Meas1/PRIM', 'CT033/Meas1/PRIM']
    if byName:    
        clvls = [ds.tag2name(c) for c in clvls]
        
    df.loc[df[clvls[0]] < 0, clvls[0]] = np.nan
    df.loc[df[clvls[0]] > 30, clvls[0]] = np.nan
    df.loc[df[clvls[1]] < 0, clvls[1]] = np.nan
    df.loc[df[clvls[1]] > 30, clvls[1]] = np.nan
    df.loc[df[clvls[2]] < 0, clvls[2]] = np.nan
    df.loc[df[clvls[2]] > 30, clvls[2]] = np.nan
    df.loc[df[clvls[3]] < 0, clvls[3]] = np.nan
    df.loc[df[clvls[3]] > 30, clvls[3]] = np.nan
    return df

def speedOverGroundFilter(df, byName=1):
    sog= 'IBS001/Meas1/PRIM'
    if byName:
        sog = 'SPEED OVER GROUND  [kts]'
    
    # Removing anomalous points where speed over ground is larger than
    # 40 knotw
    df.loc[df[sog] > 40, sog] = np.nan
    return df
    
def windSpeedFilter(df, byName=1):
    windspeed = 'IBS003/Meas1/PRIM'
    if byName:
        windspeed = 'WIND SPEED  [kts]'
     
    # Removing anomalous points where wind speed is negative or larger
    # than 40 knots (Storm)
    df.loc[df[windspeed] < 0, windspeed] = np.nan
    df.loc[df[windspeed] > 40, windspeed] = np.nan
    return df
    
def fixWindDir(df, byName=1):
    winddir ='IBS004/Meas1/PRIM'
    if byName:
        winddir = 'WIND REL DIRECTION  [deg]'
        
    df[winddir] = coscos(df[winddir])
    return df
 
def splitWind(df, byName=1):
    wind ='IBS004/Meas1/PRIM'
    if byName:
        wind = 'WIND REL DIRECTION  [deg]'
    df['SIN(WIND)'] = map(np.sin, np.radians((df[wind])))
    df['COS(WIND)'] = map(np.cos, np.radians((df[wind])))
    return df

def recomputeWind(df, byName=1):
    wind ='IBS004/Meas1/PRIM'
    if byName:
        wind = 'WIND REL DIRECTION  [deg]'
    df[wind] = map(np.arctan2, df['SIN(WIND)'], df['COS(WIND)'])
    df[wind] = np.add(map(np.degrees, df[wind]), 180) 
    df.drop(['SIN(WIND)','COS(WIND)'], axis=1, inplace=True)
    return df
 
def coscos(deg):
    """
    Mirrors all angles to be between 0 and 180 degrees
    
    Parameter: Deg, can be array or single value
    
    Return: A deg between 0 and 180 degrees
    """
    return np.degrees((np.arccos(np.cos(np.radians((deg))))))
    
def laden(df, byName=1, interp=True):
    """
    Returns a DataFrame where the vessel is in laden condition
    
    Input:
        df: Pandas DataFrame
        byName: Change to 0 if input df has tagIDs as column names
        
    Output:
        df: Pandas DataFrame
    
    """
    
    clvls = ['CT030/Meas1/PRIM', 'CT031/Meas1/PRIM', 'CT032/Meas1/PRIM', 'CT033/Meas1/PRIM']
    if byName:    
        clvls = [ds.tag2name(c) for c in clvls]
    
    csmax = 1
    csmin = -1
    if interp:
        csmax = 0.0002099
        csmin = -0.0002034
    
    for i in np.arange(4):
        df = df[df[clvls[i]]>10]
        n = len(df[clvls[i]])
        df['CS'] = np.subtract(df[clvls[i]].iloc[1:n], df[clvls[i]].iloc[0:n-1])
        df = df[(df['CS'] < csmax) & (df['CS'] > csmin)]
    #df.drop('CS', axis=1, inplace=True)
    return df