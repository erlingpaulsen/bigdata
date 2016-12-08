# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:21:49 2016

@author: ERLING
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


def plotAll(df, mru=0):
    if mru==1:
        for col in df.columns:
            fig = plt.figure(figsize=(28,14),dpi=80, facecolor='w', edgecolor='k')    
            ax = fig.add_subplot(111)
            ax.plot(df.index, df[col], marker='o', linestyle='', markersize=3, markeredgewidth=0.0)
            xlabel = col.split(':')[0][18:]
            if col.split(':')[0][-2:] == 'm0':
                ylabel = 'm^2'
            elif col.split(':')[0][-2:] == 'm1':
                ylabel = 'm^2/s'
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_ylabel(ylabel, fontsize=16)
            ax.set_title(col, fontsize=20)
            ax.grid()
            fig.savefig('figs/mru_full/'+xlabel+'.jpg', bbox_inches='tight')
            plt.close('all')
        
    else:
        for col in df.columns:
            fig = plt.figure(figsize=(28,14),dpi=80, facecolor='w', edgecolor='k')    
            ax = fig.add_subplot(111)
            ax.plot(df.index, df[col], marker='o', linestyle='', markersize=3, markeredgewidth=0.0)
            ax.set_xlabel(col.split('[')[0], fontsize=16)
            ax.set_ylabel(col.split('[')[1].split(']')[0], fontsize=16)
            ax.set_title(col, fontsize=20)
            ax.grid()
            fig.savefig('figs/variable_plots_interpolated/'+col.split('[')[0]+'.jpg', bbox_inches='tight')
            plt.close('all')
        
def plotCorrMat(df, filename):
    fig = plt.figure(figsize=(20,20),dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)    
    cax = ax.matshow(np.corrcoef(np.transpose(df.values)))
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.columns)))
    ax.set_xticklabels(['']+df.columns, rotation=90)
    ax.set_yticklabels(['']+df.columns, rotation=0)
    fig.savefig('figs/correlationmat/'+filename+'.pdf', bbox_inches='tight')
    plt.close('all')
    
def plotSeriesLadenBallast(series, start, stop):
    col = series.name    
    fig = plt.figure(figsize=(28,14),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.plot(series.loc[:start].index, series.loc[:start], marker='o', linestyle='', markersize=3, markeredgewidth=0.0, color='b', label='Ballast')
    ax.plot(series.loc[stop:].index, series.loc[stop:], marker='o', linestyle='', markersize=3, markeredgewidth=0.0, color='b')
    ax.plot(series.loc[start:stop].index, series.loc[start:stop], marker='o', linestyle='', markersize=3, markeredgewidth=0.0, color='r', label='Laden')
    ax.set_xlabel(col.split('[')[0], fontsize=16)
    ax.set_ylabel(col.split('[')[1].split(']')[0], fontsize=16)
    ax.set_title(col, fontsize=20)
    ax.grid()
    ax.legend(prop={'size':20}, markerscale=5)
    fig.savefig('figs/laden_ballast/'+col.split('[')[0]+'.jpg', bbox_inches='tight')
    plt.close('all')
    
def plotPCA(pca, scores, labels):
    
    labelpc1 = 'Principal Component 1 ('+str(np.round(pca.explained_variance_ratio_[0]*100))+'%)'
    labelpc2 = 'Principal Component 2 ('+str(np.round(pca.explained_variance_ratio_[1]*100))+'%)'
    labelpc3 = 'Principal Component 3 ('+str(np.round(pca.explained_variance_ratio_[2]*100))+'%)'
    labelpc4 = 'Principal Component 4 ('+str(np.round(pca.explained_variance_ratio_[3]*100))+'%)'
    labelpc5 = 'Principal Component 5 ('+str(np.round(pca.explained_variance_ratio_[4]*100))+'%)'
    labelpc6 = 'Principal Component 6 ('+str(np.round(pca.explained_variance_ratio_[5]*100))+'%)'
    
    # Explained Variance
    fig = plt.figure(figsize=(28,12),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ex_var_cum = np.multiply(100,np.array([sum(pca.explained_variance_ratio_[0:i+1]) for i,x in enumerate(pca.explained_variance_ratio_)]))
    x = np.arange(pca.n_components)+1
    ax.plot(x, ex_var_cum, marker='x', linestyle='-', markersize=10, markeredgewidth=2, linewidth=2, color='b', label='Explained Variance')
    for percent, x, y in zip(np.round(ex_var_cum), x, ex_var_cum):
        ax.text(x+0.02,y-1.5,str(percent)+'%', fontsize=16)     
    ax.set_xlabel('Principal Component', fontsize=18)
    ax.set_ylabel('Explained Variance [%]', fontsize=18)
    ax.set_title('Explained Variance', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.axis([1,pca.n_components,40,100])
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/explained_variance.pdf', bbox_inches='tight')
    plt.close('all')
    
    # Loadings PC1 vs. PC2
    fig = plt.figure(figsize=(28,12),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(pca.components_[0], pca.components_[1], marker='o', s=60, c='b', label='Loadings')
    ax.scatter(0,0, marker='x', s=100, c='k')
    for label, x, y in zip(labels, pca.components_[0], pca.components_[1]):
        plt.annotate(label, xy = (x, y), xytext = (8, 0),
                     textcoords = 'offset points', ha = 'left', va = 'center')   
    ax.set_xlabel(labelpc1, fontsize=18)
    ax.set_ylabel(labelpc2, fontsize=18)
    ax.set_title('Loadings PC1 and PC2', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/loadings_pc1_pc2.pdf', bbox_inches='tight')
    plt.close('all')
    
    # Loadings PC1 vs. PC2 Cluster 1
    fig = plt.figure(figsize=(18,9),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(pca.components_[0], pca.components_[1], marker='o', s=40, c='b', label='Loadings')
    ax.scatter(0,0, marker='x', s=100, c='k')
    for label, x, y in zip(labels, pca.components_[0], pca.components_[1]):
        plt.annotate(label, xy = (x, y), xytext = (8, 0),
                     textcoords = 'offset points', ha = 'left', va = 'center')   
    ax.set_xlabel(labelpc1, fontsize=18)
    ax.set_ylabel(labelpc2, fontsize=18)
    ax.set_title('Loadings PC1 and PC2', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.axis([0.1, 0.25, 0.3, 0.4])
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/loadings_pc1_pc2_cluster1.pdf', bbox_inches='tight')
    plt.close('all')
    
    # Loadings PC1 vs. PC2 Cluster 2
    fig = plt.figure(figsize=(18,9),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(pca.components_[0], pca.components_[1], marker='o', s=40, c='b', label='Loadings')
    ax.scatter(0,0, marker='x', s=100, c='k')
    for label, x, y in zip(labels, pca.components_[0], pca.components_[1]):
        plt.annotate(label, xy = (x, y), xytext = (8, 0),
                     textcoords = 'offset points', ha = 'left', va = 'center')    
    ax.set_xlabel(labelpc1, fontsize=18)
    ax.set_ylabel(labelpc2, fontsize=18)
    ax.set_title('Loadings PC1 and PC2', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.axis([0.25, 0.31, -0.16, -0.06])
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/loadings_pc1_pc2_cluster2.pdf', bbox_inches='tight')
    plt.close('all')
    
    # Scores PC1 vs. PC2
    fig = plt.figure(figsize=(28,12),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(scores[:,0], scores[:,1], marker='o', s=40, c='b', label='Scores')
    ax.set_xlabel(labelpc1, fontsize=18)
    ax.set_ylabel(labelpc2, fontsize=18)
    ax.set_title('Scores PC1 and PC2', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/scores_pc1_pc2.pdf', bbox_inches='tight')
    plt.close('all')
    
    # Loadings PC3 vs. PC4
    fig = plt.figure(figsize=(28,12),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(pca.components_[2], pca.components_[3], marker='o', s=60, c='b', label='Loadings')
    ax.scatter(0,0, marker='x', s=100, c='k')
    for label, x, y in zip(labels, pca.components_[2], pca.components_[3]):
        plt.annotate(label, xy = (x, y), xytext = (8, 0),
                     textcoords = 'offset points', ha = 'left', va = 'center')   
    ax.set_xlabel(labelpc3, fontsize=18)
    ax.set_ylabel(labelpc4, fontsize=18)
    ax.set_title('Loadings PC3 and PC4', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/loadings_pc3_pc4.pdf', bbox_inches='tight')
    plt.close('all')
    
    # Loadings PC3 vs. PC4 Cluster 1
    fig = plt.figure(figsize=(18,9),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(pca.components_[2], pca.components_[3], marker='o', s=40, c='b', label='Loadings')
    ax.scatter(0,0, marker='x', s=100, c='k')
    for label, x, y in zip(labels, pca.components_[2], pca.components_[3]):
        plt.annotate(label, xy = (x, y), xytext = (8, 0),
                     textcoords = 'offset points', ha = 'left', va = 'center')   
    ax.set_xlabel(labelpc3, fontsize=18)
    ax.set_ylabel(labelpc4, fontsize=18)
    ax.set_title('Loadings PC3 and PC4', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.axis([-0.07, 0.25, -0.06, 0.08])
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/loadings_pc3_pc4_cluster1.pdf', bbox_inches='tight')
    plt.close('all')
    
    # Scores PC3 vs. PC4
    fig = plt.figure(figsize=(28,12),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(scores[:,2], scores[:,3], marker='o', s=40, c='b', label='Scores')
    ax.set_xlabel(labelpc3, fontsize=18)
    ax.set_ylabel(labelpc4, fontsize=18)
    ax.set_title('Scores PC3 and PC4', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/scores_pc3_pc4.pdf', bbox_inches='tight')
    plt.close('all')
    
    # Loadings PC5 vs. PC6
    fig = plt.figure(figsize=(28,12),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(pca.components_[4], pca.components_[5], marker='o', s=60, c='b', label='Loadings')
    ax.scatter(0,0, marker='x', s=100, c='k')
    for label, x, y in zip(labels, pca.components_[4], pca.components_[5]):
        plt.annotate(label, xy = (x, y), xytext = (8, 0),
                     textcoords = 'offset points', ha = 'left', va = 'center')   
    ax.set_xlabel(labelpc5, fontsize=18)
    ax.set_ylabel(labelpc6, fontsize=18)
    ax.set_title('Loadings PC5 and PC6', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/loadings_pc5_pc6.pdf', bbox_inches='tight')
    plt.close('all')
    
    # Loadings PC5 vs. PC6 Cluster 1
    fig = plt.figure(figsize=(18,9),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(pca.components_[4], pca.components_[5], marker='o', s=40, c='b', label='Loadings')
    ax.scatter(0,0, marker='x', s=100, c='k')
    for label, x, y in zip(labels, pca.components_[4], pca.components_[5]):
        plt.annotate(label, xy = (x, y), xytext = (8, 0),
                     textcoords = 'offset points', ha = 'left', va = 'center')   
    ax.set_xlabel(labelpc5, fontsize=18)
    ax.set_ylabel(labelpc6, fontsize=18)
    ax.set_title('Loadings PC5 and PC6', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.axis([-0.1, 0.3, -0.15, 0.1])
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/loadings_pc5_pc6_cluster1.pdf', bbox_inches='tight')
    plt.close('all')
    
    # Scores PC5 vs. PC6
    fig = plt.figure(figsize=(28,12),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(scores[:,4], scores[:,5], marker='o', s=40, c='b', label='Scores')
    ax.set_xlabel(labelpc5, fontsize=18)
    ax.set_ylabel(labelpc6, fontsize=18)
    ax.set_title('Scores PC5 and PC6', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/scores_pc5_pc6.pdf', bbox_inches='tight')
    plt.close('all')
    
    # Loadings PC1 vs. PC2 vs. PC3
    fig = plt.figure(figsize=(28,12),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca.components_[0], pca.components_[1], pca.components_[3], marker='o', s=60, c='b')
    ax.scatter(0,0, marker='x', s=100, c='k')
    for label, x, y, z in zip(labels, pca.components_[0], pca.components_[1], pca.components_[2]):
        ax.text(x,y,z, label)
    ax.set_xlabel(labelpc1, fontsize=18)
    ax.set_ylabel(labelpc2, fontsize=18)
    ax.set_zlabel(labelpc3, fontsize=18)
    ax.set_title('Loadings PC1, PC2 and PC3', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.legend(prop={'size':20}, markerscale=1)   
   
    
    # Loadings PC1 vs. PC3
    fig = plt.figure(figsize=(28,12),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(pca.components_[0], pca.components_[2], marker='o', s=60, c='b', label='Loadings')
    ax.scatter(0,0, marker='x', s=100, c='k')
    for label, x, y in zip(labels, pca.components_[0], pca.components_[2]):
        plt.annotate(label, xy = (x, y), xytext = (8, 0),
                     textcoords = 'offset points', ha = 'left', va = 'center')   
    ax.set_xlabel(labelpc1, fontsize=18)
    ax.set_ylabel(labelpc3, fontsize=18)
    ax.set_title('Loadings PC1 and PC3', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/loadings_pc1_pc3.pdf', bbox_inches='tight')
    plt.close('all')    
 
    # Scores PC1 vs. PC3
    fig = plt.figure(figsize=(28,12),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(scores[:,0], scores[:,2], marker='o', s=40, c='b', label='Scores')
    ax.set_xlabel(labelpc1, fontsize=18)
    ax.set_ylabel(labelpc3, fontsize=18)
    ax.set_title('Scores PC1 and PC3', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/scores_pc1_pc3.pdf', bbox_inches='tight')
    plt.close('all')
   
    # Scores PC1 vs. PC2 vs. PC3
    fig = plt.figure(figsize=(28,12),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(scores[:,0], scores[:,1], scores[:,2], marker='o', s=40, c='b')
    ax.set_xlabel(labelpc1, fontsize=18)
    ax.set_ylabel(labelpc2, fontsize=18)
    ax.set_zlabel(labelpc3, fontsize=18)
    ax.set_title('Scores PC1, PC2 and PC3', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.legend(prop={'size':20}, markerscale=1)
    fig.savefig('figs/pca/scores_pc1_pc2_pc3.pdf', bbox_inches='tight')
    plt.close('all')

def plotScoresByCluster(scores, scores2, kmeans, ward, pca):
    
    labelpc1 = 'Principal Component 1 ('+str(np.round(pca.explained_variance_ratio_[0]*100))+'%)'
    labelpc2 = 'Principal Component 2 ('+str(np.round(pca.explained_variance_ratio_[1]*100))+'%)'
    
    # Scores PC1 vs. PC2 Clustered using Ward
    fig = plt.figure(figsize=(28,12),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(scores2[:,0], scores2[:,1], marker='o', s=40, c=ward.labels_, label='Scores')
    ax.set_xlabel(labelpc1, fontsize=18)
    ax.set_ylabel(labelpc2, fontsize=18)
    ax.set_title('Scores PC1 and PC2 - Ward', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    fig.savefig('figs/pca/scores_pc1_pc2_clustered_ward.pdf', bbox_inches='tight')
    plt.close('all')
    
    # Scores PC1 vs. PC2 Clustered using KMeans
    fig = plt.figure(figsize=(28,12),dpi=80, facecolor='w', edgecolor='k')    
    ax = fig.add_subplot(111)
    ax.scatter(scores[:,0], scores[:,1], marker='o', s=40, c=kmeans.labels_, label='Scores')
    ax.set_xlabel(labelpc1, fontsize=18)
    ax.set_ylabel(labelpc2, fontsize=18)
    ax.set_title('Scores PC1 and PC2 - KMeans', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    fig.savefig('figs/pca/scores_pc1_pc2_clustered_kmeans.pdf', bbox_inches='tight')
    plt.close('all')