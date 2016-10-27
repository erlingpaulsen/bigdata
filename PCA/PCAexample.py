# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:44:29 2016

@author: anders
"""

import numpy as np

"""
Generate 4D-sample data. 

We make two classes of 4D data. We use data for 5 sports cars and 5 avarage
cars. The variables for each car is horsepower, displacement[L] , weight [lbs],
and Co2 emmision [g/mi]
""" 

"Class 1 sample data (5x4 matrix)"
'horsepower, displacement[L] , weight [lbs], Co2 emmision [g/mi]'
normalcars = np.array([[211,1.984,3875,314],[128,1.4,3125,250.38],
                       [175,2.5,3875,264.27],[132,1.798,3125,246.19],
                       [170,1.798,3375,174.55]])

"Class 2 sample data (5x4 matrix)"
sportscars = np.array([[568,6,4500,309.22],[553,4.4,4750,602.28],
                       [650,6.2,3875,280.2],[621,5.98,5500,646.63],
                       [572,3.8,3875,264.87]])
                       
"Lables for variables and objects"                       
variables = np.array(['Horsepower','Displacements','Weight','Co2emmision'])
names = np.array(['audia4','hyndaielantra','subarulegaxy','corolla','golf','astonmartin','bmwm5','corvette','mercedess65amg','prosche911turbo'])


"""
Since PCA is unsupervised we do not know that the data consists of 2 classes. 
Therefore we combine normal- and sportscars to the matrix allcars(10x4)
"""
allcars = np.concatenate((normalcars, sportscars), axis=0)


"""
Since the variables are in different units we standardize the data with mean=0
and variance = 1.
"""
from sklearn.preprocessing import StandardScaler
allcarsstd = StandardScaler().fit_transform(allcars)


"We calculate the covariance matrix of the standardized data"
covmat=np.cov(allcarsstd.T)


" We find eigenvalues and eigenvectors, and sort them in decending order."
eigvals, eigvecs = np.linalg.eig(covmat)
eigpairs = [(np.abs(eigvals[i]), eigvecs[:,i]) for i in range(len(eigvals))]
eigpairs.sort(key=lambda x: x[0], reverse=True)



" Calculate cummulative explaiend variance for PC component i"
sumeigvals = sum(eigvals)
sortedeigvals = sorted(eigvals, reverse=True)
explvar=np.zeros(len(eigvals))
for idx, i in enumerate(sortedeigvals):
    explvar[idx] = (i/sumeigvals)*100
cumvarexp = np.cumsum(explvar)


" Plot cummulative explaiend variance "
from matplotlib import pyplot as plt
plt.close('all')
xaxis =np.arange(0.5,len(cumvarexp)+0.5)
fig1= plt.figure()
ax1 = fig1.add_subplot(111)
ax1.bar(xaxis, cumvarexp, width=0.8, align='center')
ax1.set_xticks(xaxis)
ax1.set_xticklabels([1,2,3,4])
ax1.set_ylabel('Cumulative explained variance')
ax1.set_xlabel('Principal components')




"Extract loadings"
P = np.hstack((eigpairs[0][1].reshape(4,1),
                      eigpairs[1][1].reshape(4,1)))



Z = np.dot(allcarsstd,P)
carclass=np.array([0,0,0,0,0,1,1,1,1,1])
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.grid()
ax2.set_title('Scores plot')
ax2.scatter(Z[:,0],Z[:,1],c=carclass,s=50)
for label, x, y in zip(names, Z[:, 0], Z[:, 1]):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (-10, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'grey', alpha = 0.5),
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))


fig3= plt.figure()
ax3 = fig3.add_subplot(111)
ax3.scatter(P[:,0],P[:,1])
ax3.grid()
ax3.set_title('Loadings plot')
for label, x, y in zip(variables, P[:, 0], P[:, 1]):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (-10, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'grey', alpha = 0.5),
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))





