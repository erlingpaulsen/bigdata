# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:02:13 2016

@author: Christian
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)
spread = np.random.rand(50) * 100 #Generate 50 random numbers from 0-100
center = np.ones(25) * 50 #Vector of 25 numbers with value 50
flier_high = np.random.rand(10) * 100 + 100 #10 numbers above 100
flier_low = np.random.rand(10) * -100 #10 numbers below 0
data = np.concatenate((spread, center, flier_high, flier_low), 0)
fig = plt.figure(figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
asd = ax.boxplot(data)
asd['medians'].label('Median')