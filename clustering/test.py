# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:50:57 2016

@author: ERLING
"""

import numpy as np
import matplotlib.pyplot as plt
import complete_linkage as cl

x = np.array([[1,1],[2,2],[3,3]])

#plt.scatter(x[:,0], x[:,1])

cl.complete_linkage(x)