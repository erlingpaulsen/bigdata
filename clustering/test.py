# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:50:57 2016

@author: ERLING
"""

import numpy as np
import matplotlib.pyplot as plt
import complete_linkage as cl

x = np.array([[1,1],[1,2],[2,1],[3,5],[4,4]])

plt.scatter(x[:,0], x[:,1])

(z, L) = cl.complete_linkage(x)
