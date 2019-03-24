#!/usr/bin/env python
# coding: utf-8

# # Rectified Linear Unit- Relu 
# 
# f(x)=max(0,x)
# 

# In[ ]:


import numpy as np
from numpy import *
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

'''
作業:
    寫出 ReLU & dReLU 一階導數
    並列印
'''

def ReLU(x):
    return abs(x) * (x > 0)

def dReLU(x):
    return (1 * (x > 0))

# linespace generate an array from start and stop value
# with requested number of elements.
x = plt.linspace(-10,10,100)

# prepare the plot, associate the color r(ed) or b(lue) and the label 
plt.plot(x, ReLU(x), 'r')
plt.plot(x, dReLU(x), 'b')


# Draw the grid line in background.
plt.grid()

# Title
plt.title('ReLU Function')

# write the ReLU formula
plt.text(0, 9, r'$f(x)= (abs(x) * (x > 0))$', fontsize=15)

# create the graph
plt.show()