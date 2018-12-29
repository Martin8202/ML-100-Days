# -*- coding: utf-8 -*-
"""
Day 15
Martin.Lee
2018/12/29
"""
'''
作業
1.請用 numpy 建立一個 10 x 10, 數值分布自 -1.0 ~ 1.0 的矩陣並繪製 Heatmap
2.請用 numpy 建立一個 1000 x 3, 數值分布為 -1.0 ~ 1.0 的矩陣，並繪製 PairPlot (上半部為 scatter, 對角線為 hist, 下半部為 density)
3.請用 numpy 建立一個 1000 x 3, 數值分布為常態分佈的矩陣，並繪製 PairPlot (上半部為 scatter, 對角線為 hist, 下半部為 density)
'''

# Import 需要的套件
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 另一個繪圖-樣式套件
plt.style.use('ggplot')

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

#1.請用 numpy 建立一個 10 x 10, 數值分布自 -1.0 ~ 1.0 的矩陣並繪製 Heatmap

matrix = np.random.random((10,10))*2-1

plt.figure(figsize=(10,10))

heatmap = sns.heatmap(matrix, cmap = plt.cm.RdYlBu_r #颜色設定 默认为cubehelix map (数据集为连续数据集时) 或 RdBu_r (数据集为离散数据集时)
            , vmin = -1, vmax = 1 #center 為平均值,vmin and vmax為最大最小值，若center為non，則center為vmin&vmax平均
            , annot = True) #寫入數據

plt.show()

#2.請用 numpy 建立一個 1000 x 3, 數值分布為 -1.0 ~ 1.0 的矩陣，並繪製 PairPlot (上半部為 scatter, 對角線為 hist, 下半部為 density)
nrow = 1000
ncol = 3

matrix = np.random.random((nrow,ncol))*2-1

indice = np.random.choice([0,1,2], size=nrow)
plot_data = pd.DataFrame(matrix, indice)


grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False)


grid.map_upper(plt.scatter , alpha = 0.2)
grid.map_diag(plt.hist)
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)

plt.show()

#3.請用 numpy 建立一個 1000 x 3, 數值分布為常態分佈的矩陣，並繪製 PairPlot (上半部為 scatter, 對角線為 hist, 下半部為 density)
nrow = 1000
ncol = 3
"""
Your Code Here
"""
matrix = np.random.randn(nrow,ncol)

indice = np.random.choice([0,1,2], size=nrow)
plot_data = pd.DataFrame(matrix, indice)


grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False)

grid.map_upper(plt.scatter , alpha = 0.2)
grid.map_diag(plt.hist)
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)

plt.show()