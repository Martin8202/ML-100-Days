# -*- coding: utf-8 -*-
"""
今日作業

觀察有興趣的欄位的資料分佈，並嘗試找出有趣的訊息
舉例來說：
    計算任意欄位的平均數及標準差
    畫出任意欄位的直方圖

Descriptive Statistics For pandas Dataframe：https://chrisalbon.com/python/data_wrangling/pandas_dataframe_descriptive_stats/
pandas 中的繪圖函數 :https://amaozhao.gitbooks.io/pandas-notebook/content/pandas%E4%B8%AD%E7%9A%84%E7%BB%98%E5%9B%BE%E5%87%BD%E6%95%B0.html
敘述統計與機率分析：http://www.hmwu.idv.tw/web/R_AI_M/AI-M1-hmwu_R_Stat&Prob.pdf
Standard Statistical Distributions：https://www.healthknowledge.org.uk/public-health-textbook/research-methods/1b-statistical-methods/statistical-distributions
List of probability distributions：https://en.wikipedia.org/wiki/List_of_probability_distributions
"""

import os
import numpy as np
import pandas as pd



os.chdir('F:\learning\機器學習百日馬拉松活動\Part1_資料清理數據前處理')
# 設定 data_path
dir_data = './data/'
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)

import matplotlib.pyplot as plt
%matplotlib inline
#計算任意欄位的平均數及標準差
print("AMT_ANNUITY Mean:%s ,AMT_ANNUITY Variance:%s"%(app_train['AMT_ANNUITY'].mean(),app_train['AMT_ANNUITY'].var()))
#畫出任意欄位的直方圖
app_train_norm = (app_train['AMT_ANNUITY']-app_train['AMT_ANNUITY'].mean())/(app_train['AMT_CREDIT'].max()-app_train['AMT_CREDIT'].min())
app_train_norm.hist(bins=50)
