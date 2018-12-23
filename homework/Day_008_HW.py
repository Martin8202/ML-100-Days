# -*- coding: utf-8 -*-
"""
Day08
Martin.Lee
2018/12/23

"""

# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline


# 設定 data_path
os.chdir('F:\learning\機器學習百日馬拉松活動\Part1_資料清理數據前處理')
dir_data = './data/'


f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()

'''
作業
1.請將 app_train 中的 CNT_CHILDREN 依照下列規則分為四組，並將其結果在原本的 dataframe 命名為 CNT_CHILDREN_GROUP
    0 個小孩
    有 1 - 2 個小孩
    有 3 - 5 個小孩
    有超過 5 個小孩
2.請根據 CNT_CHILDREN_GROUP 以及 TARGET，列出各組的平均 AMT_INCOME_TOTAL，並繪製 baxplot
3.請根據 CNT_CHILDREN_GROUP 以及 TARGET，對 AMT_INCOME_TOTAL 計算 Z 轉換 後的分數
'''

#1

cut_rule =  [-1,0,2,5,100] # 需要加上下限:/
#cut函数是利用數值區間將數值分類
app_train['CNT_CHILDREN_GROUP'] = pd.cut(app_train['CNT_CHILDREN'].values, cut_rule, include_lowest=True)
app_train['CNT_CHILDREN_GROUP'].value_counts()

#2-1

grp = ['CNT_CHILDREN_GROUP','TARGET']

grouped_df = app_train.groupby(grp)['AMT_INCOME_TOTAL']
grouped_df.mean()

#2-2
plt_column = ['AMT_INCOME_TOTAL']
plt_by = ['CNT_CHILDREN_GROUP','TARGET']

app_train.boxplot(column=plt_column, #以什麼資料話箱型圖
                  by = plt_by, #箱線圖x的條件
                  showfliers = False, #F = 箱型圖 T=散步圖
                  figsize=(12,12))
plt.suptitle('')
plt.show()

#3

app_train['AMT_INCOME_TOTAL_Z_BY_CHILDREN_GRP-TARGET'] = grouped_df.apply(lambda x: (x-x.mean())/x.std())

app_train[['AMT_INCOME_TOTAL','AMT_INCOME_TOTAL_Z_BY_CHILDREN_GRP-TARGET']].head()
