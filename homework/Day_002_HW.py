# -*- coding: utf-8 -*-
"""
D2：EDA-1/讀取資料 EDA: Data summary
author： Martin.Lee
Date：2018/12/15
"""
import pandas as pd
import numpy as np
import os

os.chdir('F:\\learning\\機器學習百日馬拉松活動\\Part1_資料清理數據前處理') #設定路徑
#1_用 pd.read_csv 來讀取資料
#f_app = os.path.join(dir_data, 'application_train.csv') #将多个路径组合后返回
#print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv('data/application_train.csv')

#Note: 在 jupyter notebook 中，可以使用 ? 來調查函數的定義
#?pd.read_csv

#2_接下來我們可以用 .head() 這個函數來觀察前 5 row 資料
app_train.head()


#3_資料的 row 數以及 column 數
print("row 數：%d\ncolumn 數：%d" % app_train.shape) #\代表換行，%執行的function
#row 數：307511
#column 數：122

#4_列出所有欄位
print(app_train.columns)

#5_截取部分資料
#擷取SK_ID_CURR最後五筆
app_train['SK_ID_CURR'].tail()
