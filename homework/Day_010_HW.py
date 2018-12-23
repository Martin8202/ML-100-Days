# -*- coding: utf-8 -*-
"""
Day10
Martin.Lee
2018/12/23

"""

# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# 設定 data_path
os.chdir('F:\learning\機器學習百日馬拉松活動\Part1_資料清理數據前處理')
dir_data = './data/'


f_app_train = os.path.join(dir_data, 'application_train.csv')
f_app_test = os.path.join(dir_data, 'application_test.csv')

app_train = pd.read_csv(f_app_train)
app_test = pd.read_csv(f_app_test)

from sklearn.preprocessing import LabelEncoder

# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
# also apply to testing dataset
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

# absolute the value of DAYS_BIRTH
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_test['DAYS_BIRTH'] = abs(app_test['DAYS_BIRTH'])

#相關係數
#一樣，pandas 很貼心地讓我們可以非常容易計算相關係數
corr = app_train.corr()['TARGET']

'''
練習時間
1.列出目標 (TARGET) 與所有欄位之間相關係數，數值最大以及最小各 15 個
2.通過相關係數的結果觀察有興趣的欄位與 TARGET 或其他欄位的相關係數，並嘗試找出有趣的訊息
    最好的方式當然是畫圖，舉例來說，我們知道 EXT_SOURCE_3 這個欄位和 TARGET 之間的相關係數是 -0.178919 
    (在已經這個資料集已經是最負的了！)，那我們可以 EXT_SOURCE_3 為 x 軸， TARGET 為 y 軸，把資料給畫出來
'''
#1
dow15 = corr.sort_values().head(15)
top15 = corr.sort_values().tail(15)

#2
#直接畫沒什麼意義，取log雷同
plt.plot(sub_df['DAYS_EMPLOYED'] / (-365), np.log10(sub_df['TARGET']), '.')
plt.xlabel('Days of employed (year)')
plt.ylabel('TARGET')
plt.show()

#EXT_SOURCE_3  可以看出資料分布的狀態
app_train.boxplot('EXT_SOURCE_3',by = 'TARGET',showfliers=True)
