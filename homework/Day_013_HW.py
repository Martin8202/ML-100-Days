# -*- coding: utf-8 -*-
"""
Day 13
Martin.lee
2018.12.25
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


# #### 等寬劃分
app_train["equal_width_DAYS_BIRTH"] = pd.cut(app_train['DAYS_BIRTH'], 4)
app_train["equal_width_DAYS_BIRTH"].value_counts()

# #### 等頻劃分
app_train["equal_width_DAYS_BIRTH"] = pd.qcut(app_train['DAYS_BIRTH'], 4)
app_train["equal_width_DAYS_BIRTH"].value_counts()

# #### 自定義的 bin
app_train["equal_width_DAYS_BIRTH"] = pd.cut(app_train['DAYS_BIRTH'], 10)
app_train["equal_width_DAYS_BIRTH"].value_counts()

