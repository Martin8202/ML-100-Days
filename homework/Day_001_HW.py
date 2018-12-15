# -*- coding: utf-8 -*-
"""
Day 01 資料介紹與評估指標
author： Martin.Lee
Date：2018/12/15
"""
#==============================================================================
'''
作業1：
請上 Kaggle, 在 Competitions 或 Dataset 中找一組競賽或資料並寫下：

Data：New York City Taxi Fare Prediction https://www.kaggle.com/c/new-york-city-taxi-fare-prediction

1. 你選的這組資料為何重要
Ans：透過預測紐約市計程車的票價與通行費，可以預估何種路線能夠接到最多客人，
    增加計程車司機收入

2. 資料從何而來 (tips: 譬如提供者是誰、以什麼方式蒐集)
提供者：Goolge cloud

3. 蒐集而來的資料型態為何
結構化，數值資料

4. 這組資料想解決的問題如何評估
透過test的資料確認模型預估的準確性
'''
#====================================================================================
'''
作業2：

想像你經營一個自由載客車隊，你希望能透過數據分析以提升業績，請你思考並描述你如何規劃整體的分析/解決方案：

1. 核心問題為何 (tips：如何定義 「提升業績 & 你的假設」)
    利用資料預測乘客聚集地點與時間，調整車輛調度狀況與乘客收費標準，以提升利潤

2. 資料從何而來 (tips：哪些資料可能會對你想問的問題產生影響 & 資料如何蒐集)
    a.爬蟲--蒐集機票，以推估來客數量，or，蒐集其他競爭對手的資料，以確認定價
    b.計程車公會--平均月載客數
    c.觀光局網站--旅客數量
    d.乘客的app紀錄與司機的駕車系統
    e.氣象局--天氣資訊

3. 蒐集而來的資料型態為何
    a.機票價格為數字屬於結構資料，競爭對手的評價為文字屬於非結構資料
    b.數值表格
    c.數值表格
    d.行經路線(非結構化)、行徑距離(結構化數值)、價格收入(結構化數值)
    e.數值資料
    
4. 你要回答的問題，其如何評估 (tips：你的假設如何驗證)
    a.檢驗司機空車時間是否縮短
    b.尖峰時刻總金額是否提升
    c.比較調整前後，乘客滿意度的變化

'''
#=========================================================================================
'''
作業3：

練習時間
請寫一個函式用來計算 Mean Square Error
$ MSE = \frac{1}{n}\sum_{i=1}^{n}{(Y_i - \hat{Y}_i)^2} $
Hint: 如何取平方
'''

import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y, yp):
    """
    計算MSE
    Args:
        - y: 實際值
        - yp: 預測值
    Return:
        - mse: MSE
    """
    mse = MSE = sum(abs(y - yp)**2) / len(y)
    return mse

def mean_absolute_error(y, yp):
    """
    計算 MAE
    Args:
        - y: 實際值
        - yp: 預測值
    Return:
        - mae: MAE
    """
    mae = MAE = sum(abs(y - yp)) / len(y)
    return mae

w = 3
b = 0.5

x_lin = np.linspace(0, 100, 101)

y = (x_lin + np.random.randn(101) * 5) * w + b

plt.plot(x_lin, y, 'b.', label = 'data points')
plt.title("Assume we have data points")
plt.legend(loc = 2)
plt.show()

y_hat = x_lin * w + b
plt.plot(x_lin, y, 'b.', label = 'data')
plt.plot(x_lin, y_hat, 'r-', label = 'prediction')
plt.title("Assume we have data points (And the prediction)")
plt.legend(loc = 2)
plt.show()

# 執行 Function, 確認有沒有正常執行
MSE = mean_squared_error(y, y_hat)
MAE = mean_absolute_error(y, y_hat)
print("The Mean squared error is %.3f" % (MSE))
print("The Mean absolute error is %.3f" % (MAE))
