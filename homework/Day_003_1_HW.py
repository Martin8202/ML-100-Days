# -*- coding: utf-8 -*-
"""
Day 03_1 新建一個Dataframe
author： Martin.Lee
Date：2018/12/15

練習時間
在小量的資料上，我們用眼睛就可以看得出來程式碼是否有跑出我們理想中的結果

請嘗試想像一個你需要的資料結構 (裡面的值可以是隨機的)，然後用上述的方法把它變成 pandas DataFrame

Ex: 想像一個 dataframe 有兩個欄位，一個是國家，一個是人口，求人口數最多的國家

"""


import pandas as pd
import numpy as np

#生成一组随机数列表 https://blog.csdn.net/christianashannon/article/details/78867204

countries = ['Taiwan','U.S.A','Japan','China']
alldata = {'countries' : countries,
           'people' : np.random.randint(1e7,size =len(countries))}
data = pd.DataFrame(alldata)

print('人口數最多的國家: %s' %data.sort_values(by='people', ascending=False).ix[0,0])
