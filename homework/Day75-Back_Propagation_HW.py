#!/usr/bin/env python
# coding: utf-8

# # 作業: 
# 3  層神經網路
# 
# 通過增加更多的中間層，以對更多關係的組合進行建模
# 
# syn1 權值矩陣將隱層的組合輸出映射到最終結果，
# 
# 而在更新 syn1 的同時，還需要更新 syn0 權值矩陣，
# 
# 以從輸入資料中更好地產生這些組合

# # BP 演算法訓練的神經網路
# 
# 
# 目標: 嘗試著用輸入去預測輸出
#  
# 考慮以上情形：
# 給定三列輸入，試著去預測對應的一列輸出。
# 
# 我們可以通過簡單測量輸入與輸出值的資料來解決這一問題。
# 
# 最左邊的一列輸入值和輸出值是完美匹配/完全相關的。
# 
# 反向傳播演算法便是通過這種方式來衡量資料間統計關係進而得到模型的。
# 

# # 更新3 Layers 所需參數定義
# 
# X 輸入資料集，形式為矩陣，每 1 行代表 1 個訓練樣本。
# 
# y 輸出資料集，形式為矩陣，每 1 行代表 1 個訓練樣本。
# 
# l0 網路第 1 層，即網路輸入層。
# 
# l1 網路第 2 層，常稱作隱藏層。
# 
# l2 假定為網路最後一層，隨著訓練進行，其輸出應該逐漸接近正確結果
# 
# syn0 第一層權值
# 
# syn1 第二層權值
# 
# l2_error 該值說明了神經網路預測時“丟失”的數目。
# 
# l2_delta 該值為經確信度加權後的神經網路的誤差，除了確信誤差很小時，它近似等於預測誤差。
# 
# l1_error 該值為 l2_delta 經 syn1 加權後的結果，從而能夠計算得到中間層/隱層的誤差。
# 
# l1_delta 該值為經確信度加權後的神經網路 l1 層的誤差，除了確信誤差很小時，它近似等於 l1_error 。

# In[ ]:


import numpy as np
 
# Sigmoid 函數可以將任何值都映射到一個位於 0 到  1 範圍內的值。通過它，我們可以將實數轉化為概率值
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])  
        
# define y for output dataset            
y = np.array([[0],
              [1],
              [0],
              [1]])

# In[ ]:



# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)
#亂數設定產生種子得到的權重初始化集仍是隨機分佈的，
#但每次開始訓練時，得到的權重初始集分佈都是完全一致的。
 
# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,2)) - 1
# define syn1
syn1 = 2*np.random.random((2,1)) - 1

iter = 0
#該神經網路權重矩陣的初始化操作。
#用 “syn0” 來代指 (即“輸入層-第一層隱層”間權重矩陣）
#用 “syn1” 來代指 (即“輸入層-第二層隱層”間權重矩陣）


# 神經網路訓練
# for 迴圈反覆運算式地多次執行訓練代碼，使得我們的網路能更好地擬合訓練集

# In[ ]:


for iter in range(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    '''
    新增
    l2_error 該值說明了神經網路預測時“丟失”的數目。
    l2_delta 該值為經確信度加權後的神經網路的誤差，除了確信誤差很小時，它近似等於預測誤差。
    '''
 
    # how much did we miss?
    l1_error = y - l1
    l2_error = y - l2
 
    if (iter% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))
 
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)
 
    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
 
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)
 
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
    
print("Output After Training:")
print(l1)
print("\n\n")
print(l1)


# In[ ]:




