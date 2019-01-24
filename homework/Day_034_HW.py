
# coding: utf-8

# ## 練習時間
# 假設我們資料中類別的數量並不均衡，在評估準確率時可能會有所偏頗，試著切分出 y_test 中，0 類別與 1 類別的數量是一樣的 (亦即 y_test 的類別是均衡的)

# In[2]:


import numpy as np
X = np.arange(1000).reshape(200, 5)
y = np.zeros(200)
y[:40] = 1


# In[3]:


y


# 可以看見 y 類別中，有 160 個 類別 0，40 個 類別 1 ，請試著使用 train_test_split 函數，切分出 y_test 中能各有 10 筆類別 0 與 10 筆類別 1 。(HINT: 參考函數中的 test_size，可針對不同類別各自作切分後再合併)

from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X[:40], y[:40], test_size=10/40)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X[40:], y[40:], test_size=10/160)