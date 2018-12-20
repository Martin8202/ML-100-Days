
# coding: utf-8

# ## 練習時間
# 資料的操作有很多，接下來的馬拉松中我們會介紹常被使用到的操作，參加者不妨先自行想像一下，第一次看到資料，我們一般會想知道什麼訊息？
# 
# #### Ex: 如何知道資料的 row 數以及 column 數、有什麼欄位、多少欄位、如何截取部分的資料等等
# 
# 有了對資料的好奇之後，我們又怎麼通過程式碼來達成我們的目的呢？
# 
# #### 可參考該[基礎教材](https://bookdata.readthedocs.io/en/latest/base/01_pandas.html#DataFrame-%E5%85%A5%E9%97%A8)或自行 google

# In[1]:


import os
import numpy as np
import pandas as pd


# In[2]:


# 設定 data_path
dir_data = 'F:\\learning\\機器學習百日馬拉松活動\\Part1_資料清理數據前處理\\data\\'


# In[3]:


f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)


# #### 資料的 row 數以及 column 數

# In[4]:


print(app_train.shape) # 有 307511 row 以及 122 column


# #### 列出所有欄位

# In[5]:


app_train.columns


# #### 截取部分資料

# In[6]:


app_train.iloc[:10, 0:5] # 前 10 row 以及前 5 個 column


# #### 還有各種數之不盡的資料操作，重點還是取決於實務中遇到的狀況和你想問的問題
