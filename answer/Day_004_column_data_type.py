
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd


# In[2]:


# 設定 data_path, 並讀取 app_train
os.chdir('F:\learning\機器學習百日馬拉松活動\Part1_資料清理數據前處理')
dir_data = './data/'
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)


# ## 作業
# 將下列部分資料片段 sub_train 使用 One Hot encoding, 並觀察轉換前後的欄位數量 (使用 shape) 與欄位名稱 (使用 head) 變化

# In[3]:


sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])
print(sub_train.shape)
sub_train.head()


# In[4]:


sub_train = pd.get_dummies(sub_train)
print(sub_train.shape)
sub_train.head()

