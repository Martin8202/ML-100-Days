
# coding: utf-8

# ## 練習時間
# 在小量的資料上，我們用眼睛就可以看得出來程式碼是否有跑出我們理想中的結果
# 
# 請嘗試想像一個你需要的資料結構 (裡面的值可以是隨機的)，然後用上述的方法把它變成 pandas DataFrame
# 
# #### Ex: 想像一個 dataframe 有兩個欄位，一個是國家，一個是人口，求人口數最多的國家
# 
# ### Hints: [隨機產生數值](https://blog.csdn.net/christianashannon/article/details/78867204)

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = {'國家': ['Taiwan', 'United States', 'Thailand'],
        '人口': np.random.randint(low=10000, high=1000000, size=3)}
data = pd.DataFrame(data)


# #### 求人口數最多的國家，方法有很多種，這裡列舉其二

# In[3]:


data.loc[data['人口']==data['人口'].max()]['國家']


# In[4]:


data.iloc[data['人口'].idxmax()]['國家']

