
# coding: utf-8

# # 作業 : (Kaggle)鐵達尼生存預測
# ***
# https://www.kaggle.com/c/titanic

# In[ ]:


# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy, os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

os.chdir('F:\learning\機器學習百日馬拉松活動\Part2_特徵工程')
data_path = './data/'
df = pd.read_csv(data_path + 'titanic_train.csv')

train_Y = df['Survived']
df = df.drop(['PassengerId'] , axis=1)
df.head()


# In[ ]:


# 計算df整體相關係數, 並繪製成熱圖
import seaborn as sns
import matplotlib.pyplot as plt
corr = df.corr()
sns.heatmap(corr)
plt.show()


# In[ ]:


# 記得刪除 Survived
df = df.drop(['Survived'] , axis=1)

#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
df = df.fillna(-1)
MMEncoder = MinMaxScaler()
df.head()


# # 作業1
# * 鐵達尼生存率預測中，試著變更兩種以上的相關係數門檻值，觀察預測能力是否提升? 沒改變

# In[ ]:


# 原始特徵 + 邏輯斯迴歸
train_X = MMEncoder.fit_transform(df)
estimator = LogisticRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()


# In[ ]:


# 篩選相關係數大於 0.2 或小於 -0.2 的特徵
high_list = list(corr[(corr['Survived']>0.2) | (corr['Survived']<-0.2)].index)
high_list.pop(-3) #pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
print(high_list)



# In[ ]:


# 特徵1 + 邏輯斯迴歸
train_X = MMEncoder.fit_transform(df[high_list])
cross_val_score(estimator, train_X, train_Y, cv=5).mean()


# In[ ]:



# 篩選相關係數大於 0.2 或小於 -0.2 的特徵
high_list = list(corr[(corr['Survived']>0.26) | (corr['Survived']<-0.26)].index)
high_list.pop(-2) #pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
print(high_list)



# In[ ]:


# 特徵2 + 邏輯斯迴歸
train_X = MMEncoder.fit_transform(df[high_list])
cross_val_score(estimator, train_X, train_Y, cv=5).mean()


# # 作業2
# * 續上題，使用 L1 Embedding 做特徵選擇(自訂門檻)，觀察預測能力是否提升?

# In[ ]:


from sklearn.linear_model import Lasso
L1_Reg = Lasso(alpha=0.05)
train_X = MMEncoder.fit_transform(df)
L1_Reg.fit(train_X, train_Y)
L1_Reg.coef_


# In[ ]:


from itertools import compress
L1_mask = list((L1_Reg.coef_>0) | (L1_Reg.coef_<0))
L1_list = list(compress(list(df), list(L1_mask)))
L1_list


# In[ ]:


# L1_Embedding 特徵 + 線性迴歸
train_X = MMEncoder.fit_transform(df[L1_list])
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

