
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
from sklearn.ensemble import GradientBoostingClassifier

os.chdir('F:\learning\機器學習百日馬拉松活動\Part2_特徵工程')
data_path = './data/'
df = pd.read_csv(data_path + 'titanic_train.csv')

train_Y = df['Survived']
df = df.drop(['PassengerId', 'Survived'] , axis=1)
df.head()


# In[ ]:


# 因為需要把類別型與數值型特徵都加入, 故使用最簡版的特徵工程
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LEncoder.fit_transform(list(df[c].values))
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
df.head()


# In[ ]:


# 梯度提升樹擬合後, 將結果依照重要性由高到低排序 (note : D27作業中'Ticket'是第一名特徵, 'Age'是數值特徵中排名最高者)
estimator = GradientBoostingClassifier()
estimator.fit(df.values, train_Y)
feats = pd.Series(data=estimator.feature_importances_, index=df.columns)
feats = feats.sort_values(ascending=False)
feats


# ## 先用梯度提升機對鐵達尼生存預測做訓練，再用其特徵重要性回答下列問題
# 
# # 作業1
# * 將特徵重要性較低的一半特徵刪除後，再做生存率預估，正確率是否有變化? 正確率反而降低了

# In[ ]:


# 原始特徵 + 梯度提升樹
train_X = MMEncoder.fit_transform(df)
cross_val_score(estimator, train_X, train_Y, cv=5).mean()


# In[ ]:


# 高重要性特徵 + 梯度提升樹 
high_feature = list(feats[:5].index)

train_X = MMEncoder.fit_transform(df[high_feature])
cross_val_score(estimator, train_X, train_Y, cv=5).mean()


# # 作業2
# * 將特徵重要性最高的兩個特徵做特徵組合，是否能再進一步提升預測力?

# In[ ]:


# 觀察重要特徵與目標的分布
# 第一名 : Ticket              
import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x=df['Ticket'], y=train_Y, fit_reg=False)
plt.show()


# In[ ]:


# 第二名 : Name        
sns.regplot(x=df['Name'], y=train_Y, fit_reg=False)
plt.show()


# In[ ]:


# 製作新特徵看效果
df['add'] = df['Ticket']+df['Name']
df['multi'] = df['Ticket']*df['Name']
df['TN_div1p'] = df['Ticket']/(df['Name']+1)
df['NT_div1p'] = df['Name']/(df['Ticket']+1)

train_X = MMEncoder.fit_transform(df)
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

