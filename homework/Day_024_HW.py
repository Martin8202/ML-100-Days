
# coding: utf-8

# # 作業 : (Kaggle)鐵達尼生存預測
# https://www.kaggle.com/c/titanic

# # 作業1
# * 參考範例，將鐵達尼的船票票號( 'Ticket' )欄位使用特徵雜湊 / 標籤編碼 / 目標均值編碼三種轉換後，  
# 與其他數值型欄位一起預估生存機率

# In[ ]:


# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy, time, os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

os.chdir('F:\learning\機器學習百日馬拉松活動\Part2_特徵工程')
data_path = './data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()


# In[ ]:


#只取類別值 (object) 型欄位, 存於 object_features 中
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'object':
        object_features.append(feature)
print(f'{len(object_features)} Numeric Features : {object_features}\n')

# 只留類別型欄位
df = df[object_features]
df = df.fillna('None')
train_num = train_Y.shape[0]
df.head()


# # 作業2
# * 承上題，三者比較效果何者最好?

# In[ ]:


# 對照組 : 標籤編碼 + 邏輯斯迴歸
df_temp = pd.DataFrame()
for c in df.columns:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
df_temp.head()


# In[ ]:


# 加上 'Cabin' 欄位的計數編碼
count_df = df.groupby(['Cabin'])['Name'].agg({'Cabin_Count':'size'}).reset_index()
df = pd.merge(df, count_df, on=['Cabin'], how='left')
count_df.sort_values(by=['Cabin_Count'], ascending=False).head(10)

# In[ ]:


# 'Cabin'計數編碼 + 邏輯斯迴歸
df_temp = pd.DataFrame()
for c in object_features:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
df_temp['Cabin_Count'] = df['Cabin_Count']
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
df_temp.head()


# In[ ]:


# 'Cabin'特徵雜湊 + 邏輯斯迴歸
df_temp = pd.DataFrame()
for c in object_features:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
df_temp['Cabin_hash'] = df['Cabin'].map(lambda x:hash(x) % 10)
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
df_temp.head()



# In[ ]:


# 'Cabin'計數編碼 + 'Cabin'特徵雜湊 + 邏輯斯迴歸
df_temp = pd.DataFrame()
for c in object_features:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
df_temp['Cabin_hash'] = df['Cabin'].map(lambda x:hash(x) % 10)
df_temp['Cabin_Count'] = df['Cabin_Count']
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
df_temp.head()
