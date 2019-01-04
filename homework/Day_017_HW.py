
# coding: utf-8

# # 作業 : (Kaggle)鐵達尼生存預測精簡版 
# https://www.kaggle.com/c/titanic

# # 作業1
# * 下列A~E五個程式區塊中，哪一塊是特徵工程?
# answer is C
#
# # 作業2
# * 對照程式區塊 B 與 C 的結果，請問那些欄位屬於"類別型欄位"? (回答欄位英文名稱即可) 
# answers are 'Name, Sex, Ticket, Cabin, Embarked
#
# # 作業3
# * 續上題，請問哪個欄位是"目標值"?
# answer is 'Survived'

# In[ ]:


# 程式區塊 A
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

os.chdir('F:\learning\機器學習百日馬拉松活動\Part2_特徵工程')
data_path = './data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')
df_train.shape


# In[ ]:


# 程式區塊 B
train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()


# In[ ]:


# 程式區塊 C
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LEncoder.fit_transform(list(df[c].values))
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
df.head()


# In[ ]:


# 程式區塊 D
train_num = train_Y.shape[0]
train_X = df[:train_num]
test_X = df[train_num:]

from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression()
estimator.fit(train_X, train_Y)
pred = estimator.predict(test_X)


# In[ ]:


# 程式區塊 E
sub = pd.DataFrame({'PassengerId': ids, 'Survived': pred})
sub.to_csv('titanic_baseline.csv', index=False) 

