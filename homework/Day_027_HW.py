
# coding: utf-8

# # 作業 : (Kaggle)鐵達尼生存預測
# ***
# https://www.kaggle.com/c/titanic

# # 作業1
# * 試著使用鐵達尼號的例子，創立兩種以上的群聚編碼特徵( mean、median、mode、max、min、count 均可 )

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
df = df.drop(['PassengerId', 'Survived'] , axis=1)
df.head()


# In[ ]:


# 取一個類別型欄位, 與一個數值型欄位, 做群聚編碼
df['Sex'] = df['Sex'].fillna('None')
mean_df = df.groupby(['Sex'])['Fare'].mean().reset_index()
mode_df = df.groupby(['Sex'])['Fare'].apply(lambda x: x.mode()[0]).reset_index()
median_df = df.groupby(['Sex'])['Fare'].median().reset_index()
max_df = df.groupby(['Sex'])['Fare'].max().reset_index()
temp = pd.merge(mean_df, mode_df, how='left', on=['Sex'])
temp = pd.merge(temp, median_df, how='left', on=['Sex'])
temp = pd.merge(temp, max_df, how='left', on=['Sex'])
temp.columns = ['Sex', 'Fare_Mean', 'Fare_Mode', 'Fare_Median', 'Fare_Max']
df = pd.merge(df, temp, how='left', on=['Sex'])
df = df.drop(['Sex'] , axis=1)
df.head()


# In[ ]:


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


# # 作業2
# * 將上述的新特徵，合併原有的欄位做生存率預估，結果是否有改善? 有

# In[ ]:
df_minus = df.drop(['Fare_Mean', 'Fare_Mode', 'Fare_Median', 'Fare_Max'] , axis=1)

# 原始特徵 + 邏輯斯迴歸
train_X = MMEncoder.fit_transform(df_minus)
estimator = LogisticRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()


# In[ ]:


# 新特徵 + 邏輯斯迴歸
train_X = MMEncoder.fit_transform(df)
cross_val_score(estimator, train_X, train_Y, cv=5).mean()


