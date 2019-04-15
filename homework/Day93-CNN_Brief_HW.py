#!/usr/bin/env python
# coding: utf-8

# # 目標:
#     運用 Keras 模組建構CNN, 了解 CNN 的架構
#     
#     

# # 範例重點
#     CNN 模型必要的: Convolution, Pooling, Flatten, Fully connection, Output, 

# # 作業¶
# (1)嘗試比對 Dense 與 layers.Conv2D 架構NN 的差異
# 
# (2) 有沒有Pooling layer, 對於參數量的差異
# 沒有使用pooling layer total params 多了一倍
# 
# 注意: input_shape 請勿修改

# In[2]:


#導入相關模組
import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense

#確認keras 版本
print(keras.__version__)


# In[5]:


'''
model = Sequential([
    Dense(xxx),
    Activation(xxx),
    Dense(xxx),
    Activation(xxx),
])
'''
#model.summary()


# # layers.Conv2D 模型, 用作比對
# 
# ![CNN_Model.png](attachment:CNN_Model.png)
# 
# 

# In[ ]:


#建立一個序列模型
model = models.Sequential()
#建立兩個卷積層, 32 個內核, 內核大小 3x3, 
#輸入影像大小 28x28x1
model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))


model.add(layers.Conv2D(25, (3, 3)))


model.add(Flatten())

#建立一個全連接層
model.add(Dense(units=100))
model.add(Activation('relu'))

#建立一個輸出層, 並採用softmax
model.add(Dense(units=10))
model.add(Activation('softmax'))

model.summary()

