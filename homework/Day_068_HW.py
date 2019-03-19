
# coding: utf-8

# In[1]:


# 載入必須使用的 Library
import keras
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


# # 作業:
# 請修改input shape: (Conv2D(64, (3, 3))的設定, 新增一層 Dense 並觀看 model.summary 的輸出

# In[5]:


# build our CNN model, 多加幾層
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dense(num_classes))
model.add(Dense(16))
model.add(Activation('softmax'))

print(model.summary())