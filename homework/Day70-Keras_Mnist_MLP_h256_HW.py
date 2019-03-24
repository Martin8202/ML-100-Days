#!/usr/bin/env python
# coding: utf-8

# # 作業:
# [練習]請嘗試新增一層或兩層看看最後預測結果是否有差異
# model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))

# # 資料預處理

# In[ ]:


from keras.utils import np_utils
import numpy as np
np.random.seed(10)


# In[ ]:


#載入手寫辨識的資料集
from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label)= mnist.load_data()


# In[ ]:


#指定測試集與訓練資料集
x_Train =x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')


# In[ ]:


# normalize inputs from 0-255 to 0-1
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255


# In[ ]:


#把LABEL轉成NUMERICAL Categorical 
y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)


# # 建立模型

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


#建立模型


# In[ ]:

model = Sequential()

#建構輸入層
model.add(Dense(units=256, 
                input_dim=784, 
                kernel_initializer='normal', 
                activation='relu'))

model.add(Dense(units=15, input_dim=250, kernel_initializer='normal', activation='relu'))

#建構輸出層
model.add(Dense(units=10, 
                kernel_initializer='normal', 
                activation='softmax'))

print(model.summary())



# # 訓練模型

# In[ ]:


model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])


# In[ ]:


train_history =model.fit(x=x_Train_normalize,
                         y=y_Train_OneHot,validation_split=0.2, 
                         epochs=10, batch_size=32,verbose=1)


# # 以圖形顯示訓練過程

# In[ ]:


import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[ ]:


show_train_history(train_history,'acc','val_acc')


# In[ ]:


show_train_history(train_history,'loss','val_loss')


# # 評估模型準確率

# In[ ]:


scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
print()
print('accuracy=',scores[1])

