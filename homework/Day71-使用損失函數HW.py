#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[ ]:


from keras.datasets import cifar10
import numpy as np
np.random.seed(10)


# # 資料準備

# In[ ]:


#取得Keras Dataset
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()


# In[ ]:


#確認 CIFAR10 Dataset 資料維度
print("train data:",'images:',x_img_train.shape,
      " labels:",y_label_train.shape) 
print("test  data:",'images:',x_img_test.shape ,
      " labels:",y_label_test.shape) 


# In[ ]:


#資料正規化
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0


# In[ ]:


#針對Label 做 ONE HOT ENCODE
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
y_label_test_OneHot.shape


# # 建立模型

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D


# In[ ]:


model = Sequential()


# In[ ]:


#卷積層1


# In[ ]:


model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(32, 32,3), 
                 activation='relu', 
                 padding='same'))


# In[ ]:


model.add(Dropout(rate=0.25))


# In[ ]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


#卷積層2與池化層2


# In[ ]:


model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))


# In[ ]:


model.add(Dropout(0.25))


# In[ ]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


#建立神經網路(平坦層、隱藏層、輸出層)


# In[ ]:


model.add(Flatten())
model.add(Dropout(rate=0.25))


# In[ ]:


model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))


# In[ ]:


model.add(Dense(10, activation='softmax'))


# In[ ]:


#檢查model 的STACK
print(model.summary())


# # 載入之前訓練的模型

# In[ ]:


try:
    model.load_weights("SaveModel/cifarCnnModel.h5")
    print("載入模型成功!繼續訓練模型")
except :    
    print("載入模型失敗!開始訓練一個新模型")


# # 訓練模型

# # 作業: 
#     請分別選用 "MSE", "binary _crossentropy"
#     查看Train/test accurancy and loss rate

# In[ ]:



import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

'''
作業:
請分別選用 "MSE", "binary _crossentropy"
查看Train/test accurancy and loss rate
'''


# In[ ]:

## MSE
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

train_history =model.fit(x=x_img_train_normalize,
                         y=y_label_train_OneHot,validation_split=0.2, 
                         epochs=10, batch_size=32,verbose=1)

show_train_history('acc','val_acc')
show_train_history('loss','val_loss')


scores = model.evaluate(x_img_test_normalize, y_label_test_OneHot)
print()
print('accuracy=',scores[1])

#binary _crossentropy
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

train_history =model.fit(x=x_img_train_normalize,
                         y=y_label_train_OneHot,validation_split=0.2, 
                         epochs=10, batch_size=32,verbose=1)

show_train_history('acc','val_acc')
show_train_history('loss','val_loss')


scores = model.evaluate(x_img_test_normalize, y_label_test_OneHot)
print()
print('accuracy=',scores[1])