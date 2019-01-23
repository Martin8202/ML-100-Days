
# coding: utf-8

# ## 練習時間

# 請觀看李宏毅教授以神奇寶貝進化 CP 值預測的範例，解說何謂機器學習與過擬合。並回答以下問題

# In[1]:


from IPython.display import YouTubeVideo
YouTubeVideo("fegAeph9UaA", width=720, height=480)


# ### 1. 模型的泛化能力 (generalization) 是指什麼？ 
#模型本身能否包含所有資料的訊息，產生舉一反三的效果。避免模型與trainning data太過接近，造成overfitting
# ### 2. 分類問題與回歸問題分別可用的目標函數有哪些？
#分類問題：Accuracy、F1-score、AUC(Area Under Curve)
#回歸問題：MSE、MAE(mean absolute error)、R-square
