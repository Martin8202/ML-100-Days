
# coding: utf-8

# ## 作業

# 請閱讀以下相關文獻，並回答以下問題
# 
# [Linear Regression 詳細介紹](https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-3%E8%AC%9B-%E7%B7%9A%E6%80%A7%E5%88%86%E9%A1%9E-%E9%82%8F%E8%BC%AF%E6%96%AF%E5%9B%9E%E6%AD%B8-logistic-regression-%E4%BB%8B%E7%B4%B9-a1a5f47017e5)
# 
# [Logistics Regression 詳細介紹](https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-3%E8%AC%9B-%E7%B7%9A%E6%80%A7%E5%88%86%E9%A1%9E-%E9%82%8F%E8%BC%AF%E6%96%AF%E5%9B%9E%E6%AD%B8-logistic-regression-%E4%BB%8B%E7%B4%B9-a1a5f47017e5)
# 
# [你可能不知道的 Logisitc Regression](https://taweihuang.hpd.io/2017/12/22/logreg101/)
# 

# 1. 線性回歸模型能夠準確預測非線性關係的資料集嗎?
#   理論上不行，由於資料是非線性，用線性的方法預測自然不會找出好結果
# 2. 回歸模型是否對資料分布有基本假設?
    #有，共有五個假設
    #1.線性假設：資料屬於線性
    #2.不偏性：誤差期望值為0→每個X之間獨立
    #3.不偏性：隨機樣本
    #4.有效性：同質變異數假設
    #5.有效性：服從常態分配
# 
