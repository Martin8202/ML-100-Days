
# coding: utf-8

# ## 作業

# 請閱讀相關文獻，並回答下列問題
# 
# [脊回歸 (Ridge Regression)](https://blog.csdn.net/daunxx/article/details/51578787)
# [Linear, Ridge, Lasso Regression 本質區別](https://www.zhihu.com/question/38121173)
# 
# 1. LASSO 回歸可以被用來作為 Feature selection 的工具，請了解 LASSO 模型為什麼可用來作 Feature selection
#   因為採用L1，可以針對某些變數調整至0，這樣就可以把係數為0的變量刪除，但Ridge 沒辦法
# 2. 當自變數 (X) 存在高度共線性時，Ridge Regression 可以處理這樣的問題嗎?
#   可以，但處理效果可能不彰
