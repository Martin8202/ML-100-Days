
# coding: utf-8

# ## 作業
# 
# 閱讀以下兩篇文獻，了解隨機森林原理，並試著回答後續的思考問題
# - [隨機森林 (random forest) - 中文](http://hhtucode.blogspot.tw/2013/06/ml-random-forest.html)
# - [how random forest works - 英文](https://medium.com/@Synced/how-random-forest-algorithm-works-in-machine-learning-3c0fe15b6674)

# 
# 1. 隨機森林中的每一棵樹，是希望能夠
# 
#     - 沒有任何限制，讓樹可以持續生長 (讓樹生成很深，讓模型變得複雜)
#     
#     - 不要過度生長，避免 Overfitting
#     ==> 雖然隨機森林可以避免overfitting, 但是對個別的樹還是應該要找出較適合的超參數, 避免模型過於複雜.
#     
# 2. 假設總共有 N 筆資料，每棵樹用取後放回的方式抽了總共 N 筆資料生成，請問這棵樹大約使用了多少 % 不重複的原資料生成?
# hint: 0.632 bootstrap
# 
