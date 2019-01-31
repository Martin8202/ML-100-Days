
# coding: utf-8

# ## 作業
# 
# 1. 試著調整 DecisionTreeClassifier(...) 中的參數，並觀察是否會改變結果？

from sklearn import datasets, metrics

# 如果是分類問題，請使用 DecisionTreeClassifier，若為回歸問題，請使用 DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# 讀取鳶尾花資料集
iris = datasets.load_iris()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)

# 建立模型
clf = DecisionTreeClassifier(criterion= 'entropy',
                             max_depth= 3)

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)


acc = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy: ", acc)


# 2. 改用其他資料集 (boston, wine)，並與回歸模型的結果進行比較
boston = datasets.load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25, random_state=4)

Reg = DecisionTreeRegressor(criterion= 'mae')

# 訓練模型
Reg.fit(x_train, y_train)

# 預測測試集
y_pred = Reg.predict(x_test)


MSE = metrics.mean_absolute_error(y_test, y_pred)
print("MAE: ", MSE)
