
# coding: utf-8

# # 作業
# 請使用不同的資料集，並使用 hyper-parameter search 的方式，看能不能找出最佳的超參數組合

from sklearn import datasets, metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, KFold, GridSearchCV



# 讀取鳶尾花資料集
iris = datasets.load_iris()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)

# 建立模型
clf = GradientBoostingClassifier()

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)


# 先看看使用預設參數得到的結果，約為 0.2631 的 MSE
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(metrics.mean_squared_error(y_test, y_pred))


# 設定要訓練的超參數組合
n_estimators = [10, 100, 1000]
max_depth = [1, 5, 10]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)

## 建立搜尋物件，放入模型及參數組合字典 (n_jobs=-1 會使用全部 cpu 平行運算)
grid_search = GridSearchCV(clf, param_grid, scoring="accuracy", verbose=1) #, n_jobs=-1

# 開始搜尋最佳參數
grid_result = grid_search.fit(x_train, y_train)

#預設會跑 3-fold cross-validadtion，總共 9 種參數組合，總共要 train 27 次模型

# 印出最佳結果與最佳參數
print("Best Accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

grid_result.best_params_


# 使用最佳參數重新建立模型
clf_bestparam = GradientBoostingClassifier(max_depth=grid_result.best_params_['max_depth'],
                                           n_estimators=grid_result.best_params_['n_estimators'])

# 訓練模型
clf_bestparam.fit(x_train, y_train)

# 預測測試集
y_pred = clf_bestparam.predict(x_test)

# 調整參數後約可降至 0.02 的 MSE
print(metrics.mean_squared_error(y_test, y_pred))
