# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 08:53:45 2024

@author: Administrator
"""
# 导入科学计算库
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# 导入决策树模块
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# 导入红酒数据库
from sklearn.datasets import load_wine
# 导入数据划分模块
from sklearn.model_selection import train_test_split
# 导入数据标准化模块
from sklearn.preprocessing import StandardScaler

# 加载红酒数据
wine = load_wine()
print('数据基本信息:', wine.data.shape)
print('特征名称：', wine.feature_names)
# 分离特征与类别标记
x = wine.data
y = wine.target
# 数据标准化
scaler = StandardScaler()
x_ = scaler.fit_transform(x)
# 将数据划分为训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=0.3, random_state=10)
# 构建决策树模型[不同深度]
depth_list = np.arange(2, 10, 1)
Acc_Train = []
Acc_Test = []
for i in depth_list:
    DT = DecisionTreeClassifier(max_depth=i, random_state=10)
    DT.fit(x_train, y_train)
    Acc_Train.append(DT.score(x_train, y_train))
    Acc_Test.append(DT.score(x_test, y_test))
# 显示精度变化
plt.figure(1)
plt.plot(depth_list, Acc_Train, color='r', marker='s', label='Training Data', linewidth=2)
plt.plot(depth_list, Acc_Test, color='g', marker='o', label='Testing Data', linewidth=2)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()
# 显示特征的重要性
DT = DecisionTreeClassifier(max_depth=4, random_state=10)
DT.fit(x_train, y_train)
feature_importance = DT.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
plt.figure(2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(wine.feature_names)[sorted_idx])
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.grid(True)
plt.show()
# plt.title('Feature Importance')
# 绘制决策树
tree.plot_tree(DT)
