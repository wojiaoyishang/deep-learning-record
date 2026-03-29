# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 08:58:36 2024

@author: Administrator
@author: 梦见吃了一大碗皮卡丘
"""

# 导入数据分析库和科学计算库
import pandas as pd
import numpy as np
# 导入绘图库
from matplotlib import pyplot as plt
import seaborn as sns
# 导入决策树模块
from sklearn.tree import DecisionTreeRegressor
# 导入数据划分模块
from sklearn.model_selection import train_test_split
# 导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 导入 R2 分数模块
from sklearn.metrics import r2_score

# 加载数据
data = pd.read_csv('Salary_Data.csv')

# 数据编码
EC = {
    'Gender': {
        'Male': 1,
        'Female': 0
    },
    'Education Level': {
        'Bachelor': 0,
        'Master': 1,
        'PhD': 2
    },
}

for column in data:
    if column in EC.keys():
        try:
            data[column] = data[column].apply(lambda x: EC[column][x])
        except:
            print(f"Skipped {column}")

print('数据基本信息:', data.shape)  # 显示数据基本信息（样本数与特征数）

x = data.drop('Salary', axis=1)
y = data['Salary']

# 特征相关性分析 
corr = x[x.columns[0:]].corr()  # 求相关系数等同于协方差
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=False, annot=True, fmt='.1f')
ax.set_xticklabels(
    ax.get_xticklabels(),  # 获取 x 轴刻度标签文本
    rotation=30,
    horizontalalignment='right'
)

# 数据标准化
scaler = StandardScaler()
x_ = scaler.fit_transform(x)

# 将数据划分为训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=0.3)

# 构建决策树模型（不同深度） 
depth_list = np.arange(2, 10, 1)
R2_Train = []  # 保存训练样本拟合优度
R2_Test = []  # 保存测试样本拟合优度
for i in depth_list:
    DT = DecisionTreeRegressor(max_depth=i)
    DT.fit(x_train, y_train)
    R2_Train.append(r2_score(y_train, DT.predict(x_train)))
    R2_Test.append(r2_score(y_test, DT.predict(x_test)))

# 显示拟合优度 
plt.figure(2)
plt.plot(depth_list, R2_Train, color='r', marker='s', label='Training Data', linewidth=2)
plt.plot(depth_list, R2_Test, color='g', marker='o', label='Testing Data', linewidth=2)
plt.xlabel('Depth')
plt.ylabel('R2')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# 特征重要性 
DT = DecisionTreeRegressor(max_depth=3)
DT.fit(x_train, y_train)

# 特征数 
n_features = data.shape[1]
plt.figure(3)
plt.bar(np.arange(1, n_features, 1), np.array(DT.feature_importances_), align='center')
plt.xticks(np.arange(1, n_features, 1), data.columns[:-1], fontsize=8)
plt.xlabel('Features')
plt.ylabel('Feature importance')
plt.grid(True)
plt.show()
