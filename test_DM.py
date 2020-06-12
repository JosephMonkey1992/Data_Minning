"""
1/数据预处理： 标准化： Z-score, min-max, Maxabs, RobustScaler
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
%matplotlib inline

# data input
data = make_moons(n_samples=200, noise=10)[0]

# -*- Z-Score标准化   = (原数据-均值)/标准差
#建立StandardScaler对象
zscore = preprocessing.StandardScaler()
# 标准化处理
data_zs = zscore.fit_transform(data)

#Max-Min标准化 （归一化）   新数据=（原数据-最小值）/（最大值-最小值）
#建立MinMaxScaler对象
minmax = preprocessing.MinMaxScaler()
# 标准化处理
data_minmax = minmax.fit_transform(data)

#MaxAbs标准化
#建立MinMaxScaler对象
maxabs = preprocessing.MaxAbsScaler()
# 标准化处理
data_maxabs = maxabs.fit_transform(data)

#RobustScaler标准化   **有时候过于集中，z-score后会失去这个特性，因此robust更适合集中性的数据。
#建立RobustScaler对象
robust = preprocessing.RobustScaler()
# 标准化处理
data_rob = robust.fit_transform(data)

# 可视化数据展示
# 建立数据集列表
data_list = [data, data_zs, data_minmax, data_maxabs, data_rob]
# 创建颜色列表
color_list = ['blue', 'red', 'green', 'black', 'pink']
# 创建标题样式
title_list = ['source data', 'zscore', 'minmax', 'maxabs', 'robust']

# 设置画幅
plt.figure(figsize=(9, 6))
# 循环数据集和索引
for i, dt in enumerate(data_list):
    # 子网格
    plt.subplot(2, 3, i+1)
    # 数据画散点图
    plt.scatter(dt[:, 0], dt[:, 1], c=color_list[i])
    # 设置标题
    plt.title(title_list[i])
# 图片储存 
plt.savefig('xx.png')
# 图片展示
plt.show()


"""
 
"""
