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
# 创建标题样式处理
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
 2/数据清洗：重复值处理
"""
sample = pd.DataFrame({'id':[1,1,1,1,3,4,5],
                       'name':['Mark','Bob','Bob','Mark','Miki','Sully','Rose'],
                       'score':[80,99,99,87,77,77,np.nan],
                       'group':[2,1,1,1,2,1,2],})

# 发现重复值的方法: 重复的判断为所有字段
sample[sample.duplicated()]
# 发现重复值的方法: 重复的判断为规定的字段
sample[sample.duplicated(['id','name'])]

# 去除重复值：去除那个所有字段都重复的record
sample_1 = sample.drop_duplicates()
# 去除重复值：去除规定的字段
sample_2 = sample.drop_duplicates(['id','name'])

"""
 2/数据清洗：缺失值的处理

 *//首先，需要根据业务理解处理缺失值，弄清楚缺失值产生的原因是故意缺失还是随机缺失，再通过一些业务经验进行填补。
 一般来说当缺失值少于20%时，连续变量可以使用均值或中位数填补；分类变量不需要填补，单算一类即可，或者也可以用众数填补分类变量。
当缺失值处于20%-80%之间时，填补方法同上。另外每个有缺失值的变量可以生成一个指示哑变量，参与后续的建模。
当缺失值多于80%时，每个有缺失值的变量生成一个指示哑变量，参与后续的建模，不使用原始变量。
"""

# 查看缺失值
sample_1.apply(lambda col:sum(col.isnull())/col.size)

# 以指定值弥补 (还可以中位数弥补)
sample_1.score = sample_1.score.fillna(sample_1.score.mean())
sample_1.score = sample_1.score.fillna(sample_1.score.median())

# 缺失值指示变量
sample_1.score.isnull()
# 若想转换为数值0，1型指示变量，可以使用apply方法，int表示将该列替换为int类型。
sample_1.score.isnull().apply(int)

"""
 2/数据清洗: 噪声处理
"""
