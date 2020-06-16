"""
1/数据预处理： 标准化： Z-score, min-max, Maxabs, RobustScaler
"""
import math
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
#建立MinMaxScaler对象等深分箱
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
# 创建颜色列表等深分箱
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
"""等深分箱
sample = pd.DataFrame({'id':[1,1,1,1,3,4,5],
                       'name':['Mark','Bob','Bob','Mark','Miki','Sully','Rose'],
                       'score':[80,99,99,87,77,77,np.nan],
                       'group':silhouette_score字段
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
 2/数据清洗: 噪声处理 (异常值，离群值)

 *//噪声值的处理方法很多，对于单变量，常见的方法有盖帽法、分箱法；多变量的处理方法为聚类法。
"""
"""
盖帽法
*//盖帽法本质就是将一个seires中 大于，小于制定分为数的数值替换为该分为数对应的数值。
"""

def cap(x,quantile=[0.01,0.99]):
    """盖帽法处理异常值
    Args：
        x：pd.Series列，连续变量
        quantile：指定盖帽法的上下分位数范围
    """

# 生成分位数
    Q01,Q99=x.quantile(quantile).values.tolist()

# 替换异常值为指定的分位数
    if Q01 > x.min():
        x = x.copy()
        x.loc[x<Q01] = Q01

    if Q99 < x.max():
        x = x.copy()
        x.loc[x>Q99] = Q99

    return(x)

sample_test = pd.DataFrame({'normal':np.random.randn(1000)})
sample_test.hist(bins=50)

sample_test['normal'] = cap(sample_test['normal'], quantile=[0.01,0.09])

"""
分箱法
*//分箱法通过考察数据的“近邻”来光滑有序数据的值。有序值分布到一些桶或箱中。
分箱法将异常数据包含在了箱子中，在进行建模的时候，不直接进行到模型中，因而可以达到处理异常值的目的。
"""
sample = pd.DataFrame({'normal':np.random.randn(10)})

# 等分n箱
pd.cut(sample.normal, 5)
# 为分箱贴上标签
pd.cut(sample.normal, bins=5, labels=['bad',2,3,4,'good'])

"""
等深分箱
*//等深分箱中，各个箱的宽度可能不一，但频数是几乎相等的，所以可以采用数据的分位数来进行分箱。
依旧以之前的sample数据为例，现进行等深度分2箱，首先找到2箱的分位数.
"""
sample.normal.quantile([0,0.5,1])
pd.cut(sample.normal, bins=sample.normal.quantile([0,0.5,1]), include_lowest=True, labels=['bad','good'])


"""
多变量异常值处理-聚类法
*//通过快速聚类法将数据对象分组成为多个簇，在同一个簇中的对象具有较高的相似度，而不同的簇之间的对象差别较大。
聚类分析可以挖掘孤立点以发现噪声数据，因为噪声本身就是孤立点
"""

"""
a. 欧式距离：n维空间距离
 dist(x,y) = math.pow(sum(math.pow(xi-yi, 2)), 0.5)

b. 曼哈顿距离:
d(i,j) = abs(xi-xj)+abs(yi-yj)

c. 协方差covariance：
协方差用于衡量两个变量的总体误差的期望。而方差是协方差的一种特殊情况，即当两个变量是相同的情况。
eg: x大于自身期望时候，y小于自身期望，cov为负，反之为正。 如果cov为0，则x/y相互独立。

d. 轮廓系数: https://www.jianshu.com/p/6352d9d468f8
计算样本i到同簇其他样本的平均距离
si接近1，则说明样本i聚类合理。
si接近-1，则说明样本i更应该分类到另外的簇。
若si近似为0，则说明样本i在两个簇的边界上。
"""

# 导入第三方包
import numpy as np
import matplotlib.pyplot as plt
# 随机生成两组二元正态分布随机数
np.random.seed(1234)    #这个random.seed()是一个随机数的种子，随即调用前seed 固定你的那些随机数。                              
mean1 = [0.5, 0.5]
cov1 = [[0.3, 0], [0, 0.1]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 5000).T
mean2 = [0, 8]
cov2 = [[0.8, 0], [0, 2]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 5000).T
# 绘制两组数据的散点图
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(x1, y1)
plt.scatter(x2, y2)
# 显示图形
plt.show()

"""
K-means 聚类
a. 生成哑变量 pd.get_dummies(df[], prefix='a')
b. elbow method手肘法确定最佳k值（利用每个簇中p样本点与ci质点的SSE找，随着K增加，SSE增速变小）
"""
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
%matplotlib inline
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.model_selection import LearningCurve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")
 
df = pd.read_excel('raw_data_k-means.xlsx')
df['a'] = df['a'].astype(object)
dummies = pd.get_dummies(df['a'], prefix='a')
bcd = df.iloc[:, 2:5]
 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(bcd)
X_scaled = pd.DataFrame(x_scaled,columns=bcd.columns)
X_scaled = pd.concat([X_scaled,dummies], axis=1,)
 
# Elbow method 手肘法 1
plt.figure(figsize=(12,9))
 
model = KMeans()
 
visualizer = KElbowVisualizer(model, k=(1,5))
visualizer.fit(X_scaled)       
visualizer.show()

# Elbow method 手肘法 2
SSE = []  # 存放每次结果的误差平方和
for k in range(1,5):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(X_scaled)
    SSE.append(estimator.inertia_) # estimator.inertia_获取聚类准则的总和
X = range(1,5)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X,SSE,'o-')
plt.show()
 
model=MiniBatchKMeans(n_clusters=2)
model.fit(X_scaled)
print("Predicted labels ----")
model.predict(X_scaled)
df['cluster'] = model.predict(X_scaled)
 
plt.figure(figsize=(12,9))
 
model=MiniBatchKMeans(n_clusters=2).fit(X_scaled)
 
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
visualizer.fit(X_scaled)      
visualizer.show()
 
plt.figure(figsize=(12,9))
 
visualizer = InterclusterDistance(model, min_size=10000)
visualizer.fit(X_scaled)
visualizer.show()
 
df = pd.concat([df,X_scaled], axis=1)

"""
k-prototype 聚类算法
"""

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
%matplotlib inline
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.model_selection import LearningCurve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
%matplotlib inline
import matplotlib.pyplot as pltfrom IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
%matplotlib inline
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.model_selection import LearningCurve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
 
df = pd.read_excel('raw_data_k-means.xlsx')
df['a'] = df['a'].astype(object)
 
X = df.iloc[:, 1:5]
X.columns = ['a','b','c','d']
X.head()
 
min_max_scaler = preprocessing.MinMaxScaler() 
bcd = X.iloc[:,1:4]
x_scaled = min_max_scaler.fit_transform(bcd)
X_scaled = pd.DataFrame(x_scaled,columns=bcd.columns)
X = pd.concat([df['a'],X_scaled], axis=1)
 
X_matrix = X.values
cost = []
for num_clusters in list(range(1,5)):
    kproto = KPrototypes(n_clusters=num_clusters, init='Cao')
    kproto.fit_predict(X_matrix, categorical=[0])
    cost.append(kproto.cost_)
    
plt.plot(cost)
pd.DataFrame(cost)
 
kproto = KPrototypes(n_clusters=1, init='Cao')
clusters = kproto.fit_predict(X_matrix, categorical=[0])
print('====== Centriods ======')
kproto.cluster_centroids_
print()
print('====== Cost ======')
kproto.cost_
 
centroids = pd.concat([pd.DataFrame(kproto.cluster_centroids_[1]),pd.DataFrame(kproto.cluster_centroids_[0])], axis=1)
centroids
df['cluster'] = clusters
df.head()

