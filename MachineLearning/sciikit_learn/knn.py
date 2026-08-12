"""
knn(k nearest neighbors):k 近邻算法
欧氏距离: 对应维度差值平方和,再开平方根  (勾股定理)
曼哈顿距离(城市街区距离): 对应维度差值的绝对值,再求和 (在一个棋盘中,只能横着或者竖着移动,横平竖直)
切比雪夫距离: 对应维度差值的绝对值,再求最大值 (在一个棋盘中,能横,竖,斜移动)
原理:
    基于 欧氏距离 或其他计算方式,计算测试集 和 每个训练样本之间的距离,然后根据距离升序排序,找到最近的k个样本
    分类问题: 基于k个样本投票,票数多的为最终结果
    回归问题: 基于k个样本计算平均值,作为预测结果
思路:
    分类问题: 适用于 有特征, 有标签, 且标签不是连续的
    回归问题: 有特征, 有标签, 标签是连续的
注意:
    k值过小: 会导致学习到"脏的特征",容易受到异常值的影响,导致过拟合
    k值过大: 模型会变得简单,导致欠拟合

"""

from sklearn.neighbors import KNeighborsClassifier  # knn分类包
from sklearn.neighbors import KNeighborsRegressor  # knn回归包


def knn_classify_demo():
    """
    knn分类
    根据欧氏距离:对应维度差值的平方和,再开根,从小到大,去最近的k个,再投票
    当测试集是5,k=3时,各个距离:
    x_train      y_train        距离
    0               0           math.sqrt((5-0)**2) = 5
    1               0           math.sqrt((5-1)**2) = 4
    2               1           math.sqrt((5-2)**2) = 3
    3               1           math.sqrt((5-3)**2) = 2
    5               ?
    根据欧氏距离,当k=3时,距离最近的3个邻居距离是 2 3 4 ,他们的对应的标签分别是 1 1 0
    所以 测试集5 应当归为分类1
    :return:
    """
    # 1.准备数据
    x_train = [[0], [1], [2], [3]]  # 训练(特性)集,因为特征可以有多个,所以是一个多维数组
    y_train = [0, 0, 1, 1]  # 标签集,标签集是离散的,所以标签集是一个一维数组
    x_test = [[5]]  # 测试集
    # 2.创建(knn分类)模型对象
    model = KNeighborsClassifier(n_neighbors=3)  # 创建模型对象,n_neighbors为k值,建议选取奇数
    # 3.训练模型
    model.fit(x_train, y_train)
    # 4.预测
    y_predict = model.predict(x_test)
    print(y_predict)


def knn_regression_demo():
    """
    knn回归
    根据欧氏距离:对应维度差值的平方,再开根,从小到大,去最近的k个,求平均值
    当k=3时,去最近的3组数据,分别是[3,10,10] [4,11,12] [1,1,0] 他们对应的标签是 0.3 0.4 0.2
    所以预测值是: (0.3+0.4+0.2)/3 = 0.3
    :return:
    """
    # 1.准备数据
    # 开根号:      14.53       14.28       1           2.24
    # 平方和:      211         204         1           5
    # 差值:     (3,11,9)   (2,10,10)  (0, 1, 0)    (1, 0,  2)
    x_train = [[0, 0, 1], [1, 1, 0], [3, 10, 10], [4, 11, 12]]  # 训练(特性)集,因为特征可以有多个,所以是一个多维数组
    x_test = [[3, 11, 10]]  # 测试集
    y_train = [0.1, 0.2, 0.3, 0.4]  # 标签集,标签集是离散的,所以标签集是一个一维数组
    # 2.创建(knn分类)模型对象
    model = KNeighborsRegressor(n_neighbors=3)  # 创建模型对象,n_neighbors为k值,建议选取奇数
    # 3.训练模型
    model.fit(x_train, y_train)
    # 4.预测
    y_predict = model.predict(x_test)
    print(y_predict)


if __name__ == '__main__':
    knn_classify_demo()
    knn_regression_demo()
