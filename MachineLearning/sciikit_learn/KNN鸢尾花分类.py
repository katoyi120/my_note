import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # 分割测试集和训练集
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def get_iris_data(start=0, end=150):
    """
    获取数据
    :param start: 开始索引
    :param end: 结束索引
    :return:
    """
    iris_datas = load_iris()
    # print(iris_datas.keys())
    # print(iris_datas.data)  # 特征数据
    # print(iris_datas.data.shape)
    # print(iris_datas.feature_names)  # 特征名称
    # print(iris_datas.target)  # 标签数据 0,1,2,
    # print(iris_datas.target_names)  # 标签名称
    return {
        'data': iris_datas.data[start:end],
        'target': iris_datas.target[start:end],
        'feature_names': iris_datas.feature_names,
        'target_names': iris_datas.target_names
    }


def show_iris():
    """
    可视化
    :return:
    """
    iris = get_iris_data()
    iris_df = pd.DataFrame(iris.get('data'), columns=iris.get('feature_names'))
    # 新增标签列
    iris_df['label'] = iris.get('target')
    # 绘制散点图
    sns.lmplot(
        x='sepal length (cm)',
        y='sepal width (cm)',
        hue='label',
        data=iris_df,
        fit_reg=True  # fit_reg=False 表示没有辅助线
    )
    plt.title('iris data')
    plt.tight_layout()
    plt.show()


def split_data(test_size=0.2, random_state=1):
    """
    切割数据集和测试集
    从总的数据中,按照一点的比例,切分训练集和测试集
    训练集用于训练模型,测试集用来测试模型的准确性
    :param test_size: 切割比例
    :param random_state: 随机种子,确保随机生成的数据集都是固定的
    :return:
    """
    iris = get_iris_data()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.get('data'),
        iris.get('target'),
        test_size=test_size,
        random_state=random_state
    )
    print("训练集的个数", x_train.shape)
    return x_train, x_test, y_train, y_test


def train_model():
    """

    :return:
    """
    x_train, x_test, y_train, y_test = split_data()

    # 对特征进行预处理,此处使用标准化
    transfer = StandardScaler()

    # fit_transfer: 先训练,再转换,兼具fit和transfer的功能,第一次进行标准化时使用,一般用于处理训练集
    x_train = transfer.fit_transform(x_train)

    # transform: 只有转换,适用于需要重复进行标准化动作时使用,一般用于处理测试集
    x_test = transfer.transform(x_test)

    # 训练模型
    estimator = KNeighborsClassifier(n_neighbors=5)
    estimator.fit(x_train, y_train)

    # 模型预测
    # 预测切割的数据集
    y_predict = estimator.predict(x_test)
    print("预测结果", y_predict)

    # 预测自定义测试集
    my_data = [[7.8, 2.1, 3.9, 1.6]]
    my_transfer = transfer.transform(my_data)
    my_predict = estimator.predict(my_transfer)
    print("自定义测试集预测结果", my_predict)

    # 查看预测的准确率
    my_proba = estimator.predict_proba(my_transfer)
    print("预测的概率", my_proba)

    # 模型评估
    # 方式1:直接评分,基于: 测试集的特征 和 测试集的标签,评估100个样本中,模型预测对了多少个
    print("准确率", estimator.score(x_test, y_test))
    # 方式2:基于: 测试集的标签 和 预测结果
    print("准确率", accuracy_score(y_test, y_predict))


if __name__ == '__main__':
    train_model()
