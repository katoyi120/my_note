import pandas as pd
import matplotlib.pyplot as plt
import joblib
from collections import Counter

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier


def show_image(index):
    """
    显示图片
    :param index: 索引
    :return:
    """
    df = pd.read_csv('./data/mnist_784.csv')
    print(df.shape)
    if index < 0 or index > len(df):
        print("索引超出范围")
        return
    x = df.iloc[:, :-1]  # 获取特征数据,所有行,去掉最后一列
    y = df.iloc[:, -1]  # 获取标签数据,所有行,最后一列
    print('特征数据: ', x.shape)
    print('该索引对应的图片是:', y.iloc[index])
    print('所有标签的个数: ', Counter(y))
    # 将图片数据还原成二维图片
    image = x.iloc[index].values.reshape(28, 28)
    plt.imshow(image, cmap='gray')  # 灰度图
    plt.axis('off')  # 不显示坐标轴
    plt.show()


def train_model():
    """
    训练模型,并将模型保存
    :return:
    """
    # 1.加载数据
    df = pd.read_csv('./data/mnist_784.csv')
    x = df.iloc[:, :-1]  # 特征
    y = df.iloc[:, -1]  # 标签

    # 2.预处理数据,归一化
    x = x / 255  # 归一化公式：x_new = (x - x_min) / (x_max - x_min)

    # 3.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # 参考y轴进行抽取,保持测试集和训练集的标签比例一致(数据均衡)
    )

    # 4.训练模型
    # 创建模型对象
    estimator = KNeighborsClassifier(n_neighbors=3)
    # 训练模型
    estimator.fit(x_train, y_train)

    # 5.模型评估
    print('准确率: ', estimator.score(x_test, y_test))
    print('准确率: ', accuracy_score(y_test, estimator.predict(x_test)))

    # 6.保存模型
    joblib.dump(estimator, './model/knn.pkl')  # pickle文件
    print('模型保存成功')


def use_model():
    """
    测试模型
    :return:
    """
    # 1.加载图片
    image = plt.imread('./data/9.bmp')
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.show()
    # 2.加载模型
    estimator = joblib.load('./model/knn.pkl')
    image = 255 - image  # 像素反转,训练模型的数据是黑底白字,但是测试图片是白底黑字
    image = image.reshape(1, -1)  # 改变图片的维度 效果和reshape(1,784)相同
    image = image / 255  # 归一化,因为训练的时候使用了归一化,测试数据也要归一化
    print('预测结果: ', estimator.predict(image))


if __name__ == '__main__':
    # show_image(6)
    # train_model()
    use_model()
