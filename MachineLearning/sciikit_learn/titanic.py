import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV



def load_data():
    data = pd.read_csv('./data/titanic.csv')
    # print(data.shape)
    # 查看数据信息
    # data.info()
    df = data[['Survived', 'Pclass', 'Sex', 'Age']].copy()
    if df['Age'].isnull().values.any():
        df['Age'] = df['Age'].fillna(df['Age'].mean())
    if df['Sex'].dtype == 'object':
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    return df


def cart_tree():
    """
    决策树
    :return:
    """
    # 只选择部分特征
    df = load_data()
    x = df[['Pclass', 'Sex', 'Age']]
    y = df['Survived']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    print('决策树预测值:', y_pred)
    print('决策树准确率:', dt.score(x_test, y_test))
    print('-' * 20)
    # print('决策树评估报告\n', classification_report(y_test, y_pred))
    # plt.figure(figsize=(30, 20))
    # plot_tree(dt, feature_names=x.columns, class_names=['0', '1'], filled=True, max_depth=5)
    # plt.savefig('./data/dt.png')
    # plt.show()


def random_forest():
    """
    随机森林
    :return:
    """
    df = load_data()
    x = df[['Pclass', 'Sex', 'Age']]
    y = df['Survived']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    estimator = RandomForestClassifier()
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    print('随机森林预测值:', y_pred)
    print('随机森林准确率:', estimator.score(x_test, y_test))
    print('-' * 20)
    # 采用网格搜索调参
    params = {
        'n_estimators': [30, 50, 70, 90, 110],
        'max_depth': [3, 5, 7, 9, 11],
    }
    estimator2 = RandomForestClassifier()
    gs_estimator = GridSearchCV(estimator2, params, cv=5)
    gs_estimator.fit(x_train, y_train)
    print('网格搜索预测值为:', gs_estimator.predict(x_test))
    print('网格搜索准确率:', gs_estimator.score(x_test, y_test))
    print('网格搜索最佳参数:', gs_estimator.best_params_)
    print('网格搜索最佳分数:', gs_estimator.best_score_)

if __name__ == '__main__':
    cart_tree()
    random_forest()
