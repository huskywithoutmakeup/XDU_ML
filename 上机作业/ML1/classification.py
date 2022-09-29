# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold  # 交叉验证
from sklearn.model_selection import GridSearchCV  # 网格搜索
import joblib


def readData(path):
    with open(path, 'r')as f:  # 读取data数据
        data = f.readlines()
    label = np.zeros(len(data))
    newData = np.zeros([len(data), 4])
    for i in range(len(data)):  # 拆分替换字符串，转为浮点数类型数组
        data[i] = data[i].split(',')
        newData[i][0] = float(data[i][0])
        newData[i][1] = float(data[i][1])
        newData[i][2] = float(data[i][2])
        newData[i][3] = float(data[i][3])
        label[i] = tranStr(data[i][4])
    return newData, label


def tranStr(name):
    if name == "Iris-versicolor\n":
        return 1
    elif name == "Iris-setosa\n":
        return 2
    elif name == "Iris-virginica\n":
        return 3


if __name__ == '__main__':
    # —————————————————————————————————————————
    # 数据读取
    # |测试 30| 训练 90| 验证 30| ，| 类1 32.2%| 类2 32.2%| 类3 35.6%|
    # 因为类别分布较为均匀，因此无需特殊的数据处理
    path1 = 'iris_test.data'  # 测试集路径
    path2 = 'iris_train.data'  # 训练集路径
    path3 = 'iris_validation.data'  # 验证集路径
    test_data, test_label = readData(path1)
    train_data, train_label = readData(path2)
    validation_data, validation_label = readData(path3)
    # —————————————————————————————————————————
    # 模型训练

    # SVC 用于当样例数少于10000时的二元和多元分类
    # C 误差项的惩罚参数，C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，
    # 这样会出现训练集测试时准确率很高，但泛化能力弱; C值小，对误分类的惩罚减小，容错能力增强，泛化能力较强。
    # kernel svc中指定核函数的类型, 可以是： ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 或者自己指定。 默认使用‘rbf’
    # kernel='rbf' 时，为高斯核函数，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。

    # 最优超参数选择(网络搜索)
    kernel = ['linear', 'rbf', 'poly', 'sigmoid']
    C = np.arange(0.01, 1.01, 0.01)

    max_val_score = 0  # 选取验证集上具有最高准确率的超参数
    p_1, p_2 = 0, 0

    for i in range(len(kernel)):
        for j in range(100):
            model = SVC(kernel=kernel[i], C=C[j])
            model.fit(train_data, train_label)
            val_pred = model.predict(validation_data)
            val_score = metrics.accuracy_score(validation_label, val_pred)
            if val_score>max_val_score:
                max_val_score = val_score
                p_1, p_2 = i, j
    print()
    print('网格搜索 for循环法')
    print('kernel = ', kernel[p_1], '  C = ', C[p_2])
    print('Validation Set Accuary = ', max_val_score)

    print()
    print('网格搜索 GridSearchCV') # 通过交叉验证确定最佳效果参数
    # 超参数
    parameters = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'C': np.linspace(0.1, 1, 10), 'gamma': np.linspace(0.25, 1.25, 10)}
    svc = SVC()
    # 结合验证集与数据集，利用交叉验证选择参数
    k_data = np.vstack((validation_data, train_data))
    k_label = np.concatenate((validation_label, train_label))
    # 模型训练
    grid = GridSearchCV(svc, parameters, cv=10, scoring='accuracy')
    grid.fit(k_data, k_label)
    print(grid.best_params_)
    model1 = grid.best_estimator_
    k_pred1 = model1.predict(k_data)
    print('k Set Accuary = ', metrics.accuracy_score(k_label, k_pred1))
    print('StratifiedKFold score = ', grid.best_score_)  # 平均交叉验证分数

    # —————————————————————————————————————————
    # 模型评价 指标:Accuary
    model1.fit(train_data, train_label)
    test_pred = model1.predict(test_data)
    test_score = metrics.accuracy_score(test_label, test_pred)
    print()
    print('Test Set Accuary 1 = ', test_score)

    model2 = SVC(kernel=kernel[p_1], C=C[p_2])
    model2.fit(train_data, train_label)
    test_pred = model2.predict(test_data)
    test_score = metrics.accuracy_score(test_label, test_pred)
    print()
    print('Test Set Accuary 2 = ', test_score)
    print()
    # 另外的指标评价 Precision, Recall
    test_precision1 = metrics.precision_score(test_label, test_pred, average='macro')
    test_recall1 = metrics.recall_score(test_label, test_pred, average='macro')
    print('Test Set precision 1 = ', test_precision1)
    print()
    print('Test Set recall 1 = ', test_recall1)

    # 导出模型
    joblib.dump(model1, 'model_1.model')