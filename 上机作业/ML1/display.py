# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib.pylab import plt
from mpl_toolkits.mplot3d import Axes3D


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

    all_data = np.vstack((test_data, train_data, validation_data))
    all_label = np.concatenate((test_label, train_label, validation_label))

    c = ['r', 'b', 'g']

    plt.figure(1)
    plt.subplot(231)
    plt.title('SL+SW')
    for i in range(len(all_data)):
        index = int(all_label[i])-1
        plt.scatter(all_data[i, 0], all_data[i, 1], c=c[index], s=10)

    plt.subplot(232)
    plt.title('SL+PL')
    for i in range(len(all_data)):
        index = int(all_label[i]) - 1
        plt.scatter(all_data[i, 0], all_data[i, 2], c=c[index], s=10)

    plt.subplot(233)
    plt.title('SL+PW')
    for i in range(len(all_data)):
        index = int(all_label[i]) - 1
        plt.scatter(all_data[i, 0], all_data[i, 3], c=c[index], s=10)

    plt.subplot(234)
    plt.title('SW+PL')
    for i in range(len(all_data)):
        index = int(all_label[i]) - 1
        plt.scatter(all_data[i, 1], all_data[i, 2], c=c[index], s=10)

    plt.subplot(235)
    plt.title('SW+PW')
    for i in range(len(all_data)):
        index = int(all_label[i]) - 1
        plt.scatter(all_data[i, 1], all_data[i, 3], c=c[index], s=10)

    plt.subplot(236)
    plt.title('PL+PW')
    for i in range(len(all_data)):
        index = int(all_label[i]) - 1
        plt.scatter(all_data[i, 2], all_data[i, 3], c=c[index], s=10)
    plt.show()

    fig = plt.figure(2)
    ax = fig.add_subplot(221, projection='3d')
    plt.title('SL+SW+PL')
    for i in range(len(all_data)):
        index = int(all_label[i]) - 1
        ax.scatter3D(all_data[i, 0], all_data[i, 1], all_data[i, 2], c=c[index], s=10)

    ax = fig.add_subplot(222, projection='3d')
    plt.title('SL+SW+PW')
    for i in range(len(all_data)):
        index = int(all_label[i]) - 1
        ax.scatter3D(all_data[i, 0], all_data[i, 1], all_data[i, 3], c=c[index], s=10)

    ax = fig.add_subplot(223, projection='3d')
    plt.title('SL+PL+PW')
    for i in range(len(all_data)):
        index = int(all_label[i]) - 1
        ax.scatter3D(all_data[i, 0], all_data[i, 2], all_data[i, 3], c=c[index], s=10)

    ax = fig.add_subplot(224, projection='3d')
    plt.title('SW+PL+PW')
    for i in range(len(all_data)):
        index = int(all_label[i]) - 1
        ax.scatter3D(all_data[i, 1], all_data[i, 2], all_data[i, 3], c=c[index], s=10)
    plt.show()