# -*- coding:utf-8 -*-
import numpy as np
import math
import pandas as pd
from matplotlib.pylab import plt
import seaborn as sns
from imblearn.over_sampling import SMOTE                # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSample
from imblearn.ensemble import EasyEnsembleClassifier    # 简单集成方法EasyEnsemble
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn import manifold, preprocessing, decomposition, metrics, random_projection
from sklearn.mixture import GaussianMixture
import joblib
from sklearn.preprocessing import StandardScaler


def readData(path):
    with open(path, 'r')as f:  # 读取data数据
        data = f.readlines()
    label = np.zeros(len(data))
    newData = np.zeros([len(data), 30])
    for i in range(len(data)):  # 拆分替换字符串，转为浮点数类型数组
        data[i] = data[i].split(',')
        label[i] = tranStr(data[i][1])
        for j in range(2, len(data[0])):
            newData[i][j-2] = float(data[i][j])

    return newData, label


def tranStr(name):
    if name == "M":
        return 0
    elif name == "B":
        return 1


if __name__ == '__main__':
    # —————————————————————————————————————————
    # 数据读取
    # |测试 114| 训练 341| 验证 114| 特征 20| 类别 2| ，| 类M 35.2%| 类B 64.8%|
    path1 = 'wdbc_test.data'  # 测试集路径
    path2 = 'wdbc_train.data'  # 训练集路径
    path3 = 'wdbc_validation.data'  # 验证集路径
    test_data, test_label = readData(path1)
    train_data, train_label = readData(path2)
    validation_data, validation_label = readData(path3)

    all_data = np.vstack((test_data, train_data, validation_data))
    all_label = np.concatenate((test_label, train_label, validation_label))


    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)
    data_df = pd.DataFrame(all_data)

    # 异常点显示
    B_num = int(math.fsum(all_label))
    M_num = int(len(all_data) - B_num)

    B_temp = 0
    M_temp = 0
    B_data = all_data[0:B_num]
    M_data = all_data[0:M_num]

    for i in range(len(all_data)):
        if all_label[i] == 1:
            B_data[B_temp] = all_data[i]
            B_temp = B_temp+1
        elif all_label[i] == 0:
            M_data[M_temp] = all_data[i]
            M_temp = M_temp + 1

    plt.figure(1, figsize=(14, 8))
    plt.title('M')
    sns.boxplot(data=M_data)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    plt.figure(2, figsize=(14, 8))
    plt.title('B')
    sns.boxplot(data=B_data)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    data_df_mean = data_df[data_df.columns[0:10]]
    data_df_se = data_df[data_df.columns[10:20]]
    data_df_worst = data_df[data_df.columns[20:30]]


    # 相关性热力图
    plt.figure(3)
    plt.figure(figsize=(24, 8))
    plt.subplot(1, 3, 1)
    sns.heatmap(data_df_mean.corr(), cbar=False, square=True, annot=True, fmt='.2f', annot_kws={'size': 8},
                cmap='coolwarm')
    plt.subplot(1, 3, 2)
    sns.heatmap(data_df_se.corr(), cbar=False, square=True, annot=True, fmt='.2f', annot_kws={'size': 8},
                cmap='coolwarm')
    plt.subplot(1, 3, 3)
    sns.heatmap(data_df_worst.corr(), cbar=True, square=True, annot=True, fmt='.2f', annot_kws={'size': 8},
                cmap='coolwarm')
    plt.show()