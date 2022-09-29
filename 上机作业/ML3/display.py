# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib.pylab import plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
from scipy import stats


def readData(path): # 读取并处理数据
    dataSet = pd.DataFrame(pd.read_csv(path))
    dataSet.drop('No', axis=1, inplace=True)
    dataSet.dropna(axis=0, how='any', inplace=True)
    dataSet_value = np.array(dataSet.iloc[:, 4])
    dataSet.drop('pm2.5', axis=1, inplace=True)  # pm2.5作为标签移出数据集
    dataSet_data = dataSet

    return dataSet_data, dataSet_value


if __name__ == '__main__':
    # —————————————————————————————————————————
    # |测试 8765| 训练 26294| 验证 8765| 特征 11| 预测范围 0-972|
    path1 = 'PRSA_test.data.csv'  # 测试集路径
    path2 = 'PRSA_train.data.csv'  # 训练集路径
    path3 = 'PRSA_validation.data.csv'  # 验证集路径

    # —————————————————————————————————————————
    # 数据集特征观察
    # No	year	month	day	hour	pm2.5	DEWP	TEMP	PRES	cbwd	Iws	Is	Ir
    train_data = pd.DataFrame(pd.read_csv(path2))
    train_data.drop('No', axis=1, inplace=True)
    train_data.dropna(axis=0, how='any', inplace=True)
    encoder = LabelEncoder()
    train_data.iloc[:, 8] = encoder.fit_transform(train_data.iloc[:, 8])  # 将cbwd整数编码

    # 画相关性热力图，图例最小值 -1，最大值1，颜色对象设为红蓝('RdBu'),颜色数目为128
    sns.heatmap(train_data.corr(), vmin=-1, vmax=1, cmap=sns.color_palette('RdBu', n_colors=128))
    plt.show()

    sns.set()
    sns.displot(train_data['pm2.5'], kde=True)
    plt.show()

    stats.probplot(train_data['pm2.5'], plot=plt)
    plt.show()

    train_data.drop(train_data[train_data["pm2.5"] == 0].index, inplace=True)
    sns.displot(np.log(train_data["pm2.5"]), kde=True)
    plt.show()

    stats.probplot(np.log(train_data['pm2.5']), plot=plt)
    train_data["log_pm2.5"] = np.log(train_data['pm2.5'])
    plt.show()