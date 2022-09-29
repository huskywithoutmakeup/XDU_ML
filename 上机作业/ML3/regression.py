# -*- coding:utf-8 -*-
import numpy as np
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from mlxtend import regressor
import joblib


def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


def add_timestepdim(data, n_in=1, n_out=1):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    agg.dropna(inplace=True)
    return agg


def data_deal(path1, path2, path3):
    dataset1 = read_csv(path1, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset2 = read_csv(path2, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset3 = read_csv(path3, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset = dataset1.append(dataset2)
    dataset = dataset.append(dataset3)
    test_id_list = np.array(dataset1.iloc[:, 0])
    train_id_list = np.array(dataset2.iloc[:, 0])
    validation_id_list = np.array(dataset3.iloc[:, 0])
    dataset = dataset.sort_values(by=['No'], ascending=True)
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column names
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['pm2.5'].fillna(0, inplace=True)
    # drop the first 24 hours
    dataset = dataset[24:]

    return dataset, test_id_list, train_id_list, validation_id_list


def dataset_part(id_list):
    id_list = id_list - 25
    i = 0
    temp = len(id_list)
    while i < temp:
        if id_list[i] < 0:
            index = [i]
            id_list = np.delete(id_list, index, 0)
            temp = temp - 1
            i = i - 1
        i = i + 1

    return id_list

def delete_na(data):
    i = 0
    temp = len(data)
    while i < temp:
        if data[i, 0] == 0 or data[i, -1] == 0:
            index = [i]
            data = np.delete(data, index, 0)
            temp = temp - 1
            i = i - 1
        i = i + 1

    return data


if __name__ == '__main__':
    # —————————————————————————————————————————
    # 数据读取
    # |测试 8765| 训练 26294| 验证 8765| 特征 11| 预测范围 0-972|
    path1 = 'PRSA_test.data.csv'  # 测试集路径
    path2 = 'PRSA_train.data.csv'  # 训练集路径
    path3 = 'PRSA_validation.data.csv'  # 验证集路径

    dataSet, test_id_list, train_id_list, validation_id_list = data_deal(path1, path2, path3)

    values = dataSet.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = add_timestepdim(values, 1, 1)

    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    all_data = reframed.values

    # 利用获得的id list转换测试，训练，验证集
    test_id_list = dataset_part(test_id_list)
    train_id_list = dataset_part(train_id_list)
    train_id_list = np.delete(train_id_list,np.argmax(train_id_list))
    validation_id_list = dataset_part(validation_id_list)

    test = all_data[test_id_list, :]
    train = all_data[train_id_list, :]
    validation = all_data[validation_id_list, :]

    # 去除na值
    test = delete_na(test)
    train = delete_na(train)
    validation = delete_na(validation)

    train_data, train_value = train[:, :-1], train[:, -1]
    test_data, test_value = test[:, :-1], test[:, -1]
    validation_data, validation_value = validation[:, :-1], validation[:, -1]

    # —————————————————————————————————————————
    # 模型训练
    # 使用stacking方法结合多个回归模型
    lr = LinearRegression()  # 线性回归
    dtr = DecisionTreeRegressor()  # 决策树回归
    svr_rbf = SVR(kernel='rbf', gamma='auto')  # 支持向量回归
    knr = KNeighborsRegressor()  # k近邻回归
    ridge = Ridge()  # 岭回归(L2正则化)
    lasso = Lasso()  # Lasso回归(L1正则化)
    regression_models = [lr, dtr, svr_rbf, knr, ridge, lasso]
    sclf = regressor.StackingCVRegressor(regression_models, meta_regressor=ridge)
    sclf.fit(train_data, train_value)
    # 测试数据回归预测
    test_value_sclf = sclf.predict(test_data)
    print(test_value)
    print(test_value_sclf)

    # —————————————————————————————————————————
    # 模型评估 指标: MSE
    # 另外的指标评价 MAE, RMSE, 决定系数R^2
    print('MSE = ', metrics.mean_squared_error(test_value, test_value_sclf))
    print('MAE = ', metrics.mean_absolute_error(test_value, test_value_sclf))
    print('R2 = ', metrics.r2_score(test_value, test_value_sclf))

    # 导出模型
    joblib.dump(sclf, 'model_3.model')



