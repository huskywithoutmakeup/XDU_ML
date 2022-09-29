# -*- coding:utf-8 -*-
import numpy as np
import math
import pandas as pd
from matplotlib.pylab import plt
from imblearn.over_sampling import SMOTE                # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSample
from imblearn.ensemble import EasyEnsembleClassifier    # 简单集成方法EasyEnsemble
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn import manifold, preprocessing, decomposition, metrics, random_projection
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
import joblib


def NMI(A,B):
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

def purity_score(y_true, y_pred):
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


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

    all_data = preprocessing.minmax_scale(all_data, feature_range=(0, 1))


    # —————————————————————————————————————————
    # 数据预处理: 数据降维

    # 法1 T-SNE降维
    # tsne = manifold.TSNE(n_components=3, init="pca")
    # all_data = pca.fit_transform(all_data)

    # 法2 PCA降维
    pca = decomposition.TruncatedSVD(n_components=2)
    all_data = pca.fit_transform(all_data)

    # 法3 Isomap降维
    # isomap = manifold.Isomap(n_neighbors=20, n_components=3)
    # all_data = pca.fit_transform(all_data)

    test_data = all_data[0:len(test_data)]
    train_data = all_data[(len(test_data)):(len(test_data)+len(train_data))]
    validation_data = all_data[(len(test_data)+len(train_data)):(len(test_data)+len(train_data)+len(validation_data))]

    # —————————————————————————————————————————
    # 数据预处理: 处理训练集样本不均匀的问题

    # 法1 SMOTE方法进行过采样处理
    model_smote = SMOTE()  # 建立SMOTE模型对象
    train_data_resampled1, train_label_resampled1 = model_smote.fit_resample(train_data, train_label)  # 输入数据做过抽样处理
    print('Size after SMOTE : ', np.shape(train_data_resampled1))
    print('% :', sum(train_label_resampled1)/len(train_label_resampled1))
    print()

    # 法2 RandomUnderSampler方法进行欠抽样处理
    model_RandomUnderSample = RandomUnderSampler()  # 建立RandomUnderSampler模型对象
    train_data_resampled2, train_label_resampled2 = model_RandomUnderSample.fit_resample(train_data, train_label)  # 输入数据做欠抽样处理
    print('Size after RandomUnderSampler : ', np.shape(train_data_resampled2))
    print('% :', sum(train_label_resampled2) / len(train_label_resampled2))
    print()

    # 法3 使用集成方法EasyEnsemble处理不均衡样本
    model_EasyEnsemble = EasyEnsembleClassifier()  # 建立EasyEnsemble模型对象
    train_data_resampled3, train_label_resampled3 = model_RandomUnderSample.fit_resample(train_data, train_label)  # 输入数据并应用集成方法处理
    print('Size after EasyEnsemble : ', np.shape(train_data_resampled3))
    print('% :', sum(train_label_resampled3) / len(train_label_resampled3))
    print()

    # —————————————————————————————————————————
    # 模型训练
    # Kmeans聚类
    model_kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
    model_kmeans.fit(train_data_resampled2)

    # GMM聚类
    model_gmm = GaussianMixture(n_components=2)
    model_gmm.fit_predict(train_data_resampled2)

    # DBSCAN聚类
    model_dbscan = DBSCAN(eps=0.1, min_samples=10)
    model_dbscan.fit_predict(train_data_resampled2)

    # Agg聚类
    model_agg = AgglomerativeClustering(n_clusters=2, linkage="ward")
    model_agg.fit_predict(train_data_resampled2)

    # —————————————————————————————————————————
    # 模型评价 指标:NMI
    test_pred_kmeans = model_kmeans.predict(test_data)
    model_score1 = metrics.normalized_mutual_info_score(test_label, test_pred_kmeans)
    print('Kmeans NMI 1 = ', model_score1)

    test_pred_gmm = model_gmm.predict(test_data)
    model_score2 = metrics.normalized_mutual_info_score(test_label, test_pred_gmm)
    print('Gmm NMI 2 = ', model_score2)

    test_pred_dbscan = model_dbscan.fit_predict(test_data)
    model_score3 = metrics.normalized_mutual_info_score(test_label, test_pred_gmm)
    print('DBSCAN NMI 3 = ', model_score3)

    # 另外的指标评价 Purity, ARI
    print()
    model_score4 = purity_score(test_label, test_pred_kmeans)
    print('Kmeans Purity = ', model_score4)

    model_score5 = metrics.adjusted_rand_score(test_label, test_pred_kmeans)
    print('Kmeans ARI = ', model_score2)

    # 导出模型
    joblib.dump(model_kmeans, 'model_2.model')