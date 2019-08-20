#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from xzc_tools import tools

@tools.funcRunTime
def load_data(file_path):
    try:
        feature = []
        label = []
        with open(file_path) as f:
            for line in f:
                feature_temp = []
                line = [float(i) for i in line.strip().split()]
                feature_temp.append(line[0:3])
                label.append(int(line[-1]))
                feature.extend(feature_temp)
        return np.mat(feature), label
    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

@tools.funcRunTime
def normalization_data(feature):
    try:
        # 计算特征的均值和标准差
        feature_u = feature.mean(axis=0)
        feature_sigma = np.std(feature, axis=0)

        # 计算特征矩阵的大小
        feature_size = np.shape(feature)

        feature_u_temp = feature_u.repeat(feature_size[0], axis=0)
        feature_sigma_temp = feature_sigma.repeat(feature_size[0], axis=0)

        # 归一化
        feature = (feature - feature_u_temp) / feature_sigma_temp
        return feature, feature_u, feature_sigma

    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

def knn(unknown_data, feature, feature_u, feature_sigma, label):
    try:
        # 未知数据归一化
        unknown_data = (unknown_data - feature_u) / feature_sigma

        feature_size = np.shape(feature)
        # 未知数据与已知数据的距离
        unknown_data = unknown_data.repeat(feature_size[0], axis=0)
        unknown_data = unknown_data - feature
        unknown_data = np.multiply(unknown_data, unknown_data)
        unknown_data = np.sum(unknown_data, axis=1)
        unknown_data = np.sqrt(unknown_data)

        # 未知数据距离排序取前五的类别
        unknown_data = [i[0] for i in unknown_data.tolist()]
        k_label = [i[0] for i in sorted(enumerate(unknown_data), key=lambda x:x[1])]
        k_label = k_label[0:5]
        k_label = [label_data[i] for i in k_label]
        return k_label

    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

if __name__ == '__main__':

    data_file_path = os.path.abspath(sys.argv[1])
    feature_data, label_data = load_data(data_file_path)

    feature_data, feature_u, feature_sigma = normalization_data(feature_data)

    unknown_data = input('请输入需要进行分类的数据(以空格分隔)：\n')
    unknown_data = np.mat([float(i) for i in unknown_data.strip().split()])
    knn(unknown_data,feature_data, feature_u, feature_sigma, label_data)

