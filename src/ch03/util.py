#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: util.py
@time: 2020/7/1 13:44
@project: deeplearning-with-pytorch-notes
@desc:
"""

import torch
from matplotlib import pyplot as plt


def plot_curve(data):
    """
    下降曲线的绘制
    :param data:
    :return:
    """
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


def plot_image(img, label, name):
    """
    可视化识别结果
    :param img:
    :param label:
    :param name:
    :return:
    """
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def one_hot(label, depth=10):
    # 第1步：创建一个全0矩阵，形状为 (batch_size, depth)
    # label.size(0) 获取批次大小，depth 是类别总数
    out = torch.zeros(label.size(0), depth)
    
    # 第2步：将标签转换为 LongTensor 并调整形状为 (batch_size, 1)
    # scatter_ 要求 index 的维度与输出张量匹配，这里需要2D索引
    idx = torch.LongTensor(label).view(-1, 1)
    
    # 第3步：核心操作 - 在指定位置填充1
    # dim=1: 沿着列方向（类别维度）进行散射
    # index=idx: 指定每一行中哪个列位置填1
    # value=1: 填充的值
    out.scatter_(dim=1, index=idx, value=1)
    
    return out