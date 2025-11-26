# -*- coding: utf-8 -*-

"""
@author: ZZX
@project: table_Deep
@file: create_data_list.py
@function:
@time: 2021/2/2 上午11:23
"""
import os
import numpy as np

def create_data():
    print('正在创建图像列表,请勿停止程序....')
    data = np.load('../data/data.npy')
    data = data.astype('float32')
    class_sum = 0
    img_paths = os.listdir('../data/hsv_img/')
    for img_path in img_paths:
        name_path = '../data/hsv_img/' + img_path
        index = int(img_path.split('.')[0])
        if class_sum % 10 == 0:
            with open('../data/test.list', 'a') as f:
                f.write(name_path + "\t%d" % data[index] + "\n")
        else:
            with open('../data/train.list', 'a') as f:
                f.write(name_path + "\t%d" % data[index] + "\n")
        class_sum += 1
    print('图像列表创建完成')
create_data()
