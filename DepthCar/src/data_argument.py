import os
import cv2
import numpy as np
from paddle.io import Dataset

class MyDataset(Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """
    def __init__(self, data_dir, label_path, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(MyDataset, self).__init__()
        self.data_list = []
        with open(label_path,encoding='utf-8') as f:
            for line in f.readlines():
                image_path, label = line.strip().split('\t')
                # image_path = os.path.join(data_dir, image_path)
                self.data_list.append([image_path, label])
        # 2. 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        image_path, label = self.data_list[index]
        print(image_path, ' ', label)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # image = image.astype('float32')
        cv2.imshow('image', image)
        cv2.waitKey(1)
        # 3. 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        label = int(label)
        return image, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)
# 1. 定义随机旋转和改变图片大小的数据处理方法
from paddle.vision.transforms import Compose, ColorJitter, RandomCrop
transform = Compose([RandomCrop(320), ColorJitter(0.4, 0.4, 0.4, 0.4)])
img, lbl = MyDataset('../data/hsv_img','../data/train.list', transform).__getitem__(5)
print(lbl)
cv2.imshow('img', img)
cv2.waitKey()
