from ctypes import *
import numpy as np
import cv2

import paddle.fluid as fluid
from PIL import Image
import paddle

paddle.enable_static()
import sys, os
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (non_max_suppression, scale_coords)
from utils.torch_utils import select_device
from multiprocessing import Process, Queue
import time


# 速度
vel = 1555
# 转向角
angle = 1500
t = 0
num = 0
num_red = 0
counter = 0
counter_lane = 0
slow_flag = 0
q = Queue()
ser = None
last_label = 'start'


def clear_img_txt():
    # 清空图片
    # D:\dlCar2\yolov5_v6_0\yolov5_test.py
    # D:\dlCar2\DepthCar\data\running_img
    os.system('rm ../data/running_img/*')
    # 清空操作记录
    with open('../data/running_data.txt', 'w') as f:
        f.truncate(0)
        f.close()

def sign():
    global vel
    global angle
    global counter
    global last_label
    global slow_flag
    device = select_device('cpu')
    # device = select_device('0')
    # half = device.type != '0'  # half precision only supported on CUDA
    weights = '../model/yolov5_model/bestn.pt'
    # 加载模型
    # model = attempt_load(weights, device=device)  # load FP32 model
    model = DetectMultiBackend(weights, device=device, dnn=False, data='models/my.yaml', fp16=False)
    stride, pt = model.stride, model.pt
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # D:\dlCar3\DepthCar\src\yolov5_test.py
    # D:\dlCar3\yolov5_v6_0\data\images0725
    imgPath = '../../yolov5_v6_0/data/images0723/'
    savPath = '../data/running_img/'
    vidPath = '../data/running_video/'
    imgNames = os.listdir(imgPath)
    imgNames.sort(key=lambda x:int(x[:-4]))
    print(imgNames)
    tempImg = cv2.imread(imgPath + imgNames[0])
    # 设置编码格式、输出路径、fps、size
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    i = 1
    while os.path.isfile(vidPath + f'output_{i}.avi'):
        i += 1
    video = cv2.VideoWriter(vidPath + f'output_{i}.avi', fourcc, 30, (tempImg.shape[1], tempImg.shape[0]))
    for imgName in imgNames:
        startTime = time.time()
        image = cv2.imread(imgPath + imgName)
        with torch.no_grad():
            # Padded resize
            img = letterbox(image, new_shape=160, stride=stride, auto=pt)[0]
            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img, augment=False, visualize=False)
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
            f = pred[0].tolist()

            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    coor = []
                    label = names[int(cls)]
                    conf = round(float(conf), 2)
                    if conf >= 0.80:
                        for i in xyxy:
                            i = i.tolist()
                            i = int(i)
                            # label = '%s %.2f' % (names[int(cls)], conf)
                            # print(conf)
                            coor.append(i)
                        cv2.rectangle(image, (int(coor[0]), int(coor[1])),
                                      (int(coor[2]), int(coor[3])),
                                      (0, 255, 0), 7)
                        label = str(label)
                        # print(last_label, ' ', label)
                        area = (coor[2] - coor[0]) * (coor[3] - coor[1])

                        cv2.putText(image, str(conf), (coor[0], coor[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(image, label, (coor[0], coor[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(image, str(area), (coor[0], coor[3] + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        last_label = label
                    del coor
        endTime = time.time()
        fps = round(1.0 / (endTime - startTime), 2)
        cv2.putText(image, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow('sign', image)
        video.write(image)
        cv2.imwrite(savPath + str(counter) + '.jpg', image)
        counter += 1
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit(0)
            break
    video.release()

if __name__ == '__main__':
    clear_img_txt()
    sign()
