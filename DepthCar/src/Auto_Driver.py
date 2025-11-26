from ctypes import *
import numpy as np
import cv2

import paddle.fluid as fluid
from PIL import Image
import paddle

paddle.enable_static()
import sys, os
import torch
import serial
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (non_max_suppression)
from utils.torch_utils import select_device
from multiprocessing import Process, Queue
import time
from hex_change import car_drive

# 速度
vel = 1550
# 转向角
angle = 1500
t = 0
num = 0
q = Queue()
ser = None


def lane():
    global vel
    global angle
    global t
    global t_end
    global num
    global ser

    def dataset(frame):
        lower_hsv = np.array([26, 43, 46])
        upper_hsv = np.array([34, 255, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        img = Image.fromarray(mask)
        img = img.resize((120, 120), Image.ANTIALIAS)
        img = np.array(img).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.transpose((2, 0, 1))
        img = img[(2, 1, 0), :, :] / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    # 加载模型
    save_path = "../model/model_infer/"
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)
    ser = serial.Serial('/dev/ttyACM0', 38400)
    time.sleep(1)
    cap = cv2.VideoCapture('/dev/cam_lane')
    while True:
        ret, frame = cap.read()
        if ret == True:
            img = dataset(frame)
            result = exe.run(program=infer_program, feed={feeded_var_names[0]: img}, fetch_list=target_var)
            angle = result[0][0][0]
            angle = int(angle + 0.5)

            if angle < 1100:
                angle = 1100
            if angle > 1900:
                angle = 1900

            if not q.empty() and num == 0:
                vel = q.get()
                # print('vel=', vel)
                if vel == 1505:
                    num = 1
                ser.write(car_drive(vel, 1500))

            elif not q.empty():
                a = q.get()
                pass
            if vel == 1505 and num == 1:
                time.sleep(3)
                vel = 1550
                num = 2
                ser.write(car_drive(vel, angle))
            if num == 2:
                t += 1
                if int(t) >= 70:
                    num = 0
                    t = 0
                    t_end = 0
                    vel = 1550

            ser.write(car_drive(vel, angle))
            cv2.imshow('lane', frame)
            if cv2.waitKey(1) == 27:
                ser.write(car_drive(1500, 1500))
                cv2.destroyAllWindows()
                cap.release()
                sys.exit(0)
                break
        else:
            print('lane相机打不开')


def sign():
    global vel
    global angle
    device = select_device('cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    weights = '../model/yolov5_model/best.pt'
    # Load model

    # 加载模型
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    cap = cv2.VideoCapture('/dev/cam_sign')
    print('打开相机')

    while True:
        ret, image = cap.read()
        if ret == True:
            # image = cv2.resize(image, (120, 120))
            with torch.no_grad():
                img = letterbox(image, new_shape=320)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = model(img, augment=False)[0]
                pred = non_max_suppression(pred, 0.4, 0.5, classes=False, agnostic=False)

                f = pred[0].tolist()
                for i, det in enumerate(pred):
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
                            cv2.rectangle(image, (int(coor[0] * 2), int(coor[1] * 2)),
                                          (int(coor[2] * 2), int(coor[3] * 2)),
                                          (0, 255, 0), 7)
                            label = str(label)
                            print(label)
                            area = (coor[2] - coor[0]) * (coor[3] - coor[1])
                            print(area)
                            if label == 'cancel_10':
                                # 限速解除
                                if area >= 3000:
                                    print('限速')
                                    vel = 1540
                                    q.put(vel)
                            elif label == 'crossing':
                                if area >= 900:
                                    print('人行道')
                                    vel = 1505
                                    q.put(vel)
                            elif label == 'limit_10':
                                if area >= 1350:
                                    print('限速解除')
                                    vel = 1550
                                    q.put(vel)
                            elif label == 'turn_left':
                                # print('左转')
                                pass
                            elif label == 'uphill_slope':
                                if area >= 1500:
                                    print('上坡')
                                    vel = 1505
                                    q.put(vel)
                            elif label == 'paper_red':
                                if 520 >= area >= 335:
                                    print('红灯')
                                    vel = 1495
                                    q.put(vel)
                            elif label == 'paper_greend':
                                if area >= 250:
                                    print('绿灯')
                                    vel = 1545
                                    q.put(vel)
                        del coor

            cv2.imshow('sign', image)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                cap.release()
                sys.exit(0)
                break
        else:
            print('sign相机打不开')


if __name__ == '__main__':
    lane_run = Process(target=lane)
    sign_run = Process(target=sign)

    lane_run.start()
    sign_run.start()
