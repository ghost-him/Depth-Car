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
vel = 1561
# 转向角
angle = 1500
t = 0
num = 0
num_red = 0
isFisrtU = 0
counter = 0
counter_lane = 0
slow_flag = 0
q = Queue()
ser = None
last_label = 'start'


def clear_img_txt():
    # 清空图片
    os.system('rm ../data/running_img/*')
    # 清空操作记录
    with open('../data/running_data.txt', 'w') as f:
        f.truncate(0)
        f.close()


def lane():
    global vel
    global angle
    global t
    global t_end
    global num
    global num_red
    global ser
    global counter_lane
    global isFisrtU

    def dataset(frame):
        lower_hsv = np.array([28, 50, 100])
        upper_hsv = np.array([60, 170, 255])
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
            # angle = int(angle + 0.5)
            # angle = int((angle - 1500) * 2 + 1500)
            # if angle < 100:
            #     angle = 100
            # if angle > 2900:
            #     angle = 2900
            # 进入限速区
            if vel == 1546:
                angle = int((angle - 1500) * 0.7 + 1500)
                if angle < 500:
                    angle = 500
                if angle > 2500:
                    angle = 2500
            # 进入S弯
            elif vel == 1556:
                angle = int((angle - 1500) * 0.9 + 1500)
                if angle < 500:
                    angle = 500
                if angle > 2500:
                    angle = 2500
            # 初始路段
            elif vel == 1561:
                angle = int((angle - 1500) * 2 + 1500)
                if angle < 1100:
                    angle = 1100
                    vel = 1541
                if angle > 1900:
                    angle = 1900
                    vel = 1541
            # 初始弯道
            elif vel == 1541:
                isFisrtU = 1
                angle = int((angle - 1500) * 0.7 + 1500)
                if angle < 1100:
                    angle = 1100
                if angle > 1900:
                    angle = 1900
            # 最后直角弯
            elif vel == 1570:
                angle = int((angle - 1500) * 1.2 + 1500)
                if angle < 1500 - 400 * 1.2:
                    angle = 1500 - 400 * 1.2
                if angle > 1500 + 400 * 1.2:
                    angle = 1500 + 400 * 1.2
            # 其他路段
            else:
                angle = int((angle - 1500) * 1 + 1500)
                if angle < 500:
                    angle = 500
                if angle > 2500:
                    angle = 2500
            if not q.empty() and num == 0 and num_red ==0:
                vel = q.get()
                # 接收到停车指令
                if vel == 1485:
                    num = 1
                # 进入S弯
                if vel == 1556:
                    num_red = 1
                ser.write(car_drive(vel, angle))
            elif not q.empty():
                a = q.get()
                pass
            # 停车等待3秒，不遵循标志识别控速
            if vel == 1485 and num == 1:
                time.sleep(0.5)
                ser.write(car_drive(vel, 1500))
                time.sleep(2)
                vel = 1560
                num = 2
                ser.write(car_drive(vel, angle))
            # 重新起步，越过需等待标志区，不遵循标志识别控速
            if num == 2:
                if isFisrtU == 1:
                    angle = int((angle - 1500) * 0.5 + 1500)
                    if angle < 1500 - 400 * 0.5:
                        angle = 1500 - 400 * 0.5
                    if angle > 1500 + 400 * 0.5:
                        angle = 1500 + 400 * 0.5
                t += 1
                if int(t) >= 70:
                    num = 0
                    t = 0
                    t_end = 0
                    isFisrtU = 0
                    vel = 1560
            # S弯道 + 环岛 区忽略标志检测，以免红灯误判
            if num == 0 and num_red == 1:
                t += 1
                # print("忽略标志 ", t, ' 个周期')
                if int(t) >= 470:
                    num_red = 0
                    t = 0
                    t_end = 0
                    vel = 1546
            ser.write(car_drive(vel, angle))
            cv2.imshow('lane', frame)
            cv2.moveWindow('lane', 100, 100)
            cv2.imwrite('../data/images/' + str(counter_lane) + '.jpg', frame)
            counter_lane += 1
            with open('../data/running_data.txt', 'a') as f:
                f.write(str(vel) + '\t' + str(angle) + '\n')
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
    global counter
    global last_label
    global slow_flag
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
                            # print(last_label, ' ', label)
                            area = (coor[2] - coor[0]) * (coor[3] - coor[1])
                            print(area, ' ', vel)
                            if label != 'paper_red':
                                cv2.putText(image, label, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.putText(image, str(area), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            else:
                                cv2.putText(image, label, (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.putText(image, str(area), (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            if label == 'cancel_10':
                                # 限速解除
                                if area >= 1000:
                                    print('解除限速')
                                    vel = 1556
                                    q.put(vel)
                            elif label == 'crossing':
                                if area >= 800:
                                    print('人行道')
                                    vel = 1485
                                    q.put(vel)
                            elif label == 'limit_10':
                                # if area >= 1350:
                                if area >= 550:
                                    print('限速')
                                    vel = 1546
                                    q.put(vel)
                            elif label == 'turn_left':
                                # print('左转')
                                # if slow_flag == 0 and (last_label == 'cancel_10' or last_label == 'paper_red'):
                                #     vel = 1545
                                #     slow_flag = 1
                                #     print('减速,', vel)
                                #     q.put(vel)
                                pass
                            elif label == 'uphill_slope':
                                if area >= 900:
                                    print('上坡,', area)
                                    vel = 1485
                                    q.put(vel)
                            elif label == 'paper_red':
                                # if 350 >= area >= 300:
                                if 300 >= area >= 245:
                                    print('红灯,', area)
                                    vel = 1486
                                    q.put(vel)
                                # elif slow_flag == 0 and 350 > area >= 100:
                                #     vel = 1545
                                #     slow_flag = 1
                                #     print('减速,', vel)
                                #     q.put(vel)
                            elif label == 'paper_greend':
                                if area >= 250:
                                    print('绿灯')
                                    vel = 1570
                                    q.put(vel)
                            last_label = label
                        del coor

            cv2.imshow('sign', image)
            cv2.moveWindow('sign', 800, 100)
            cv2.imwrite('../data/running_img/' + str(counter) + '.jpg', image)
            counter += 1
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                cap.release()
                sys.exit(0)
                break
        else:
            print('sign相机打不开')


if __name__ == '__main__':
    clear_img_txt()
    lane_run = Process(target=lane)
    sign_run = Process(target=sign)

    lane_run.start()
    sign_run.start()
