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
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (non_max_suppression, scale_coords)
from utils.torch_utils import select_device
from multiprocessing import Process, Queue
import time
from hex_change import car_drive


# 分段速度字典
vel_dic = {"start":1580,
           "Uturn":1570,
           "cross":1485,
           "c-2-s":1567,
           "slope":1486,
           "s-2-l":1572,
           "limit":1566,
           "cancl":1579,
           "slowd":1565,
           "redlt":1487,
           "green":1581}
# 速度
vel = vel_dic["start"]
# 转向角
angle = 1500

t = 0
num = 0
num_red = 0
counter = 0
counter_lane = 0
slow_flag = 0
detect_flag = 1
q = Queue()
qq = Queue()
ser = None


def clear_img_txt():
    # 清空图片
    os.system('rm ../data/running_img/*')
    os.system('rm ../data/images/*')
    os.system('rm ../data/hsv_img/*')
    # 清空操作记录
    with open('../data/running_data.txt', 'w') as f:
        f.truncate(0)
        f.close()


def lane():
    global vel
    global angle
    global t
    global num
    global num_red
    global ser
    global counter_lane

    def dataset(frame, counter_lane):
        lower_hsv = np.array([30, 60, 186])
        upper_hsv = np.array([71, 126, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        cv2.imshow('lane', mask)
        cv2.imwrite('../data/images/' + str(counter_lane) + '.jpg', frame)
        cv2.imwrite('../data/hsv_img/' + str(counter_lane) + '.jpg', mask)

        img = Image.fromarray(mask)
        img = img.resize((120, 120), Image.ANTIALIAS)
        img = np.array(img).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.transpose((2, 0, 1))
        img = img[(2, 1, 0), :, :] / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    # 加载模型
    save_path = "../model/model_infer0726_1580_2_100"
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
            img = dataset(frame, counter_lane)
            counter_lane += 1
            result = exe.run(program=infer_program, feed={feeded_var_names[0]: img}, fetch_list=target_var)
            angle = result[0][0][0]
            if not q.empty():
                if num == 0 and num_red != 1:
                    vel = q.get()
                else:
                    a = q.get()     # 忽略sign()进程发过来的消息
                    pass

            # 1. 起始直线段
            if vel == vel_dic["start"]:
                # 转向角过大时，判定进入U形弯
                if 1000 <= angle < 1300 or 2000 >= angle > 1700:
                    vel = vel_dic["Uturn"]
            # 2. U形弯
            elif vel == vel_dic["Uturn"]:
                # 减速过弯，转向角按比例缩放
                angle = int((angle - 1500) * 1.2 + 1500)
            # 3. 人行横道
            elif vel == vel_dic["cross"]:
                if num == 0:    # 首次接收人行横道停车指令
                    num = 1
                # 人行道 停车等待2秒，不遵循标志识别控速
                if num == 1:
                    # 车头微调
                    ser.write(car_drive(1485, angle))
                    time.sleep(0.1)
                    # 停车等待
                    ser.write(car_drive(1485, 1500))
                    time.sleep(2)
                    vel = vel_dic["c-2-s"]
                    num = 2
            # 4. 人行横道--上坡 过渡段
            elif vel == vel_dic["c-2-s"]:
                # 低速行驶，转向角成比例减小，并设置上下限，防止车头剧烈摆动
                angle = int((angle - 1500) * 0.6 + 1500)
                if angle < 500:
                    angle = 500
                if angle > 2500:
                    angle = 2500
                # 重新起步，一定时间内不遵循标志识别控速，越过需等待标志区
                if num == 2:
                    t += 1
                    if int(t) >= 70:
                        num = 0
                        t = 0
            # 5. 上坡
            elif vel == vel_dic["slope"]:
                if num == 0:  # 首次接收上坡停车指令
                    num = 1
                # 上坡 停车等待2秒，不遵循标志识别控速
                if num == 1:
                    # 车头微调
                    ser.write(car_drive(1485, angle))
                    time.sleep(0.1)
                    # 停车等待
                    ser.write(car_drive(1485, 1500))
                    time.sleep(2)
                    vel = vel_dic["s-2-l"]
                    num = 2
            # 6. 上坡--限速区 过渡段
            elif vel == vel_dic["s-2-l"]:
                # 低速行驶，转向角成比例减小，并设置上下限，防止车头剧烈摆动
                angle = int((angle - 1500) * 0.6 + 1500)
                if angle < 500:
                    angle = 500
                if angle > 2500:
                    angle = 2500
                # 重新起步，一定时间内不遵循标志识别控速，越过需等待标志区
                if num == 2:
                    t += 1
                    if int(t) >= 50:
                        num = 0
                        t = 0
            # 7. 限速区
            elif vel == vel_dic["limit"]:
                # 低速行驶，转向角成比例减小，并设置上下限，防止车头剧烈摆动
                angle = int((angle - 1500) * 0.6 + 1500)
                if angle < 1000:
                    angle = 1000
                if angle > 2000:
                    angle = 2000
            # 8. 解除限速，进入S弯和环岛
            elif vel == vel_dic["cancl"]:
                # 高速过弯，转向角成比例增大，并设置上下限，防止车头剧烈摆动
                angle = int((angle - 1500) * 1.1 + 1500)
                if angle < 200:
                    angle = 200
                if angle > 2800:
                    angle = 2800
                if num_red == 0:  # 首次接收解除限速指令
                    num_red = 1
                # S弯道 + 环岛 区忽略标志检测，以免红灯误判
                if num_red == 1:
                    t += 1
                    # 跨线程发送消息，停止标志检测
                    qq.put(0)
                    # print("忽略标志 ", t, ' 个周期')
                    if int(t) >= 350:
                        num_red = 2
                        t = 0
                        # 减速看红灯
                        vel = vel_dic["slowd"]
                        qq.put(1)
            # 9. 减速看红灯
            elif vel == vel_dic["slowd"]:
                # 低速行驶，转向角成比例减小，并设置上下限，防止车头剧烈摆动
                angle = int((angle - 1500) * 0.6 + 1500)
                if angle < 1000:
                    angle = 1000
                if angle > 2000:
                    angle = 2000
            # 10. 红灯停
            elif vel == vel_dic["redlt"]:
                # 车头微调
                ser.write(car_drive(1485, angle))
                time.sleep(0.1)
                # 停车等待
                ser.write(car_drive(1485, 1500))
                time.sleep(2)
                vel = vel_dic["green"]
            # 11. 绿灯行
            elif vel == vel_dic["green"]:
                # 转向角成比例增大，应对直角弯
                angle = int((angle - 1500) * 1.1 + 1500)
                if angle < 200:
                    angle = 200
                if angle > 2800:
                    angle = 2800
            ser.write(car_drive(vel, angle))

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
    global angle
    global counter
    global slow_flag
    global detect_flag
    device = select_device('cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    weights = '../model/yolov5_model/bestn.pt'
    # 加载模型
    # model = attempt_load(weights, device=device)  # load FP32 model
    model = DetectMultiBackend(weights, device=device, dnn=False, data='models/my.yaml', fp16=False)
    stride, pt = model.stride, model.pt
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    savPath = '../data/running_img/'
    vidPath = '../data/running_video/'
    # 设置编码格式、输出路径、fps、size
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    i = 1
    while os.path.isfile(vidPath + f'output_{i}.avi'):
        i += 1
    video = cv2.VideoWriter(vidPath + f'output_{i}.avi', fourcc, 30, (640, 480))

    cap = cv2.VideoCapture('/dev/cam_sign')
    print('打开相机')

    while True:
        startTime = time.time()
        ret, image = cap.read()
        if not qq.empty():
            detect_flag = qq.get()
        if ret == True:
            if detect_flag == 1:
                with torch.no_grad():
                    # Padded resize
                    img = letterbox(image, new_shape=256, stride=stride, auto=pt)[0]
                    # Convert
                    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    pred = model(img, augment=False, visualize=False)
                    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

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
                                cx = str(int((coor[2] + coor[0]) / 2))
                                cy = str(int((coor[3] + coor[1]) / 2))
                                area = (coor[2] - coor[0]) * (coor[3] - coor[1])
                                print(area, ' ', vel, ' ', angle)
                                cv2.putText(image, str(conf), (coor[0], coor[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2)
                                cv2.putText(image, label, (coor[0], coor[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                            2)
                                cv2.putText(image, str(area), (coor[0], coor[3] + 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)
                                cv2.putText(image, cx + ' ' + cy, (coor[0], coor[3] + 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)
                                if label == 'crossing':
                                    # 人行道
                                    if area >= 4000:
                                        print('人行道')
                                        vel = vel_dic["cross"]
                                        q.put(vel)
                                elif label == 'uphill_slope':
                                    # 上坡
                                    if area >= 7000:
                                        print('上坡,', area)
                                        vel = vel_dic["slope"]
                                        q.put(vel)
                                elif label == 'limit_10':
                                    # 限速
                                    if area >= 8000:
                                        print('限速')
                                        vel = vel_dic["limit"]
                                        q.put(vel)
                                elif label == 'cancel_10':
                                    # 限速解除
                                    if area >= 3000:
                                        print('解除限速')
                                        vel = vel_dic["cancl"]
                                        q.put(vel)
                                elif label == 'turn_left':
                                    pass
                                elif label == 'paper_red':
                                    # 红灯
                                    if 1500 >= area >= 1400:
                                        print('红灯,', area)
                                        vel = vel_dic["redlt"]
                                        q.put(vel)
                                elif label == 'paper_greend':
                                    # 绿灯
                                    if area >= 1400:
                                        print('绿灯')
                                        vel = vel_dic["green"]
                                        q.put(vel)
                            del coor

            endTime = time.time()
            fps = round(1.0 / (endTime - startTime), 2)
            cv2.putText(image, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            video.write(image)
            cv2.imwrite(savPath + str(counter) + '.jpg', image)
            counter += 1
            cv2.imshow('sign', image)
            k = cv2.waitKey(1)
            if k == 27:
                ser.write(car_drive(1500, 1500))
                cv2.destroyAllWindows()
                video.release()
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
