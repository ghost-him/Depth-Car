import numpy as np
import cv2
import paddle.fluid as fluid
from PIL import Image
import paddle
import serial
import time
import os

try:
    paddle.enable_static()
except:
    print('\n正在使用低版本的飞桨框架！')
import sys
from multiprocessing import Process
from hex_change import car_drive

# 速度
vel = 1550
# 转向角
angle = 1500
ser = None
num = 0
T1 = 0

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
    global ser
    global T1

    def dataset(frame):
        global num
        # lower_hsv = np.array([33, 50, 100])
        # upper_hsv = np.array([60, 150, 255])
        lower_hsv = np.array([30, 50, 100])
        upper_hsv = np.array([60, 255, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        cv2.imshow('detect', mask)
        cv2.imwrite('../data/images/' + str(num) + '.jpg', frame)
        cv2.imwrite('../data/running_img/' + str(num) + '.jpg', mask)
        num += 1
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
    # car_drive(1500, 1500)
    ser = serial.Serial('/dev/ttyACM0', 38400)
    time.sleep(1)
    cap = cv2.VideoCapture('/dev/cam_lane')
    T1 = time.clock()
    while True:
        ret, frame = cap.read()
        if ret == True:
            img = dataset(frame)
            result = exe.run(program=infer_program, feed={feeded_var_names[0]: img}, fetch_list=target_var)
            angle = result[0][0][0]
            # angle = int(angle + 0.5)
            angle = int((angle -1500) * 1.5 + 1500)
            if angle < 100:
                angle = 100
            if angle > 2900:
                angle = 2900

            if 1450 <= angle <= 1550:
               vel = 1600
            elif angle <= 500 or angle >= 2500:
               vel = 1555
            else:
                vel = 1575
	        # 最大速度目前 1575 转弯系数2.5
            # vel = 1545
            ser.write(car_drive(vel, angle))
            cv2.imshow('lane', frame)
            T2 = time.clock()
            with open('../data/running_data.txt', 'a') as f:
                f.write(str(T2 - T1) + '\t' + str(vel) + '\t' + str(angle) + '\n')
            if cv2.waitKey(1) == 27:
                ser.write(car_drive(1500, 1500))
                # ser.close()
                cv2.destroyAllWindows()
                cap.release()
                sys.exit(0)
                break
        else:
            print('lane相机打不开')
            break
    ser.close()


if __name__ == '__main__':
    clear_img_txt()
    lane_run = Process(target=lane)
    # T1 = time.clock()
    lane_run.start()
