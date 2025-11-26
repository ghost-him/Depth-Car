import numpy as np
import cv2
import paddle.fluid as fluid
from PIL import Image
import paddle
import serial
import time

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

def lane():
    global vel
    global angle
    global ser

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
    # car_drive(1500, 1500)
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

            # if 1450 <= angle <= 1550:
            #    vel = 1555
            # else:
            #    vel = 1535
            vel = 1545
            ser.write(car_drive(vel, angle))
            cv2.imshow('lane', frame)
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
    lane_run = Process(target=lane)

    lane_run.start()
