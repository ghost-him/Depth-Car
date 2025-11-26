# -*- coding: utf-8 -*-

"""
@author: ZZX
@project: table_Deep
@file: contrl.py
@function: 小车的手柄控制程序
@time: 2021/2/1 下午7:52
"""

import os
import sys, time
import struct, array
from fcntl import ioctl
import cv2
import numpy as np
import threading
from hex_change import car_drive
import serial

speed = 1500
angle = 1500
# 保存图像
get_image = 0
# 结束程序
break_f = 0
ser = None


def getvalue():
    for fn in os.listdir('/dev/input'):
        if fn.startswith('js'):
            print('/dev/input/%s' % fn)

    axis_states = {}
    button_states = {}

    axis_names = {
        0x00: 'x',
        0x01: 'y',
        0x02: 'z',
        0x03: 'rx',
        0x04: 'ry',
        0x05: 'rz',
        0x06: 'trottle',
        0x07: 'rudder',
        0x08: 'wheel',
        0x09: 'gas',
        0x0a: 'brake',
        0x10: 'hat0x',
        0x11: 'hat0y',
        0x12: 'hat1x',
        0x13: 'hat1y',
        0x14: 'hat2x',
        0x15: 'hat2y',
        0x16: 'hat3x',
        0x17: 'hat3y',
        0x18: 'pressure',
        0x19: 'distance',
        0x1a: 'tilt_x',
        0x1b: 'tilt_y',
        0x1c: 'tool_width',
        0x20: 'volume',
        0x28: 'misc',
    }
    button_names = {
        0x120: 'trigger',
        0x121: 'thumb',
        0x122: 'thumb2',
        0x123: 'top',
        0x124: 'top2',
        0x125: 'pinkie',
        0x126: 'base',
        0x127: 'base2',
        0x128: 'base3',
        0x129: 'base4',
        0x12a: 'base5',
        0x12b: 'base6',
        0x12f: 'dead',
        0x130: 'a',
        0x131: 'b',
        0x132: 'c',
        0x133: 'x',
        0x134: 'y',
        0x135: 'z',
        0x136: 'tl',
        0x137: 'tr',
        0x138: 'tl2',
        0x139: 'tr2',
        0x13a: 'select',
        0x13b: 'start',
        0x13c: 'mode',
        0x13d: 'thumbl',
        0x13e: 'thumbr',

        0x220: 'dpad_up',
        0x221: 'dpad_down',
        0x222: 'dpad_left',
        0x223: 'dpad_right',

        # XBox 360 controller uses these codes.
        0x2c0: 'dpad_left',
        0x2c1: 'dpad_right',
        0x2c2: 'dpad_up',
        0x2c3: 'dpad_down',
    }

    axis_map = []
    button_map = []

    fn = '/dev/input/js0'
    jsdev = open(fn, 'rb')

    buf = array.array('u', str(['\0'] * 5))
    ioctl(jsdev, 0x80006a13 + (0x10000 * len(buf)), buf)
    js_name = buf.tostring()
    # js_name = buf.tobytes().decode('utf-8')
    # print('device name: %s' % js_name)

    # get number of axes and buttons
    buf = array.array('B', [0])
    ioctl(jsdev, 0x80016a11, buf)  # JSIOCGAXES
    num_axes = buf[0]

    buf = array.array('B', [0])
    ioctl(jsdev, 0x80016a12, buf)  # JSIOCGBUTTONS
    num_buttons = buf[0]

    # Get the axis map
    buf = array.array('B', [0] * 0x40)
    ioctl(jsdev, 0x80406a32, buf)  # JSIOCGAXMAP
    for axis in buf[:num_axes]:
        # print(axis)
        axis_name = axis_names.get(axis, 'unknow(0x%02x)' % axis)
        axis_map.append(axis_name)
        axis_states[axis_name] = 0.0

    # Get the button map.
    buf = array.array('H', [0] * 200)
    ioctl(jsdev, 0x80406a34, buf)  # JSIOCGBTNMAP

    for btn in buf[:num_buttons]:
        btn_name = button_names.get(btn, 'unknown(0x%03x)' % btn)
        button_map.append(btn_name)
        button_states[btn_name] = 0

    return axis_map, axis_states, button_map, button_states


def control_car_process():
    global speed
    global angle
    global get_image
    global break_f
    global ser

    stop_num = 0
    ser = serial.Serial('/dev/ttyACM0', 38400)
    time.sleep(1)
    ser.write(car_drive(1500, 1500))
    while True:
        fn = '/dev/input/js0'
        jsdev = open(fn, 'rb')

        axis_map, axis_states, button_map, button_states = getvalue()

        while True:
            evbuf = jsdev.read(8)
            if evbuf:
                time_handle, value, type_b, number = struct.unpack('IhBB', evbuf)
                if type_b == 1:
                    button = button_map[number]
                    # 按键 启停
                    if button:
                        if (button == "b" and value == True):
                            # print("START")
                            speed = 1550
                            get_image = 1
                            stop_num = 0
                        if ((button == "tr" and value == True)):
                            # print("Stop")
                            speed = 1500
                            get_image = 0
                            stop_num += 1
                            if stop_num == 2:
                                break_f = 1
                                speed = 1500
                                angle = 1500
                                sys.exit(0)

                # 摇杆 转向
                if type_b == 2:
                    axis = axis_map[number]
                    if axis == "x":
                        fvalue = value / 32767
                        angle = int(1500 - (fvalue * 400))
                ser.write(car_drive(speed, angle))
    ser.close()


def cam():
    global speed
    global angle
    global break_f
    global get_image
    os.system('rm -rf ../data/*')
    os.system('mkdir ../data/images/')
    os.system('mkdir ../data/hsv_img/')
    os.system('touch ../data/data.txt')
    cap = cv2.VideoCapture('/dev/cam_lane')
    # cap.set(14, 0.5)

    num = 0
    while True:
        ret, frame = cap.read()
        if ret == True:
            if get_image == 1:
                cv2.imwrite('../data/images/' + str(num) + '.jpg', frame)
                with open('../data/data.txt', 'a') as f:
                    f.write(str(angle) + '\n')
                num += 1
            cv2.imshow('lane', frame)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                cap.release()
                sys.exit(0)
                break
            if break_f == 1:
                angledata = []
                with open('../data/data.txt', 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip('\n')
                        angledata.append(int(line))
                angle = np.array(angledata)
                np.save('../data/data.npy', angle, False)
                cv2.destroyAllWindows()
                cap.release()
                # os.system('python Img_Handle.py')
                sys.exit(0)
                break
        else:
            print('相机打不开')


if __name__ == '__main__':
    control_car_run = threading.Thread(target=control_car_process)
    cam_run = threading.Thread(target=cam)

    control_car_run.start()
    cam_run.start()

    control_car_run.join()
    cam_run.join()