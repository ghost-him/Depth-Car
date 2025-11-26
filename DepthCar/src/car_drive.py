from ctypes import *
lib_path = "../lib/libart_driver.so"
so = cdll.LoadLibrary
lib = so(lib_path)
car = "/dev/ttyACM0"
if (lib.art_racecar_init(38400, car.encode("utf-8")) < 0):
    pass

def car_drive(vel, angle):
    lib.send_cmd(vel, angle)