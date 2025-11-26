import cv2
import sys
import numpy as np

cap1 = cv2.VideoCapture('/dev/cam_lane')
cap2 = cv2.VideoCapture('/dev/cam_sign')
img_path = "../data/test_image/"
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1 == True or ret2 == True:
        cv2.imshow('lane', frame1)
        cv2.imshow('sign', frame2)
        cv2.moveWindow('sign', 100, 100)
        cv2.moveWindow('lane', 800, 100)

        if cv2.waitKey(1) == 27:
            cv2.imwrite(img_path + 'test.jpg', frame1)
            lower_hsv = np.array([33, 50, 100])
            upper_hsv = np.array([60, 160, 255])
            src = cv2.imread(img_path + 'test.jpg')
            hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
            cv2.imwrite(img_path + 'result.jpg', mask)
            cv2.destroyAllWindows()
            cap1.release()
            cap2.release()
            sys.exit(0)
            break
    else:
        print('lane相机打不开')
        break