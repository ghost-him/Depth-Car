import cv2
import os, time, sys
import numpy as np
from Create_Data_Liet import create_data
def lane_hsv():
    print('正在提取车道线信息，请勿停止程序....')
    img_path = "../data/images/"
    save_path = "../data/hsv_img/"
    img_names = os.listdir(img_path)
    img_names.sort(key=lambda x: int(x[:-4]))
    # 20220722
    # lower_hsv = np.array([20, 42, 206])
    # upper_hsv = np.array([66, 212, 255])
    # 20220725-1580-1
    # lower_hsv = np.array([23, 38, 161])
    # upper_hsv = np.array([78, 193, 255])
    # 20220725-1580-2
    # lower_hsv = np.array([23, 38, 161])
    # upper_hsv = np.array([49, 193, 255])
    # 20220725-1580-3
    # lower_hsv = np.array([27, 50, 230])
    # upper_hsv = np.array([65, 190, 255])
    # 20220726-1580-1
    lower_hsv = np.array([0, 50, 210])
    upper_hsv = np.array([58, 131, 255])
    for img in img_names:
        startTime = time.time()
        src = cv2.imread(img_path+img)
        # src = cv2.medianBlur(src, 21)  # 中值滤波
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        # kernel = np.ones((8, 8), dtype=np.uint8)  # 卷积核变为4*4
        # mask = cv2.dilate(mask, kernel, 1)
        # # 裁剪图片
        # mask = mask[0:260, 0:640]
        # mask1 = cv.inRange(hsv, lowerb=lower_hsv1, upperb=upper_hsv1)
        # mask = mask0  # + mask1
        endTime = time.time()
        fps = round(1 / (endTime - startTime))
        result = mask.copy()
        cv2.putText(result, img + f'\tFPS:{fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow('result', result)
        cv2.imwrite(save_path+img, mask)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit(0)
            break
    print('提取完成！')
lane_hsv()
# create_data()
