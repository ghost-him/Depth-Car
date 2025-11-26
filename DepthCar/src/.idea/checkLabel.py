import os

imgPath = '../../../labelImg/images/'
lblPath = '../../../labelImg/001/'
# D:\dlCar2\DepthCar\src\.idea\checkLabel.py
imgNames = os.listdir(imgPath)
lblNames = os.listdir(lblPath)

# lbl = lblNames[0]
# num = lbl[:-4]
# img = '1489.jpg'
# if img == num + '.jpg':
#     pass
# else:
#     print('get ', img)

for img in imgNames:
    num = img[:-4]
    count = 0
    for lbl in lblNames:
        if lbl == num + '.xml':
            count += 1
    if count == 0:
        print(img)