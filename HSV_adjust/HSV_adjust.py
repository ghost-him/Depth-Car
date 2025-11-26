from PyQt5.QtWidgets import QFileDialog, QWidget, QApplication
from PyQt5.QtCore import QEvent
from PyQt5.QtGui import QPixmap, QImage
import cv2, sys
import numpy as np
from PIL import Image
from Find_HSV import Ui_Form


class WIN(QWidget):
    def __init__(self):
        super().__init__()
        self.imgName = None
        self.imgType = None
        self.originImg = None
        self.lowerLimit = np.array([0, 0, 0])
        self.upperLimit = np.array([255, 255, 255])
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.Lower_H_Slider.valueChanged.connect(self.showValue)
        self.ui.Lower_S_Slider.valueChanged.connect(self.showValue)
        self.ui.Lower_V_Slider.valueChanged.connect(self.showValue)
        self.ui.Upper_H_Slider.valueChanged.connect(self.showValue)
        self.ui.Upper_S_Slider.valueChanged.connect(self.showValue)
        self.ui.Upper_V_Slider.valueChanged.connect(self.showValue)

        self.ui.Img_Origin.installEventFilter(self)

    def eventFilter(self, obj, event):
        # print(obj.objectName(), ' ', event.type())
        if obj.objectName() == 'Img_Origin' and event.type() == QEvent.Type.MouseButtonPress:
            self.selectImg()
        return super().eventFilter(obj, event)

    def selectImg(self):
        self.imgName, self.imgType = QFileDialog.getOpenFileName(self, "选择图片", "../DepthCar/data/images",
                                                                 "*.jpg;;*.png;;*.jpeg;;All Files(*)")
        print("选择的图片为： ", self.imgName)
        self.originImg = QPixmap(self.imgName).scaled(self.ui.Img_Origin.width(), self.ui.Img_Origin.height())
        self.ui.Img_Origin.setPixmap(self.originImg)
        self.showHSV()

    # cv2图像转QImage图像  原文链接：https: // blog.csdn.net / qq_26696715 / article / details / 122597667
    def cv2QImage(self, data):
        # 8-bits unsigned, NO. OF CHANNELS=1
        if data.dtype == np.uint8:
            channels = 1 if len(data.shape) == 2 else data.shape[2]
        if channels == 3:  # CV_8UC3
            # Copy input Mat
            # Create QImage with same dimensions as input Mat
            img = QImage(data, data.shape[1], data.shape[0], data.strides[0], QImage.Format_RGB888)
            return img.rgbSwapped()
        elif channels == 1:
            # Copy input Mat
            # Create QImage with same dimensions as input Mat
            img = QImage(data, data.shape[1], data.shape[0], data.strides[0], QImage.Format_Indexed8)
            return img
        else:
            print("ERROR: numpy.ndarray could not be converted to QImage. Channels = %d" % data.shape[2])
            return QImage()


    def showHSV(self):
        src = cv2.imread(self.imgName)
        # src = cv2.medianBlur(src, 21)  # 中值滤波
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lowerLimit, self.upperLimit)
        Qimg = self.cv2QImage(mask)
        # cv2.imshow('result', mask)
        result = QPixmap.fromImage(Qimg).scaled(self.ui.Img_Result.width(), self.ui.Img_Result.height())
        self.ui.Img_Result.setPixmap(result)


    def showValue(self):
        self.ui.Lower_H_Value.setText(str(self.ui.Lower_H_Slider.value()))
        self.ui.Lower_S_Value.setText(str(self.ui.Lower_S_Slider.value()))
        self.ui.Lower_V_Value.setText(str(self.ui.Lower_V_Slider.value()))
        self.ui.Upper_H_Value.setText(str(self.ui.Upper_H_Slider.value()))
        self.ui.Upper_S_Value.setText(str(self.ui.Upper_S_Slider.value()))
        self.ui.Upper_V_Value.setText(str(self.ui.Upper_V_Slider.value()))
        self.lowerLimit = np.array([self.ui.Lower_H_Slider.value(),
                                    self.ui.Lower_S_Slider.value(),
                                    self.ui.Lower_V_Slider.value()])
        self.upperLimit = np.array([self.ui.Upper_H_Slider.value(),
                                    self.ui.Upper_S_Slider.value(),
                                    self.ui.Upper_V_Slider.value()])
        self.showHSV()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWindow = WIN()
    myWindow.show()
    sys.exit(app.exec_())
