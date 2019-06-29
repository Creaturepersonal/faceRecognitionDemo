import sys
import cv2
import time
import dlib
import threading
import face_detector
import numpy as np
import loadmodel
from copy import deepcopy
from facewindow import Ui_MainWindow
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageDraw, ImageFont
import os

os.environ['TF_CPP_MIN_LOG_LEVE'] = '3'


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)
        self.th = None
        self.current_image = None
        self.label_name_dic = {1: '梁朝伟', 2: '刘德华', 3: '马云', 4: '郭冬临', 5: '暴走的小吉吉', 6: '陈豆豆',
                               7: '古天乐', 8: '赵丽颖', 9: '邓超', 10: '孙俪', 11: '岳云鹏', 12: '沈腾',
                               13: '何炅', 14: '邓紫棋', 15: '李荣浩', 16: '陈赫', 17: '钟汉良', 18: '刘涛',
                               19: '冯提莫', 20: '王力宏', 21: '吴亦凡', 22: '张杰', 23: '张家辉', 24: '佟丽娅',
                               25: '杨洋'}
        self.alliswellnamedic = {0: '石天冬', 1: '苏大强', 2: '苏明成', 3: '苏明玉', 4: '苏明哲', 5: '吴非', 6: '朱丽'}
        self.friendnamedic = {0: '李梦旋', 1: '蒋畅', 2: '钟侠骄', 3: '陈洪', 4: '李廷川', 5: '陈毅', 6: '李任宁', 7: '汪林', 8: '张城阳',
                              9: '冯松', 10: '胡稳', 11: '林雪梅', 12: '熊英英', 13: '高小霞', 14: '邓杰', 15: '叶港培', 16: '李博录',
                              17: '刘天龙', 18: '刘泽城',19: '柏现迪'}
    #打开视频，使用了线程来开启视频
    def openStream322(self):
        self.DispImage.setText('')
        self.th = Thread(self)
        self.th.changePixmap.connect(self.set_video_image322)
        self.th.start()

    #拍照
    def snap322(self):
        ret, frame = self.th.cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # rgb_image = cv2.resize(rgb_image, (28, 28))
            self.current_image = rgb_image
            face_detector.detect_and_label(rgb_image)
            convert_to_QtFormat = QtGui.QImage(rgb_image.data, rgb_image.shape[1],
                                               rgb_image.shape[0],
                                               QImage.Format_RGB888)
            p = convert_to_QtFormat.scaled(320, 240, Qt.KeepAspectRatio)
            self.set_image322(self.ImgLabel2, p)

    #打开图片
    def openImage322(self):
        self.DispResult.setText('')
        self.DispImage.setText('')
        img_name, img_type = QFileDialog.getOpenFileName(self, "选择图片", "", " *.bmp;;*.jpg;;*.png;;*.jpeg")
        # print(img_name, img_type)
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        image = image / 255.
        image = cv2.resize(image, (128, 128))

        self.current_image = np.reshape(image, (1, 128, 128, 1))
        # 利用ImgLabel1显示图片
        # 适应设计ImgLabel1时的大小
        png = QtGui.QPixmap(img_name).scaled(self.DispImage.width(), self.DispImage.height())
        self.DispImage.setPixmap(png)


    def ReadTestImage322(self):
        filename = 'D:\Comprehensive3\FaceRecognition\DataSet2-FaceOfStar\\test\\17_333.bmp'
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255.
        image = cv2.resize(image, (128, 128))
        image = np.reshape(image, (1, 128, 128, 1))
        return image

    #识别函数
    def recognize322(self):
        result = loadmodel.predict322(self.current_image)
        print(result)
        if result != None:
            print(self.friendnamedic[np.array(result)[0]])
            self.DispResult.setText('')
            self.DispResult.insertPlainText(self.friendnamedic[np.array(result)[0]])
        else:
            print('未知')
            self.DispResult.setText('')
            self.DispResult.insertPlainText('未知')

    #实时识别函数
    def RealTimeRecognition322(self, image):
        result = loadmodel.predictvideo322(image)
        return result


    def set_video_image322(self, image):
        self.set_image322(self.DispImage, image)

    def set_image322(self, label, image):
        label.setPixmap(QPixmap.fromImage(image))

#播放视频线程
class Thread(QThread):
    def __init__(self, other):
        super(Thread, self).__init__()
        self.cap = None
        self.pause = False
        self.dlib_face_detector = dlib.get_frontal_face_detector()
        self.friendnamedic = {0: '李梦旋', 1: '蒋畅', 2: '钟侠骄', 3: '陈洪', 4: '李廷川', 5: '陈毅', 6: '李任宁', 7: '汪林', 8: '张城阳',
                              9: '冯松', 10: '胡稳', 11: '林雪梅', 12: '熊英英', 13: '高小霞', 14: '邓杰', 15: '叶港培', 16: '李博录',
                              17: '刘天龙', 18: '刘泽城', 19: '柏现迪'}

    changePixmap = pyqtSignal(QtGui.QImage)
    def deal322(self, image, rgb_image):
        dets = self.dlib_face_detector(image, 1)
        tempimg = deepcopy(image)
        if len(dets) != 0:
            for detection in dets:
                image = deepcopy(tempimg)
                image1 = image[detection.top():detection.bottom(), detection.left():detection.right()]
                x = detection.left()
                y = detection.top()
                image1 = cv2.equalizeHist(image1)
                image1 = image1 / 255.0
                image1 = cv2.resize(image1, (128, 128))
                image1 = np.reshape(image1, (1, 128, 128, 1))
                result = Window.RealTimeRecognition322(self, image1)

                if result != None and result <=19:
                    # 向视频中添加字符串
                    # cv2.putText(rgb_image, self.friendnamedic[result],
                    #             (x, y+10),  # 坐标
                    #             cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                    #             2,  # 字号
                    #             (255, 0, 255),  # 颜色
                    #             3)#粗细
                    # 向视频中添加中文名字
                    rgb_image = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
                    ttfont = ImageFont.truetype("simhei.ttf", 20)
                    draw = ImageDraw.Draw(rgb_image)
                    draw.text((x, y), self.friendnamedic[result], fill=(255, 0, 255), font=ttfont)
                    rgb_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
                    convert_to_QtFormat = QtGui.QImage(rgb_image.data, rgb_image.shape[1],
                                                       rgb_image.shape[0],
                                                       QImage.Format_RGB888)
                    p = convert_to_QtFormat.scaled(411, 321, Qt.KeepAspectRatio)
                    self.changePixmap.emit(p)
                else:
                    rgb_image = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
                    ttfont = ImageFont.truetype("simhei.ttf", 20)
                    draw = ImageDraw.Draw(rgb_image)
                    draw.text((x, y), '未知', fill=(255, 0, 255), font=ttfont)
                    rgb_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
                    convert_to_QtFormat = QtGui.QImage(rgb_image.data, rgb_image.shape[1],
                                                       rgb_image.shape[0],
                                                       QImage.Format_RGB888)
                    p = convert_to_QtFormat.scaled(411, 321, Qt.KeepAspectRatio)
                    self.changePixmap.emit(p)
        else:
            convert_to_QtFormat = QtGui.QImage(rgb_image.data, rgb_image.shape[1],
                                               rgb_image.shape[0],
                                               QImage.Format_RGB888)
            p = convert_to_QtFormat.scaled(411, 321, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 360)  # 设置分辨率，分辨率太大界面会卡
        self.cap.set(4, 300)
        while self.cap.isOpened():
            if 1 - self.pause:
                ret, frame = self.cap.read()
                if ret:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 在这里可以对每帧图像进行处理
                    image = deepcopy(rgb_image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    face_detector.detect_and_label(rgb_image)
                    self.deal322(image, rgb_image)

    def FaceDetecttionCut(self, img):
        dets = self.dlib_face_detector(img, 1)
        if len(dets) != 0:
            for detection in dets:
                # img = cv2.rectangle(img,
                #                     (detection.left(), detection.top()),  # (x1,y1)
                #                     (detection.right(), detection.bottom()),  # (x2,y2)
                #                     (255, 255, 255),
                #                     2)
                img = img[detection.top():detection.bottom(), detection.left():detection.right()]
                return img
        else:
            return np.array([0])

    def threcognize(self):
        t = threading.Thread(target=Window.RealTimeRecognition, args=(self,))
        t.start()

#主函数
def main():
    qtapp = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(qtapp.exec_())

if __name__ == '__main__':
    main()
