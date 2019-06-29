# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'face.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1093, 770)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.photo = QtWidgets.QPushButton(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(20, 20, 93, 28))
        self.photo.setObjectName("photo")
        self.OpenCamera = QtWidgets.QPushButton(self.centralwidget)
        self.OpenCamera.setGeometry(QtCore.QRect(140, 20, 93, 28))
        self.OpenCamera.setObjectName("OpenCamera")
        self.RecoResult = QtWidgets.QLabel(self.centralwidget)
        self.RecoResult.setGeometry(QtCore.QRect(510, 500, 72, 15))
        self.RecoResult.setObjectName("RecoResult")
        self.StartReco = QtWidgets.QPushButton(self.centralwidget)
        self.StartReco.setGeometry(QtCore.QRect(500, 640, 93, 28))
        self.StartReco.setObjectName("StartReco")
        self.DispResult = QtWidgets.QTextBrowser(self.centralwidget)
        self.DispResult.setGeometry(QtCore.QRect(420, 520, 256, 101))
        self.DispResult.setObjectName("DispResult")
        self.DispImage = QtWidgets.QLabel(self.centralwidget)
        self.DispImage.setGeometry(QtCore.QRect(340, 120, 411, 321))
        # self.DispImage.setGeometry(QtCore.QRect(340, 120, 600, 400))
        self.DispImage.setText("")
        self.DispImage.setObjectName("DispImage")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1093, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.OpenCamera.clicked.connect(MainWindow.openStream322)
        self.photo.clicked.connect(MainWindow.openImage322)
        self.StartReco.clicked.connect(MainWindow.recognize322)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "FaceRecognition"))
        self.photo.setText(_translate("MainWindow", "选择图片"))
        self.OpenCamera.setText(_translate("MainWindow", "开启摄像头"))
        self.RecoResult.setText(_translate("MainWindow", "识别结果"))
        self.StartReco.setText(_translate("MainWindow", "开始识别"))

