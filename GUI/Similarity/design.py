# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1150, 777)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.outputWidget = QtWidgets.QWidget(self.centralwidget)
        self.outputWidget.setGeometry(QtCore.QRect(560, 150, 561, 381))
        self.outputWidget.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.outputWidget.setObjectName("outputWidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(240, 130, 70, 18))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(810, 130, 70, 18))
        self.label_2.setObjectName("label_2")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(0, 150, 541, 601))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.showFramesWidget = QtWidgets.QWidget(self.widget)
        self.showFramesWidget.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.showFramesWidget.setObjectName("showFramesWidget")
        self.verticalLayout.addWidget(self.showFramesWidget)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.GoLeft = QtWidgets.QPushButton(self.widget)
        self.GoLeft.setText("")
        icon = QtGui.QIcon.fromTheme("go-previous")
        self.GoLeft.setIcon(icon)
        self.GoLeft.setObjectName("GoLeft")
        self.horizontalLayout.addWidget(self.GoLeft)
        self.ChooseImage = QtWidgets.QPushButton(self.widget)
        self.ChooseImage.setObjectName("ChooseImage")
        self.horizontalLayout.addWidget(self.ChooseImage)
        self.GoRight = QtWidgets.QPushButton(self.widget)
        self.GoRight.setText("")
        icon = QtGui.QIcon.fromTheme("go-next")
        self.GoRight.setIcon(icon)
        self.GoRight.setObjectName("GoRight")
        self.horizontalLayout.addWidget(self.GoRight)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Ignored)
        self.verticalLayout_2.addItem(spacerItem)
        self.widget1 = QtWidgets.QWidget(self.widget)
        self.widget1.setObjectName("widget1")
        self.verticalLayout_2.addWidget(self.widget1)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.showFramesWidget.raise_()
        self.widget.raise_()
        self.showFramesWidget.raise_()
        self.outputWidget.raise_()
        self.label.raise_()
        self.label_2.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1150, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Find Closest Neighbors"))
        self.label.setText(_translate("MainWindow", "Frame"))
        self.label_2.setText(_translate("MainWindow", "Result"))
        self.ChooseImage.setText(_translate("MainWindow", "Ok"))

