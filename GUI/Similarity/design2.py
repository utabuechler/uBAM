# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design2.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1238, 871)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Label_info_inbetween = QtWidgets.QLabel(self.centralwidget)
        self.Label_info_inbetween.setGeometry(QtCore.QRect(730, 720, 461, 51))
        self.Label_info_inbetween.setText("")
        self.Label_info_inbetween.setObjectName("Label_info_inbetween")
        self.Frame_adjustBB = QtWidgets.QFrame(self.centralwidget)
        self.Frame_adjustBB.setEnabled(False)
        self.Frame_adjustBB.setGeometry(QtCore.QRect(80, 650, 261, 91))
        self.Frame_adjustBB.setStyleSheet("border-color: rgb(0, 0, 0);")
        self.Frame_adjustBB.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_adjustBB.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_adjustBB.setObjectName("Frame_adjustBB")
        self.Widget_changeBBSize = QtWidgets.QWidget(self.Frame_adjustBB)
        self.Widget_changeBBSize.setGeometry(QtCore.QRect(0, 30, 251, 60))
        self.Widget_changeBBSize.setObjectName("Widget_changeBBSize")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.Widget_changeBBSize)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(10, 10, 231, 39))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.Label_BBSize_desc = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.Label_BBSize_desc.setObjectName("Label_BBSize_desc")
        self.horizontalLayout_6.addWidget(self.Label_BBSize_desc)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem)
        self.Label_BBSize_x = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.Label_BBSize_x.setObjectName("Label_BBSize_x")
        self.horizontalLayout_6.addWidget(self.Label_BBSize_x)
        self.SpinBox_BBSize_x = QtWidgets.QSpinBox(self.horizontalLayoutWidget_3)
        self.SpinBox_BBSize_x.setMinimum(30)
        self.SpinBox_BBSize_x.setMaximum(200)
        self.SpinBox_BBSize_x.setProperty("value", 100)
        self.SpinBox_BBSize_x.setObjectName("SpinBox_BBSize_x")
        self.horizontalLayout_6.addWidget(self.SpinBox_BBSize_x)
        self.Label_BBSize_y = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.Label_BBSize_y.setObjectName("Label_BBSize_y")
        self.horizontalLayout_6.addWidget(self.Label_BBSize_y)
        self.SpinBox_BBSize_y = QtWidgets.QSpinBox(self.horizontalLayoutWidget_3)
        self.SpinBox_BBSize_y.setMinimum(30)
        self.SpinBox_BBSize_y.setMaximum(200)
        self.SpinBox_BBSize_y.setProperty("value", 100)
        self.SpinBox_BBSize_y.setObjectName("SpinBox_BBSize_y")
        self.horizontalLayout_6.addWidget(self.SpinBox_BBSize_y)
        self.horizontalLayoutWidget_3.raise_()
        self.RadioButton_adjustBB = QtWidgets.QRadioButton(self.Frame_adjustBB)
        self.RadioButton_adjustBB.setGeometry(QtCore.QRect(30, 10, 191, 23))
        self.RadioButton_adjustBB.setObjectName("RadioButton_adjustBB")
        self.Widget_changeBBSize.raise_()
        self.RadioButton_adjustBB.raise_()
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setEnabled(False)
        self.frame_2.setGeometry(QtCore.QRect(430, 350, 241, 91))
        self.frame_2.setStyleSheet("border-color: rgb(0, 0, 0);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.widget = QtWidgets.QWidget(self.frame_2)
        self.widget.setGeometry(QtCore.QRect(10, 10, 225, 70))
        self.widget.setObjectName("widget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.PushButton_play = QtWidgets.QPushButton(self.widget)
        self.PushButton_play.setObjectName("PushButton_play")
        self.horizontalLayout_9.addWidget(self.PushButton_play)
        self.PushButton_pause = QtWidgets.QPushButton(self.widget)
        self.PushButton_pause.setObjectName("PushButton_pause")
        self.horizontalLayout_9.addWidget(self.PushButton_pause)
        self.verticalLayout_3.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.Label_slider = QtWidgets.QLabel(self.widget)
        self.Label_slider.setObjectName("Label_slider")
        self.horizontalLayout_7.addWidget(self.Label_slider)
        self.DoubleSpinBox_secPerFrame = QtWidgets.QDoubleSpinBox(self.widget)
        self.DoubleSpinBox_secPerFrame.setMinimum(0.2)
        self.DoubleSpinBox_secPerFrame.setMaximum(5.0)
        self.DoubleSpinBox_secPerFrame.setSingleStep(0.2)
        self.DoubleSpinBox_secPerFrame.setProperty("value", 1.0)
        self.DoubleSpinBox_secPerFrame.setObjectName("DoubleSpinBox_secPerFrame")
        self.horizontalLayout_7.addWidget(self.DoubleSpinBox_secPerFrame)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(10, 70, 509, 121))
        self.frame_3.setStyleSheet("border-color: rgb(0, 0, 0);\n"
"border-width: 10")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.Label_info_project = QtWidgets.QLabel(self.frame_3)
        self.Label_info_project.setGeometry(QtCore.QRect(10, 40, 231, 31))
        self.Label_info_project.setStyleSheet("")
        self.Label_info_project.setText("")
        self.Label_info_project.setObjectName("Label_info_project")
        self.Label_info_video = QtWidgets.QLabel(self.frame_3)
        self.Label_info_video.setGeometry(QtCore.QRect(10, 70, 489, 41))
        self.Label_info_video.setStyleSheet("")
        self.Label_info_video.setText("")
        self.Label_info_video.setObjectName("Label_info_video")
        self.label_3 = QtWidgets.QLabel(self.frame_3)
        self.label_3.setGeometry(QtCore.QRect(10, 10, 171, 18))
        self.label_3.setObjectName("label_3")
        self.Widget_method = QtWidgets.QWidget(self.centralwidget)
        self.Widget_method.setEnabled(False)
        self.Widget_method.setGeometry(QtCore.QRect(710, 30, 491, 51))
        self.Widget_method.setObjectName("Widget_method")
        self.widget1 = QtWidgets.QWidget(self.Widget_method)
        self.widget1.setGeometry(QtCore.QRect(10, 10, 466, 30))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_2 = QtWidgets.QLabel(self.widget1)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_10.addWidget(self.label_2)
        self.ComboBox_compute = QtWidgets.QComboBox(self.widget1)
        self.ComboBox_compute.setObjectName("ComboBox_compute")
        self.ComboBox_compute.addItem("")
        self.ComboBox_compute.addItem("")
        self.horizontalLayout_10.addWidget(self.ComboBox_compute)
        self.PushButton_compute = QtWidgets.QPushButton(self.widget1)
        self.PushButton_compute.setObjectName("PushButton_compute")
        self.horizontalLayout_10.addWidget(self.PushButton_compute)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(690, 200, 521, 491))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.Label_showResult_title = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.Label_showResult_title.setObjectName("Label_showResult_title")
        self.verticalLayout_7.addWidget(self.Label_showResult_title)
        self.Label_showResult = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Label_showResult.sizePolicy().hasHeightForWidth())
        self.Label_showResult.setSizePolicy(sizePolicy)
        self.Label_showResult.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.Label_showResult.setText("")
        self.Label_showResult.setObjectName("Label_showResult")
        self.verticalLayout_7.addWidget(self.Label_showResult)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.PushButton_saveResult = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.PushButton_saveResult.setEnabled(False)
        self.PushButton_saveResult.setObjectName("PushButton_saveResult")
        self.horizontalLayout.addWidget(self.PushButton_saveResult)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout_7.addLayout(self.horizontalLayout)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(10, 200, 411, 421))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.Label_frame = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.Label_frame.setEnabled(True)
        self.Label_frame.setObjectName("Label_frame")
        self.verticalLayout_8.addWidget(self.Label_frame)
        self.Label_showFrame = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.Label_showFrame.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Label_showFrame.sizePolicy().hasHeightForWidth())
        self.Label_showFrame.setSizePolicy(sizePolicy)
        self.Label_showFrame.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.Label_showFrame.setText("")
        self.Label_showFrame.setObjectName("Label_showFrame")
        self.verticalLayout_8.addWidget(self.Label_showFrame)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.PushButton_goLeftLeft = QtWidgets.QPushButton(self.verticalLayoutWidget_4)
        self.PushButton_goLeftLeft.setObjectName("PushButton_goLeftLeft")
        self.horizontalLayout_3.addWidget(self.PushButton_goLeftLeft)
        self.PushButton_goLeft = QtWidgets.QPushButton(self.verticalLayoutWidget_4)
        self.PushButton_goLeft.setObjectName("PushButton_goLeft")
        self.horizontalLayout_3.addWidget(self.PushButton_goLeft)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem4)
        self.PushButton_goRight = QtWidgets.QPushButton(self.verticalLayoutWidget_4)
        self.PushButton_goRight.setObjectName("PushButton_goRight")
        self.horizontalLayout_3.addWidget(self.PushButton_goRight)
        self.PushButton_goRightRight = QtWidgets.QPushButton(self.verticalLayoutWidget_4)
        self.PushButton_goRightRight.setObjectName("PushButton_goRightRight")
        self.horizontalLayout_3.addWidget(self.PushButton_goRightRight)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem5)
        self.verticalLayout_8.addLayout(self.horizontalLayout_3)
        self.Widget_chooseVideo = QtWidgets.QWidget(self.centralwidget)
        self.Widget_chooseVideo.setGeometry(QtCore.QRect(0, 0, 471, 61))
        self.Widget_chooseVideo.setObjectName("Widget_chooseVideo")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.Widget_chooseVideo)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 10, 451, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.Label_video = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.Label_video.setObjectName("Label_video")
        self.horizontalLayout_2.addWidget(self.Label_video)
        self.ComboBox_chooseVideo = QtWidgets.QComboBox(self.horizontalLayoutWidget_2)
        self.ComboBox_chooseVideo.setEditable(False)
        self.ComboBox_chooseVideo.setObjectName("ComboBox_chooseVideo")
        self.horizontalLayout_2.addWidget(self.ComboBox_chooseVideo)
        self.PushButton_confirmVideo = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.PushButton_confirmVideo.setObjectName("PushButton_confirmVideo")
        self.horizontalLayout_2.addWidget(self.PushButton_confirmVideo)
        self.widget2 = QtWidgets.QWidget(self.centralwidget)
        self.widget2.setGeometry(QtCore.QRect(700, 90, 501, 81))
        self.widget2.setObjectName("widget2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget2)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(52)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.Frame_nrNN = QtWidgets.QFrame(self.widget2)
        self.Frame_nrNN.setEnabled(False)
        self.Frame_nrNN.setStyleSheet("border-color: rgb(0, 0, 0);")
        self.Frame_nrNN.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_nrNN.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_nrNN.setObjectName("Frame_nrNN")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.Frame_nrNN)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 160, 61))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.SpinBox_chooseNumberNN = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.SpinBox_chooseNumberNN.setMinimum(1)
        self.SpinBox_chooseNumberNN.setMaximum(50)
        self.SpinBox_chooseNumberNN.setProperty("value", 12)
        self.SpinBox_chooseNumberNN.setObjectName("SpinBox_chooseNumberNN")
        self.verticalLayout_2.addWidget(self.SpinBox_chooseNumberNN)
        self.verticalLayoutWidget.raise_()
        self.horizontalLayout_4.addWidget(self.Frame_nrNN)
        self.Frame_embSource = QtWidgets.QFrame(self.widget2)
        self.Frame_embSource.setEnabled(False)
        self.Frame_embSource.setStyleSheet("border-color: rgb(0, 0, 0);")
        self.Frame_embSource.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_embSource.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_embSource.setObjectName("Frame_embSource")
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.Frame_embSource)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(10, 10, 214, 61))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_5 = QtWidgets.QLabel(self.verticalLayoutWidget_5)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_9.addWidget(self.label_5)
        self.RadioButton_embedding = QtWidgets.QRadioButton(self.verticalLayoutWidget_5)
        self.RadioButton_embedding.setObjectName("RadioButton_embedding")
        self.verticalLayout_9.addWidget(self.RadioButton_embedding)
        self.horizontalLayout_4.addWidget(self.Frame_embSource)
        self.Label_info_inbetween.raise_()
        self.verticalLayoutWidget.raise_()
        self.Frame_adjustBB.raise_()
        self.verticalLayoutWidget_4.raise_()
        self.frame_2.raise_()
        self.ComboBox_compute.raise_()
        self.label_2.raise_()
        self.frame_3.raise_()
        self.Widget_method.raise_()
        self.RadioButton_adjustBB.raise_()
        self.RadioButton_adjustBB.raise_()
        self.Frame_nrNN.raise_()
        self.Frame_embSource.raise_()
        self.Frame_embSource.raise_()
        self.verticalLayoutWidget_3.raise_()
        self.PushButton_saveResult.raise_()
        self.verticalLayoutWidget_4.raise_()
        self.Widget_chooseVideo.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1238, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Similarity"))
        self.Label_BBSize_desc.setText(_translate("MainWindow", "Size:"))
        self.Label_BBSize_x.setText(_translate("MainWindow", "x"))
        self.Label_BBSize_y.setText(_translate("MainWindow", "y"))
        self.RadioButton_adjustBB.setText(_translate("MainWindow", "Adjust Area of Interest"))
        self.PushButton_play.setText(_translate("MainWindow", "Play"))
        self.PushButton_pause.setText(_translate("MainWindow", "Pause"))
        self.Label_slider.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">#Frames per Second</p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Video Information</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">3. Choose Method</span></p></body></html>"))
        self.ComboBox_compute.setItemText(0, _translate("MainWindow", "Nearest Neighbors"))
        self.ComboBox_compute.setItemText(1, _translate("MainWindow", "Low-dim Embedding"))
        self.PushButton_compute.setText(_translate("MainWindow", "Compute"))
        self.Label_showResult_title.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Result</p></body></html>"))
        self.PushButton_saveResult.setText(_translate("MainWindow", "Save"))
        self.Label_frame.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">2. Choose Frame and Area of Interest (ROI)</span></p></body></html>"))
        self.PushButton_goLeftLeft.setText(_translate("MainWindow", "-5"))
        self.PushButton_goLeft.setText(_translate("MainWindow", "-1"))
        self.PushButton_goRight.setText(_translate("MainWindow", "+1"))
        self.PushButton_goRightRight.setText(_translate("MainWindow", "+5"))
        self.Label_video.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">1. Choose Video</span></p></body></html>"))
        self.PushButton_confirmVideo.setText(_translate("MainWindow", "Load Video"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p>#Nearest Neighbors</p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "Embedding"))
        self.RadioButton_embedding.setText(_translate("MainWindow", "Show only Current Video"))
