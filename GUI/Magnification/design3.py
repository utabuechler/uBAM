# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI/Magnification/design3.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1337, 892)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/compvis.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.Widget_chooseVideo = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Widget_chooseVideo.sizePolicy().hasHeightForWidth())
        self.Widget_chooseVideo.setSizePolicy(sizePolicy)
        self.Widget_chooseVideo.setMaximumSize(QtCore.QSize(550, 16777215))
        self.Widget_chooseVideo.setObjectName("Widget_chooseVideo")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.Widget_chooseVideo)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.Label_video = QtWidgets.QLabel(self.Widget_chooseVideo)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Label_video.sizePolicy().hasHeightForWidth())
        self.Label_video.setSizePolicy(sizePolicy)
        self.Label_video.setObjectName("Label_video")
        self.horizontalLayout_2.addWidget(self.Label_video)
        self.ComboBox_chooseVideo = QtWidgets.QComboBox(self.Widget_chooseVideo)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ComboBox_chooseVideo.sizePolicy().hasHeightForWidth())
        self.ComboBox_chooseVideo.setSizePolicy(sizePolicy)
        self.ComboBox_chooseVideo.setMinimumSize(QtCore.QSize(250, 0))
        self.ComboBox_chooseVideo.setEditable(False)
        self.ComboBox_chooseVideo.setObjectName("ComboBox_chooseVideo")
        self.horizontalLayout_2.addWidget(self.ComboBox_chooseVideo)
        self.PushButton_confirmVideo = QtWidgets.QPushButton(self.Widget_chooseVideo)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PushButton_confirmVideo.sizePolicy().hasHeightForWidth())
        self.PushButton_confirmVideo.setSizePolicy(sizePolicy)
        self.PushButton_confirmVideo.setObjectName("PushButton_confirmVideo")
        self.horizontalLayout_2.addWidget(self.PushButton_confirmVideo)
        self.gridLayout_6.addWidget(self.Widget_chooseVideo, 0, 0, 1, 1)
        self.Widget_chooseReprType = QtWidgets.QWidget(self.centralwidget)
        self.Widget_chooseReprType.setEnabled(False)
        self.Widget_chooseReprType.setObjectName("Widget_chooseReprType")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.Widget_chooseReprType)
        self.horizontalLayout_4.setSpacing(21)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.Label_chooseReprType = QtWidgets.QLabel(self.Widget_chooseReprType)
        self.Label_chooseReprType.setObjectName("Label_chooseReprType")
        self.horizontalLayout_4.addWidget(self.Label_chooseReprType)
        self.ComboBox_chooseReprType = QtWidgets.QComboBox(self.Widget_chooseReprType)
        self.ComboBox_chooseReprType.setObjectName("ComboBox_chooseReprType")
        self.ComboBox_chooseReprType.addItem("")
        self.ComboBox_chooseReprType.addItem("")
        self.horizontalLayout_4.addWidget(self.ComboBox_chooseReprType)
        self.gridLayout_6.addWidget(self.Widget_chooseReprType, 1, 0, 1, 1)
        self.Frame_vidInfo = QtWidgets.QFrame(self.centralwidget)
        self.Frame_vidInfo.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Frame_vidInfo.sizePolicy().hasHeightForWidth())
        self.Frame_vidInfo.setSizePolicy(sizePolicy)
        self.Frame_vidInfo.setMinimumSize(QtCore.QSize(550, 0))
        self.Frame_vidInfo.setStyleSheet("border-color: rgb(0, 0, 0);\n"
"border-width: 10")
        self.Frame_vidInfo.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_vidInfo.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_vidInfo.setObjectName("Frame_vidInfo")
        self.gridLayout = QtWidgets.QGridLayout(self.Frame_vidInfo)
        self.gridLayout.setObjectName("gridLayout")
        self.Label_info_title = QtWidgets.QLabel(self.Frame_vidInfo)
        self.Label_info_title.setObjectName("Label_info_title")
        self.gridLayout.addWidget(self.Label_info_title, 0, 0, 1, 1)
        self.Label_info_video = QtWidgets.QLabel(self.Frame_vidInfo)
        self.Label_info_video.setStyleSheet("")
        self.Label_info_video.setText("")
        self.Label_info_video.setObjectName("Label_info_video")
        self.gridLayout.addWidget(self.Label_info_video, 2, 0, 1, 1)
        self.Label_info_project = QtWidgets.QLabel(self.Frame_vidInfo)
        self.Label_info_project.setStyleSheet("")
        self.Label_info_project.setText("")
        self.Label_info_project.setObjectName("Label_info_project")
        self.gridLayout.addWidget(self.Label_info_project, 1, 0, 1, 1)
        self.gridLayout_6.addWidget(self.Frame_vidInfo, 2, 0, 1, 2)
        self.Layout_chooseFrame = QtWidgets.QVBoxLayout()
        self.Layout_chooseFrame.setContentsMargins(12, -1, 12, -1)
        self.Layout_chooseFrame.setObjectName("Layout_chooseFrame")
        self.Label_frame = QtWidgets.QLabel(self.centralwidget)
        self.Label_frame.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Label_frame.sizePolicy().hasHeightForWidth())
        self.Label_frame.setSizePolicy(sizePolicy)
        self.Label_frame.setObjectName("Label_frame")
        self.Layout_chooseFrame.addWidget(self.Label_frame)
        self.Label_showFrame = QtWidgets.QLabel(self.centralwidget)
        self.Label_showFrame.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Label_showFrame.sizePolicy().hasHeightForWidth())
        self.Label_showFrame.setSizePolicy(sizePolicy)
        self.Label_showFrame.setMinimumSize(QtCore.QSize(450, 400))
        self.Label_showFrame.setMaximumSize(QtCore.QSize(500, 16777215))
        self.Label_showFrame.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.Label_showFrame.setText("")
        self.Label_showFrame.setObjectName("Label_showFrame")
        self.Layout_chooseFrame.addWidget(self.Label_showFrame)
        self.Widget_buttons_frame = QtWidgets.QWidget(self.centralwidget)
        self.Widget_buttons_frame.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Widget_buttons_frame.sizePolicy().hasHeightForWidth())
        self.Widget_buttons_frame.setSizePolicy(sizePolicy)
        self.Widget_buttons_frame.setMaximumSize(QtCore.QSize(500, 16777215))
        self.Widget_buttons_frame.setObjectName("Widget_buttons_frame")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.Widget_buttons_frame)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.PushButton_goLeftLeft = QtWidgets.QPushButton(self.Widget_buttons_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PushButton_goLeftLeft.sizePolicy().hasHeightForWidth())
        self.PushButton_goLeftLeft.setSizePolicy(sizePolicy)
        self.PushButton_goLeftLeft.setObjectName("PushButton_goLeftLeft")
        self.horizontalLayout_12.addWidget(self.PushButton_goLeftLeft)
        self.PushButton_goLeft = QtWidgets.QPushButton(self.Widget_buttons_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PushButton_goLeft.sizePolicy().hasHeightForWidth())
        self.PushButton_goLeft.setSizePolicy(sizePolicy)
        self.PushButton_goLeft.setObjectName("PushButton_goLeft")
        self.horizontalLayout_12.addWidget(self.PushButton_goLeft)
        self.PushButton_reloadFrame = QtWidgets.QPushButton(self.Widget_buttons_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PushButton_reloadFrame.sizePolicy().hasHeightForWidth())
        self.PushButton_reloadFrame.setSizePolicy(sizePolicy)
        self.PushButton_reloadFrame.setObjectName("PushButton_reloadFrame")
        self.horizontalLayout_12.addWidget(self.PushButton_reloadFrame)
        self.PushButton_clearFrame = QtWidgets.QPushButton(self.Widget_buttons_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PushButton_clearFrame.sizePolicy().hasHeightForWidth())
        self.PushButton_clearFrame.setSizePolicy(sizePolicy)
        self.PushButton_clearFrame.setObjectName("PushButton_clearFrame")
        self.horizontalLayout_12.addWidget(self.PushButton_clearFrame)
        self.PushButton_goRight = QtWidgets.QPushButton(self.Widget_buttons_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PushButton_goRight.sizePolicy().hasHeightForWidth())
        self.PushButton_goRight.setSizePolicy(sizePolicy)
        self.PushButton_goRight.setToolTip("")
        self.PushButton_goRight.setObjectName("PushButton_goRight")
        self.horizontalLayout_12.addWidget(self.PushButton_goRight)
        self.PushButton_goRightRight = QtWidgets.QPushButton(self.Widget_buttons_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PushButton_goRightRight.sizePolicy().hasHeightForWidth())
        self.PushButton_goRightRight.setSizePolicy(sizePolicy)
        self.PushButton_goRightRight.setToolTip("")
        self.PushButton_goRightRight.setObjectName("PushButton_goRightRight")
        self.horizontalLayout_12.addWidget(self.PushButton_goRightRight)
        self.Layout_chooseFrame.addWidget(self.Widget_buttons_frame)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        spacerItem = QtWidgets.QSpacerItem(42, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem)
        self.Frame_adjustBB = QtWidgets.QFrame(self.widget)
        self.Frame_adjustBB.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Frame_adjustBB.sizePolicy().hasHeightForWidth())
        self.Frame_adjustBB.setSizePolicy(sizePolicy)
        self.Frame_adjustBB.setStyleSheet("border-color: rgb(0, 0, 0);")
        self.Frame_adjustBB.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_adjustBB.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_adjustBB.setObjectName("Frame_adjustBB")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.Frame_adjustBB)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.RadioButton_adjustBB = QtWidgets.QRadioButton(self.Frame_adjustBB)
        self.RadioButton_adjustBB.setObjectName("RadioButton_adjustBB")
        self.verticalLayout_6.addWidget(self.RadioButton_adjustBB)
        self.Widget_changeBBSize = QtWidgets.QWidget(self.Frame_adjustBB)
        self.Widget_changeBBSize.setObjectName("Widget_changeBBSize")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.Widget_changeBBSize)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.Label_BBSize_desc = QtWidgets.QLabel(self.Widget_changeBBSize)
        self.Label_BBSize_desc.setObjectName("Label_BBSize_desc")
        self.horizontalLayout_6.addWidget(self.Label_BBSize_desc)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem1)
        self.Label_BBSize_x = QtWidgets.QLabel(self.Widget_changeBBSize)
        self.Label_BBSize_x.setObjectName("Label_BBSize_x")
        self.horizontalLayout_6.addWidget(self.Label_BBSize_x)
        self.SpinBox_BBSize_x = QtWidgets.QSpinBox(self.Widget_changeBBSize)
        self.SpinBox_BBSize_x.setMinimum(30)
        self.SpinBox_BBSize_x.setMaximum(500)
        self.SpinBox_BBSize_x.setProperty("value", 100)
        self.SpinBox_BBSize_x.setObjectName("SpinBox_BBSize_x")
        self.horizontalLayout_6.addWidget(self.SpinBox_BBSize_x)
        self.Label_BBSize_y = QtWidgets.QLabel(self.Widget_changeBBSize)
        self.Label_BBSize_y.setObjectName("Label_BBSize_y")
        self.horizontalLayout_6.addWidget(self.Label_BBSize_y)
        self.SpinBox_BBSize_y = QtWidgets.QSpinBox(self.Widget_changeBBSize)
        self.SpinBox_BBSize_y.setMinimum(30)
        self.SpinBox_BBSize_y.setMaximum(500)
        self.SpinBox_BBSize_y.setProperty("value", 100)
        self.SpinBox_BBSize_y.setObjectName("SpinBox_BBSize_y")
        self.horizontalLayout_6.addWidget(self.SpinBox_BBSize_y)
        self.gridLayout_3.addLayout(self.horizontalLayout_6, 0, 0, 1, 1)
        self.verticalLayout_6.addWidget(self.Widget_changeBBSize)
        self.horizontalLayout_11.addWidget(self.Frame_adjustBB)
        spacerItem2 = QtWidgets.QSpacerItem(21, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem2)
        self.Layout_chooseFrame.addWidget(self.widget)
        self.gridLayout_6.addLayout(self.Layout_chooseFrame, 3, 0, 1, 1)
        self.Widget_play = QtWidgets.QWidget(self.centralwidget)
        self.Widget_play.setObjectName("Widget_play")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.Widget_play)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        spacerItem3 = QtWidgets.QSpacerItem(20, 350, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout_4.addItem(spacerItem3)
        self.Frame_play = QtWidgets.QFrame(self.Widget_play)
        self.Frame_play.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Frame_play.sizePolicy().hasHeightForWidth())
        self.Frame_play.setSizePolicy(sizePolicy)
        self.Frame_play.setStyleSheet("border-color: rgb(0, 0, 0);")
        self.Frame_play.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_play.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_play.setObjectName("Frame_play")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.Frame_play)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setSpacing(6)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.PushButton_play = QtWidgets.QPushButton(self.Frame_play)
        self.PushButton_play.setObjectName("PushButton_play")
        self.horizontalLayout_9.addWidget(self.PushButton_play)
        self.PushButton_pause = QtWidgets.QPushButton(self.Frame_play)
        self.PushButton_pause.setObjectName("PushButton_pause")
        self.horizontalLayout_9.addWidget(self.PushButton_pause)
        self.verticalLayout_3.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setSpacing(6)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.Label_slider = QtWidgets.QLabel(self.Frame_play)
        self.Label_slider.setObjectName("Label_slider")
        self.horizontalLayout_7.addWidget(self.Label_slider)
        self.DoubleSpinBox_secPerFrame = QtWidgets.QDoubleSpinBox(self.Frame_play)
        self.DoubleSpinBox_secPerFrame.setMinimum(0.2)
        self.DoubleSpinBox_secPerFrame.setMaximum(5.0)
        self.DoubleSpinBox_secPerFrame.setSingleStep(0.2)
        self.DoubleSpinBox_secPerFrame.setProperty("value", 1.0)
        self.DoubleSpinBox_secPerFrame.setObjectName("DoubleSpinBox_secPerFrame")
        self.horizontalLayout_7.addWidget(self.DoubleSpinBox_secPerFrame)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.RadioButton_play_automaticStop = QtWidgets.QRadioButton(self.Frame_play)
        self.RadioButton_play_automaticStop.setChecked(True)
        self.RadioButton_play_automaticStop.setAutoExclusive(False)
        self.RadioButton_play_automaticStop.setObjectName("RadioButton_play_automaticStop")
        self.verticalLayout.addWidget(self.RadioButton_play_automaticStop)
        self.RadioButton_automaticSaving = QtWidgets.QRadioButton(self.Frame_play)
        self.RadioButton_automaticSaving.setAutoExclusive(False)
        self.RadioButton_automaticSaving.setObjectName("RadioButton_automaticSaving")
        self.verticalLayout.addWidget(self.RadioButton_automaticSaving)
        self.verticalLayout_4.addWidget(self.Frame_play)
        spacerItem4 = QtWidgets.QSpacerItem(20, 487, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout_4.addItem(spacerItem4)
        self.gridLayout_6.addWidget(self.Widget_play, 3, 1, 1, 1)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setContentsMargins(-1, -1, 14, -1)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.Label_showResult_title = QtWidgets.QLabel(self.centralwidget)
        self.Label_showResult_title.setEnabled(False)
        self.Label_showResult_title.setObjectName("Label_showResult_title")
        self.verticalLayout_7.addWidget(self.Label_showResult_title, 0, QtCore.Qt.AlignHCenter)
        self.Label_showResult = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Label_showResult.sizePolicy().hasHeightForWidth())
        self.Label_showResult.setSizePolicy(sizePolicy)
        self.Label_showResult.setMinimumSize(QtCore.QSize(0, 0))
        self.Label_showResult.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.Label_showResult.setText("")
        self.Label_showResult.setObjectName("Label_showResult")
        self.verticalLayout_7.addWidget(self.Label_showResult)
        self.Widget_buttons_result = QtWidgets.QWidget(self.centralwidget)
        self.Widget_buttons_result.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Widget_buttons_result.sizePolicy().hasHeightForWidth())
        self.Widget_buttons_result.setSizePolicy(sizePolicy)
        self.Widget_buttons_result.setObjectName("Widget_buttons_result")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.Widget_buttons_result)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_threshold = QtWidgets.QLabel(self.Widget_buttons_result)
        self.label_threshold.setEnabled(False)
        self.label_threshold.setObjectName("label_threshold")
        self.horizontalLayout_8.addWidget(self.label_threshold)
        self.doubleSpinBox_threshold = QtWidgets.QDoubleSpinBox(self.Widget_buttons_result)
        self.doubleSpinBox_threshold.setEnabled(False)
        self.doubleSpinBox_threshold.setSingleStep(0.01)
        self.doubleSpinBox_threshold.setProperty("value", 0.13)
        self.doubleSpinBox_threshold.setObjectName("doubleSpinBox_threshold")
        self.horizontalLayout_8.addWidget(self.doubleSpinBox_threshold)
        self.PushButton_reloadResult = QtWidgets.QPushButton(self.Widget_buttons_result)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PushButton_reloadResult.sizePolicy().hasHeightForWidth())
        self.PushButton_reloadResult.setSizePolicy(sizePolicy)
        self.PushButton_reloadResult.setObjectName("PushButton_reloadResult")
        self.horizontalLayout_8.addWidget(self.PushButton_reloadResult)
        self.PushButton_saveResult = QtWidgets.QPushButton(self.Widget_buttons_result)
        self.PushButton_saveResult.setObjectName("PushButton_saveResult")
        self.horizontalLayout_8.addWidget(self.PushButton_saveResult)
        self.PushButton_clearResult = QtWidgets.QPushButton(self.Widget_buttons_result)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PushButton_clearResult.sizePolicy().hasHeightForWidth())
        self.PushButton_clearResult.setSizePolicy(sizePolicy)
        self.PushButton_clearResult.setObjectName("PushButton_clearResult")
        self.horizontalLayout_8.addWidget(self.PushButton_clearResult)
        self.verticalLayout_7.addWidget(self.Widget_buttons_result, 0, QtCore.Qt.AlignHCenter)
        self.Frame_save = QtWidgets.QFrame(self.centralwidget)
        self.Frame_save.setEnabled(False)
        self.Frame_save.setStyleSheet("border-color: rgb(0, 0, 0);")
        self.Frame_save.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_save.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_save.setObjectName("Frame_save")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.Frame_save)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.PushButton_save_changeFolder = QtWidgets.QPushButton(self.Frame_save)
        self.PushButton_save_changeFolder.setObjectName("PushButton_save_changeFolder")
        self.gridLayout_4.addWidget(self.PushButton_save_changeFolder, 1, 0, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem5, 1, 1, 1, 1)
        self.Label_save_label = QtWidgets.QLabel(self.Frame_save)
        self.Label_save_label.setObjectName("Label_save_label")
        self.gridLayout_4.addWidget(self.Label_save_label, 0, 0, 1, 1)
        self.Label_save_currentFolder = QtWidgets.QLabel(self.Frame_save)
        self.Label_save_currentFolder.setText("")
        self.Label_save_currentFolder.setWordWrap(True)
        self.Label_save_currentFolder.setObjectName("Label_save_currentFolder")
        self.gridLayout_4.addWidget(self.Label_save_currentFolder, 0, 1, 1, 1)
        self.verticalLayout_7.addWidget(self.Frame_save)
        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setMinimumSize(QtCore.QSize(0, 120))
        self.widget_2.setMaximumSize(QtCore.QSize(16777215, 120))
        self.widget_2.setStyleSheet("")
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        spacerItem6 = QtWidgets.QSpacerItem(20, 100, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout_5.addItem(spacerItem6)
        self.Label_info_inbetween = QtWidgets.QLabel(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Label_info_inbetween.sizePolicy().hasHeightForWidth())
        self.Label_info_inbetween.setSizePolicy(sizePolicy)
        self.Label_info_inbetween.setMinimumSize(QtCore.QSize(0, 0))
        self.Label_info_inbetween.setText("")
        self.Label_info_inbetween.setWordWrap(True)
        self.Label_info_inbetween.setObjectName("Label_info_inbetween")
        self.verticalLayout_5.addWidget(self.Label_info_inbetween)
        self.verticalLayout_7.addWidget(self.widget_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(0, -1, 0, -1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_7.addLayout(self.horizontalLayout)
        self.gridLayout_6.addLayout(self.verticalLayout_7, 3, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Similarity"))
        self.Label_video.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">1. Choose Video</span></p></body></html>"))
        self.PushButton_confirmVideo.setText(_translate("MainWindow", "Load Video"))
        self.Label_chooseReprType.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">2. Choose Representation Type</span></p></body></html>"))
        self.ComboBox_chooseReprType.setItemText(0, _translate("MainWindow", "Postures"))
        self.ComboBox_chooseReprType.setItemText(1, _translate("MainWindow", "Sequences"))
        self.Label_info_title.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Video Information</span></p></body></html>"))
        self.Label_frame.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">3. Choose Frame and Region of Interest (ROI)</span></p></body></html>"))
        self.PushButton_goLeftLeft.setText(_translate("MainWindow", "-5"))
        self.PushButton_goLeft.setText(_translate("MainWindow", "-1"))
        self.PushButton_reloadFrame.setText(_translate("MainWindow", "Load"))
        self.PushButton_clearFrame.setText(_translate("MainWindow", "Clear"))
        self.PushButton_goRight.setText(_translate("MainWindow", "+1"))
        self.PushButton_goRightRight.setText(_translate("MainWindow", "+5"))
        self.RadioButton_adjustBB.setText(_translate("MainWindow", "Adjust Region of Interest"))
        self.Label_BBSize_desc.setText(_translate("MainWindow", "Size:"))
        self.Label_BBSize_x.setText(_translate("MainWindow", "x"))
        self.Label_BBSize_y.setText(_translate("MainWindow", "y"))
        self.PushButton_play.setText(_translate("MainWindow", "Play"))
        self.PushButton_pause.setText(_translate("MainWindow", "Pause"))
        self.Label_slider.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">#Frames per Second</p></body></html>"))
        self.RadioButton_play_automaticStop.setText(_translate("MainWindow", "Stop at the End of Video"))
        self.RadioButton_automaticSaving.setText(_translate("MainWindow", "Save Results Automatically"))
        self.Label_showResult_title.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Result</p></body></html>"))
        self.label_threshold.setText(_translate("MainWindow", "Threshold"))
        self.PushButton_reloadResult.setText(_translate("MainWindow", "Show"))
        self.PushButton_saveResult.setText(_translate("MainWindow", "Save"))
        self.PushButton_clearResult.setText(_translate("MainWindow", "Clear"))
        self.PushButton_save_changeFolder.setText(_translate("MainWindow", "Change Directory"))
        self.Label_save_label.setText(_translate("MainWindow", " Save Directory:"))


import resource_rc