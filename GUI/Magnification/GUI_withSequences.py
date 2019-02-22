# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Code for Python 2.7, Does not need GPU, runs on CPU
# Written by Uta Buechler
# E-Mail: utabuechler@web.de
# Github: utabuechler
# Homepage: utabuechler.github.io
# --------------------------------------------------------

#---------------------------------------------------------
#TO DO:
#Less Important:
# - check why humans from the same video are directly on a circle -> adjust parameters for umap?
# - better bar for moving between sequences (left side)
# - choosing ROI also for sequences
# - compute embedding in a QThread
# - don't show the centroids, but the mean of the postures belonging to a specific cluster using the autoencoder
# - bigger/smaller buttons
# - low-dim embedding with TSNE
# - give the possibility to also run it on the GPU
#---------------------------------------------------------

# pyuic5 GUI/Magnification/design3.ui -o GUI/Magnification/design3.py

import sys,os
import numpy as np
import random
from PIL import Image
import pandas as pd
from matplotlib import rcParams
from matplotlib import patches
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle
from functools import partial
try:
    import umap
    IS_UMAP=True
except ImportError:
    IS_UMAP=False
    print('!!UMAP not installed, tSNE will be used!!')

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QSizePolicy, QMessageBox, QStyle
from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont, QMouseEvent, QImage, QIcon
from PyQt5.QtCore import Qt,QRect, QCoreApplication, QSize
from PyQt5 import QtTest

import torch
from torch.autograd import Variable
#os.environ["CUDA_VISIBLE_DEVICES"]=""#make sure that it is run on the cpu
sys.path.append("features/pytorch")
from model import CaffeNet
sys.path.append('')#so that we can load modules from the current directory
from utils import load_table, load_features, transform_batch, draw_border, fig2data, load_image

sys.path.append('./GUI/Magnification/')
import design3 as design

sys.path.append('./magnification/')
from Generator import Generator, find_differences, find_differences_cc


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-cf", "--config",type=str,default='config_pytorch_human',############################################################
                    help="Define config file")
parser.add_argument("-g", "--gpu",type=int,default='0',
                    help="GPU id")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
import importlib

cfg = importlib.import_module(args.config)

encod = 'utf-8'

def save_umap(umap,file):
    for attr in ["_tree_init", "_search", "_random_init"]:
        if hasattr(umap, attr):
            delattr(umap, attr)
    
    pickle.dump(umap,file,pickle.HIGHEST_PROTOCOL)

def load_umap(file):
    umap = pickle.load(file)
    from umap.nndescent import make_initialisations, make_initialized_nnd_search
    umap._random_init, umap._tree_init = make_initialisations(
        umap._distance_func, umap._dist_args
    )
    umap._search = make_initialized_nnd_search(
        umap._distance_func, umap._dist_args
    )
    return umap

class NNInterface(QMainWindow, design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(NNInterface, self).__init__(parent)
        print('Initialize Interface...')
        self.setupUi(self)
        #self.centralwidget.showMaximized()
        self.load_detection_table()
        self.fill_comboBox()
        self.nr_embedding_dataset = 5000
        self.nr_embedding_dataset_seqs = 5000
        
        self.init_network()
        self.get_all_feats()#collect the features of all crops
        
        self.Label_info_project.setText('Project: %s \n'%(cfg.project))
        
        #self.PushButton_confirmDataset.clicked.connect(self.load_dataset)
        self.PushButton_confirmVideo.clicked.connect(self.load_video) #When the button is pressed execute show_frames function
        self.ComboBox_chooseReprType.activated.connect(self.choose_repr_type)
        
        self.PushButton_clearFrame.clicked.connect(self.clear_all_labels)
        self.PushButton_reloadFrame.clicked.connect(self.draw_frame)
        self.PushButton_goRight.clicked.connect(self.go_right)
        self.PushButton_goLeft.clicked.connect(self.go_left)
        self.PushButton_goRightRight.clicked.connect(self.go_rightright)
        self.PushButton_goLeftLeft.clicked.connect(self.go_leftleft)
        #self.PushButton_bigger.clicked.connect(self.bigger)
        #self.PushButton_smaller.clicked.connect(self.smaller)
        
        self.PushButton_play.clicked.connect(self.play)
        self.PushButton_pause.clicked.connect(self.pause)
        #self.RadioButton_play.clicked.connect(self.play)
        self.RadioButton_adjustBB.clicked.connect(self.activate_adjust_BB)
        
        #self.PushButton_compute.clicked.connect(self.compute)
        self.SpinBox_BBSize_x.valueChanged.connect(self.adjust_BB_directly)
        self.SpinBox_BBSize_y.valueChanged.connect(self.adjust_BB_directly)
        #self.PushButton_computeNN.clicked.connect(self.show_NN)
        #self.PushButton_computeTSNE.clicked.connect(self.show_TSNE)
        
        self.PushButton_reloadResult.clicked.connect(self.compute)
        self.PushButton_clearResult.clicked.connect(self.clear_result_labels)
        self.PushButton_saveResult.clicked.connect(partial(self.save_result,True))
        self.PushButton_save_changeFolder.clicked.connect(self.change_dir)
        
        self.n_mean_nn = 100
        #self.adjust_BB = False
        self.BB_size = [100,100]
        self.default_BB = True
        self.last_result='NN'
        self.save_dir = cfg.results_path+'/GUI'
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.nr_clusters = 5
        self.repr_type = self.ComboBox_chooseReprType.currentIndex()
        self.plot_centroids = True
        
        self.PushButton_goRightRight.setToolTip("Move 5 frames forward")
        self.PushButton_goRight.setToolTip("Move 1 frame forward")
        self.PushButton_goLeftLeft.setToolTip("Move 5 frames backward")
        self.PushButton_goLeft.setToolTip("Move 1 frame backward")
        self.PushButton_play.setToolTip("Show the video and results automatically")
    
    def load_detection_table(self):
        detections = load_table(cfg.detection_file,asDict=False)
        det_time = np.array(detections['time'])
        uni_videos = np.unique(detections['videos'].values)
        uni_videos = np.array([v for v in uni_videos if os.path.isdir(cfg.crops_path+v)])
        self.videos = uni_videos#np.array([vid.encode(encod) for vid in uni_videos])
        self.video_time  =np.array([det_time[detections['videos']==v][0] for v in uni_videos]).astype(int)
        #self.video_cohort=np.array([detections[self.detections['videos']==v][0] for v in uni_videos]).astype(int)
    
    def choose_repr_type(self):
        if self.ComboBox_chooseReprType.currentIndex() != self.repr_type:
            self.plot_centroids = True
            _=plt.close()#close all plots
            self.repr_type = self.ComboBox_chooseReprType.currentIndex()
            self.clear_result_labels()
            self.Label_frame.setStyleSheet("font-size:12pt; font-weight:600;")
            if self.repr_type==1:#sequences
                self.Frame_adjustBB.setEnabled(False)
            
            if self.repr_type==0:#Postures
                self.Frame_adjustBB.setEnabled(True)
                self.Label_frame.setText('3. Choose Frame and Area of Interest (ROI)')
                self.nr_embedding_dataset = min(self.nr_embedding_dataset,self.feats['features'].shape[0])
                
                #update current_idx since the idx for postures mean something else than for sequences
                self.current_idx = np.where(self.current_frame_name==np.array(self.crop_files))[0][0]
                self.current_frame_name = self.crop_files[self.current_idx]
                self.current_frame_number = int(''.join(c for c in self.current_frame_name if c.isdigit()))
                
                #set default value for number of nearest neighbors
                self.k = 12
            else:
                self.Frame_adjustBB.setEnabled(False)
                self.Label_frame.setText('3. Choose Middle Frame of Sequence')
                #self.nr_embedding_dataset = min(self.nr_embedding_dataset,self.feats_seqs['features'].shape[0])
                self.nr_embedding_dataset = min(self.nr_embedding_dataset,self.feats['features'].shape[0])
                
                #update current_idx since the idx for postures mean something else than for sequences
                self.current_idx = (np.abs(self.current_frame_number-np.array(self.middle_frames))).argmin()#find closest to this frame number
                self.current_frame_name = '%06d.jpg'%(self.middle_frames[self.current_idx])
                self.current_frame_number = self.middle_frames[self.current_idx]
                
                #set default value for number of nearest neighbors
                self.k = 5
            
            self.draw_frame()
    
    def clear_result_labels(self):
        self.Label_showResult.clear()
        self.clear_info_inbetween()
        
    def clear_all_labels(self):
        self.Label_showFrame.clear()
        self.Label_showResult.clear()
        self.clear_info_inbetween()
    
    def change_dir(self):
        fileDialog = QFileDialog()
        self.save_dir = str(fileDialog.getExistingDirectory(directory=self.save_dir))
        self.Label_save_currentFolder.setText(self.save_dir)
        
    def adjust_BB_directly(self):
        self.BB_size = [self.SpinBox_BBSize_x.value(),self.SpinBox_BBSize_y.value()]
        self.draw_frame(pos=[self.current_BB_pos[0]-self.BB_size[0]/2,
                            self.current_BB_pos[1]-self.BB_size[1]/2,
                            self.current_BB_pos[0]+self.BB_size[0]/2,
                            self.current_BB_pos[1]+self.BB_size[1]/2])
        
        #delete feats_ROI since the user might change it
        if hasattr(self,'feats_ROI'): delattr(self,'feats_ROI')
    
    def pause(self):
        self.goOn = False
        if self.repr_type==0:#only for postures possible
            self.Frame_adjustBB.setEnabled(True)
            self.RadioButton_adjustBB.setEnabled(True)
        
        if self.RadioButton_adjustBB.isChecked():
            self.Widget_changeBBSize.setEnabled(True)
        else:
            self.Widget_changeBBSize.setEnabled(False)
    
    def play(self):
        self.default_BB = True
        self.goOn = True
        self.RadioButton_adjustBB.setChecked(False)
        self.Frame_adjustBB.setEnabled(False)
        
        self.clear_info_inbetween()
        self.setFixedSize(self.size())
        
        while self.goOn:
            t1 = time.time()
            self.go_right()
            self.compute()
            if self.RadioButton_automaticSaving.isChecked():
                self.save_result(showInfo = False)#don't show it every single time
            
            t2 = time.time()
            t = t2-t1
            
            QtTest.QTest.qWait((float(1)/self.DoubleSpinBox_secPerFrame.value()-t)*1000)
            
            #stop if we are at the end of the video, if the user wants that
            if self.RadioButton_play_automaticStop.isChecked() and self.current_idx == len(self.crop_files)-1:
                self.goOn = False
            
            QCoreApplication.processEvents()
        
        self.setMaximumSize(16777215,16777215)
        self.setMinimumSize(0,0)
        
        if self.RadioButton_automaticSaving.isChecked():
                self.save_result()#now show the information when the saving is stopped
    
    def activate_adjust_BB(self):
        if self.RadioButton_adjustBB.isChecked():
            self.default_BB = False#set it to false, so that we know that we have to compute the features from scratch
            self.Widget_changeBBSize.setEnabled(True)
            
            #delete feats_ROI since the user will probably change it now
            if hasattr(self,'feats_ROI'): delattr(self,'feats_ROI')
        else:
            self.clear_info_inbetween()
            self.default_BB = True
            self.draw_frame()
            self.Widget_changeBBSize.setEnabled(False)
    
    def mousePressEvent(self, event):
        if self.childAt(event.pos()).objectName()=='Label_showFrame' and self.RadioButton_adjustBB.isChecked():
            if self.last_result=='NN':
                self.Label_showResult.clear()
            #get the chosen size of the bounding box
            self.BB_size = [self.SpinBox_BBSize_x.value(),self.SpinBox_BBSize_y.value()]
            #print self.Label_showFrame.mapFromParent(event.pos())
            click = self.Label_showFrame.mapFrom(self.centralwidget,event.pos())#map from main window
            click_adjusted = [click.x()-self.pixmap_coords_start[0],click.y()-self.pixmap_coords_start[1]]
            #get the position on the original image
            ratio =  [float(self.frame_orig_size[0])/self.frame_new_size[0],
                      float(self.frame_orig_size[1])/self.frame_new_size[1]]
            #since the image is in the middle (horizontally) we need to know where it starts
            self.current_BB_pos = [ratio[0]*click_adjusted[0],ratio[1]*click_adjusted[1]]
            
            self.draw_frame(pos=[self.current_BB_pos[0]-self.BB_size[0]/2,
                                self.current_BB_pos[1]-self.BB_size[1]/2,
                                self.current_BB_pos[0]+self.BB_size[0]/2,
                                self.current_BB_pos[1]+self.BB_size[1]/2])
            
            #delete feats_ROI since the user changed it already
            if hasattr(self,'feats_ROI'): delattr(self,'feats_ROI')
    
    def fill_comboBox(self):
        #check if all the folders exist
        keep,vids_missing = [],''
        for i,v in enumerate(self.videos):
            if os.path.isdir(cfg.crops_path+v):
                keep.append(i)
            else:
                vids_missing+=v+'\n'
        
        if len(keep)!=len(self.videos):
            msgBox = QMessageBox()
            msgBox.setWindowTitle("Video folders")
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Not all videos stated in the 'detection' file are found.\
                \n # Found Videos: %i/%i"%(len(keep),len(self.videos)))
            msgBox.setInformativeText("A listing of the videos is displayed after pressing 'Show Details...'.\
                \n \n If the videos should be available in the given location (%s), please press 'Cancel' and make sure that they are available before executing the interface again.\
                \n \n If this is not a problem, press 'Ok' and the interface will continue."%(cfg.crops_path))
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msgBox.setDetailedText(vids_missing)
            #abortButton = msg.addButton(QMessageBox.Abort)
            #abortButton.setIcon(QStyle.standardIcon(QStyle.SP_BrowserStop))
            res = msgBox.exec_()
            if res==QMessageBox.Cancel:
                sys.exit("Interface aborted because of missing videos.")
        
        self.videos = self.videos[keep]
        self.ComboBox_chooseVideo.addItem('Random')
        for i,v in enumerate(self.videos):
            if v[0]=='/': v=v[1:]
            self.ComboBox_chooseVideo.addItem('%d) %s'%(i+1,v))
    
    def save_result(self,showInfo = True):
        if not ((hasattr(self, 'plot_NN') and self.last_result == 'NN') or (hasattr(self, 'plot_embedding') and self.last_result == 'embedding')) or not self.Label_showResult.pixmap():
            self.set_info_inbetween('No results produced so far!')
            QtTest.QTest.qWait((2)*1000)
            self.clear_info_inbetween()
            return
        
        if showInfo:
            self.set_info_inbetween('Saving...')
            QtTest.QTest.qWait(0.1*1000)
        
        vid = self.videos[self.chosenVideoIdx]#.replace('/','_').replace(' ','_')
        if vid[0]=='/': vid = vid[1:]
        folder = self.save_dir+'/'+vid
        if not os.path.isdir(folder):
            os.makedirs(folder)
        
        if self.last_result=='NN':#Method Nearest Neighbor
            if self.repr_type==0:#postures
                filename = '%s/%s_frame%06d'%(folder,self.last_result,self.current_frame_number)
                #first draw the plot with query
                self.show_NN(saveNN=True)
                #then save it
                self.plot_NN.savefig(filename+'.png',bbox_inches='tight')
                
                #then save meta data in a csv file
                NN_info = pd.DataFrame(index=None, columns=['No. NN','video id','video directory','frame','x1','y1','x2','y2','distance to query'])
                
                for i,(v_id,vn,f,c,d) in enumerate(self.meta_data):
                    NN_info = NN_info.append({'No. NN': i+1, 'video id': v_id+1,
                                            'video directory': vn,'frame': f,
                                            'x1':c[0], 'y1':c[1], 'x2':c[2],'y2':c[3],
                                            'distance to query': d}, ignore_index=True)
                
                NN_info.to_csv(filename+'.csv',index=False)
            
            else:#sequences
                #first save image
                filename = '%s/%s_seq_middleFrame%06d'%(folder,self.last_result,self.current_frame_number)
                self.show_NN_seqs(saveNN=True)
                self.plot_NN.savefig(filename+'.png',bbox_inches='tight')
                
                #then save meta data
                NN_info = pd.DataFrame(index=None, columns=['No. NN','position in sequence','video id','video directory','frame','x1','y1','x2','y2','distance to query'])#'sequence id'
                for nn,md in enumerate(self.meta_data):
                    for (i,j,v_id,vn,f,c,d) in md:
                        NN_info = NN_info.append({'No. NN': nn+1,
                                                #'sequence id': i+1,
                                                'position in sequence': j+1,
                                                'video id': v_id+1,
                                                'video directory': vn,'frame': f,
                                                'x1':c[0], 'y1':c[1], 'x2':c[2],'y2':c[3],
                                                'distance to query': d}, ignore_index=True)
                    
                NN_info.to_csv(filename+'.csv',index=False)
                
            if showInfo:
                self.set_info_inbetween('Plot and Meta Data saved as ".png" and ".csv", respectively under %s'%(folder))
        
        elif self.last_result=='embedding':#method low-dim embedding
            
            if self.repr_type==0:#postures
                #1. save the plot
                filename_plot = '%s/embedding_dataset_%i_kmeans%04d_frame%06d.png'%(folder,self.nr_embedding_dataset,self.nr_clusters,self.current_frame_number)
                self.plot_embedding.savefig(filename_plot,bbox_inches='tight')
                
                #2. save the cluster information with all necessary information
                #3. save the cluster labels of all detections of the current video and also of the frame if custom bounding box
                if hasattr(self,'centroids'):
                    filename_clust_info = '%s/embedding_dataset_%i_kmeans%04d_clusters_centroids_info.csv'%(self.save_dir,self.nr_embedding_dataset,self.nr_clusters)
                    #####2.
                    if not os.path.isfile(filename_clust_info):
                        clust_info = pd.DataFrame(index=None, columns=['cluster id','video id','video directory','frame','x1','y1','x2','y2'])
                        
                        
                        for l,(v_id,vn,f,c) in enumerate(self.meta_data):
                            clust_info = clust_info.append({'cluster id': l+1,
                                                            'video id': v_id+1,
                                                            'video directory': vn,
                                                            'frame': f,
                                                            'x1':c[0], 'y1':c[1], 'x2':c[2],'y2':c[3]}, ignore_index=True)
                        
                        clust_info = clust_info.set_index('cluster id')
                        clust_info.to_csv(filename_clust_info)
                    ##### End 2.
                    
                    ######3. Only if this file is new save all the default detections, if this file already exists and it is a custom bounding box, include it to the file
                    filename_clust_info_vid = '%s/embedding_dataset_%i_kmeans%04d_clusters_info.csv'%(folder,self.nr_embedding_dataset,self.nr_clusters)
                    
                    if os.path.isfile(filename_clust_info_vid):
                        clust_info_vid = load_table(filename_clust_info_vid,asDict=False)
                    else:
                        clust_info_vid = pd.DataFrame(index=None, columns=['cluster id','video id',
                                                                                'video directory','frame',
                                                                                'x1','y1','x2','y2','default roi'])
                        
                        #save default detections
                        for i,l in enumerate(self.low_dim_points_vid_clusterlabel):
                            f = self.current_feats['frames'][i]
                            c = self.current_feats['coords'][i]
                            clust_info_vid = clust_info_vid.append({'cluster id': l+1,
                                                                    'video id': self.chosenVideoIdx+1,
                                                                    'video directory': vid,
                                                                    'frame': f,
                                                                    'x1':c[0], 'y1':c[1], 'x2':c[2],'y2':c[3],
                                                                    'default roi': 'Yes'}, ignore_index=True)
                            
                    
                    #include the current frame if custom ROI
                    if not self.default_BB:
                        c = [self.current_BB_pos[0]-self.BB_size[0]/2,
                                self.current_BB_pos[1]-self.BB_size[1]/2,
                                self.current_BB_pos[0]+self.BB_size[0]/2,
                                self.current_BB_pos[1]+self.BB_size[1]/2]
                        
                        clust_info_vid = clust_info_vid.append({'cluster id': self.low_dim_points_ROI_clusterlabel+1,
                                                                    'video id': self.chosenVideoIdx+1,
                                                                    'video directory': vid,
                                                                    'frame': self.current_frame_number,
                                                                    'x1':c[0], 'y1':c[1], 'x2':c[2],'y2':c[3],
                                                                    'default roi': 'No'}, ignore_index=True)
                    
                    #save
                    #clust_info_vid = clust_info_vid.set_index('frame')
                    clust_info_vid.to_csv(filename_clust_info_vid, index=False)
                        
                    #####End 3.
                    
                    if showInfo:
                        self.set_info_inbetween(
                            'Saved the following information: \n - Centroids of clusters under %s \n - Plot under %s \n - Video cluster information %s'
                            %(filename_clust_info,filename_plot,filename_clust_info_vid))
                        
                elif showInfo:
                    self.set_info_inbetween('Result saved as ".png" under %s'%(filename))
                    
            else:#sequences
                
                #1. save the plot
                filename_plot = '%s/embedding_dataset_%i_kmeans%04d_seq_middleFrame%06d.png'%(folder,self.nr_embedding_dataset_seqs,self.nr_clusters,self.current_frame_number)
                self.plot_embedding.savefig(filename_plot,bbox_inches='tight')
                
                #2. save the cluster information with all necessary information
                #3. save the cluster labels of all detections of the current video and also of the frame if custom bounding box
                if hasattr(self,'centroids_seqs'):
                    filename_clust_info = '%s/embedding_dataset_%i_kmeans%04d_seq_clusters_centroids_info.csv'%(self.save_dir,self.nr_embedding_dataset_seqs,self.nr_clusters)
                    #####2.
                    if not os.path.isfile(filename_clust_info):
                        clust_info = pd.DataFrame(index=None, columns=['cluster id','position in sequence','video id','video directory','frame','x1','y1','x2','y2'])
                        
                        for l,md in enumerate(self.meta_data):
                            for (i,j,vid_id,vid,f,c) in md:
                                clust_info = clust_info.append({'cluster id': l+1,
                                                                'position in sequence': j+1,
                                                                'video id': vid_id+1,
                                                                'video directory': vid,
                                                                'frame': f,
                                                                'x1':c[0], 'y1':c[1], 'x2':c[2],'y2':c[3]}, ignore_index=True)
                        
                        clust_info = clust_info.set_index('cluster id')
                        clust_info.to_csv(filename_clust_info)
                    ##### End 2.
                    
                    ######3.
                    filename_clust_info_vid = '%s/embedding_dataset_%i_kmeans%04d_seq_clusters_info.csv'%(folder,self.nr_embedding_dataset_seqs,self.nr_clusters)
                    
                    #if os.path.isfile(filename_clust_info_vid):
                    #    clust_info_vid = load_table(filename_clust_info_vid,asDict=False)
                    #else:
                    clust_info_vid = pd.DataFrame(index=None, columns=['seq id','position in sequence','cluster id','video id',
                                                                            'video directory','frame',
                                                                            'x1','y1','x2','y2','default roi'])
                        
                    #save default detections
                    for i,l in enumerate(self.low_dim_points_vid_clusterlabel_seqs):
                        frames = self.current_feats_seqs['frames'][i]
                        coords = self.current_feats_seqs['coords'][i]
                        
                        for j,(f,c) in enumerate(zip(frames,coords)):
                            clust_info_vid = clust_info_vid.append({'seq id': i+1,
                                                                    'position in sequence': j+1,
                                                                    'cluster id': l+1,
                                                                    'video id': self.chosenVideoIdx+1,
                                                                    'video directory': vid,
                                                                    'frame': f,
                                                                    'x1':c[0], 'y1':c[1], 'x2':c[2],'y2':c[3],
                                                                    'default roi': 'Yes'}, ignore_index=True)
                    #save
                    clust_info_vid.to_csv(filename_clust_info_vid, index=False)
                        
                    #####End 3.
                    if showInfo:
                        self.set_info_inbetween(
                            'Saved the following information: \n - Centroids of clusters under %s \n - Plot under %s \n - Video cluster information %s'
                            %(filename_clust_info,filename_plot,filename_clust_info_vid))
                        
                elif showInfo:
                    self.set_info_inbetween('Result saved as ".png" under %s'%(filename))
    
    
    def go_right(self):
        self.clear_info_inbetween()
        if self.last_result=='NN':
            self.Label_showResult.clear()
        
        self.RadioButton_adjustBB.setChecked(False)
        if self.repr_type==0:#postures
            if (self.current_idx+1)<len(self.crop_files):
                self.current_idx += 1
                self.current_frame_name = self.crop_files[self.current_idx]
                self.current_frame_number = int(''.join(c for c in self.current_frame_name if c.isdigit()))
                
                self.draw_frame()
            else:
                self.set_info_inbetween('Reached the end of the video')
                
        else:#sequences
            if (self.current_idx+1)<len(self.middle_frames):
                self.current_idx += 1
                self.current_frame_name = '%06d.jpg'%(self.middle_frames[self.current_idx])
                self.current_frame_number = self.middle_frames[self.current_idx]
                
                self.draw_frame()
            else:
                self.set_info_inbetween('Reached the end of the video')
    
    def go_left(self):
        self.clear_info_inbetween()
        if self.last_result=='NN':
            self.Label_showResult.clear()
        
        self.RadioButton_adjustBB.setChecked(False)
        
        if (self.current_idx-1)>=0:
            self.current_idx -= 1
            
            if self.repr_type==0:#Postures
                self.current_frame_name = self.crop_files[self.current_idx]#.encode(encod)
                self.current_frame_number = int(''.join(c for c in self.current_frame_name if c.isdigit()))
            else:#Sequences
                self.current_frame_name = '%06d.jpg'%(self.middle_frames[self.current_idx])
                self.current_frame_number = self.middle_frames[self.current_idx]
            
            self.draw_frame()
        else:
            self.set_info_inbetween('Reached the beginning of the video')
    
    def go_rightright(self):
        self.clear_info_inbetween()
        if self.last_result=='NN':
            self.Label_showResult.clear()
        
        #self.Label_info_inbetween.setText(' ')
        #self.Label_info_inbetween.setStyleSheet("")
        self.RadioButton_adjustBB.setChecked(False)
        
        if self.repr_type==0:#postures
            if self.current_idx!=(len(self.crop_files)-1):
                add = min(5,len(self.crop_files)-1-self.current_idx)
                self.current_idx += add
                self.current_frame_name = self.crop_files[self.current_idx]
                self.current_frame_number = int(''.join(c for c in self.current_frame_name if c.isdigit()))
                
                self.draw_frame()
            else:
                self.set_info_inbetween('Reached the end of the video')
        else:#sequences
            if self.current_idx!=(len(self.middle_frames)-1):
                add = min(5,len(self.crop_files)-1-self.current_idx)
                self.current_idx += add
                self.current_frame_name = '%06d.jpg'%(self.middle_frames[self.current_idx])
                self.current_frame_number = self.middle_frames[self.current_idx]
                
                self.draw_frame()
            else:
                self.set_info_inbetween('Reached the end of the video')
    
    def go_leftleft(self):
        self.clear_info_inbetween()
        if self.last_result=='NN':
            self.Label_showResult.clear()
        
        self.RadioButton_adjustBB.setChecked(False)
        
        if self.current_idx!=0:
            subtract = min(self.current_idx,5)
            self.current_idx -= subtract
            
            if self.repr_type==0:#Postures
                self.current_frame_name = self.crop_files[self.current_idx]
                self.current_frame_number = int(''.join(c for c in self.current_frame_name if c.isdigit()))
            else:#sequences
                self.current_frame_name = '%06d.jpg'%(self.middle_frames[self.current_idx])
                self.current_frame_number = self.middle_frames[self.current_idx]
            
            self.draw_frame()
        else:
            self.set_info_inbetween('Reached the beginning of the video')
    
    def set_info_inbetween(self,text):
        self.Label_showResult.setFixedSize(self.Label_showResult.size())#make sure that this label doesn't change size when 'Label_info_inbetween' is used
        self.Label_info_inbetween.setText(text)
        self.Label_info_inbetween.setStyleSheet("QLabel { background-color : blue; color : white; }")
        self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
        #set it back
        #self.Label_showResult.setSizePolicy(QSizePolicy policy(QSizePolicy::Expanding, QSizePolicy::Expanding))
        self.Label_showResult.setMaximumSize(16777215,16777215)
        self.Label_showResult.setMinimumSize(0,0)
    
    def clear_info_inbetween(self):
        self.Label_info_inbetween.clear()
        self.Label_info_inbetween.setStyleSheet("")
    
    def get_middle_frames(self):
        middle_frames = []
        for i,f in enumerate(self.current_feats_seqs['frames']):
            middle_frames.append(f[int(len(f)/2)])
            
        self.middle_frames = middle_frames
    
    def update_coords(self,c,size):
        if c[2]==-1: c[2] = size[0]
        if c[3]==-1: c[3] = size[1]
        return c
    
    def load_video(self):
        
        self.clear_all_labels()
        
        if hasattr(self,'low_dim_points_vid'): delattr(self,'low_dim_points_vid')#becaues the embedding points need to be computed again for the new video
        if hasattr(self,'low_dim_points_vid_seqs'): delattr(self,'low_dim_points_vid_seqs')
        
        #first get the chosen video
        self.chosenVideoIdx = self.ComboBox_chooseVideo.currentIndex() 
        if self.chosenVideoIdx==0:
            self.chosenVideoIdx = random.sample(range(len(self.videos)),1)[0]
            self.ComboBox_chooseVideo.setCurrentIndex(self.chosenVideoIdx+1)
        else:
            self.chosenVideoIdx-=1#because "Random" is the first position
        
        self.chosenVideo = self.videos[self.chosenVideoIdx]
        if os.path.isdir(cfg.frames_path+self.chosenVideo+'/deinterlaced'):
            self.chosenVideo = self.chosenVideo+'/deinterlaced'
        
        #save the files available in the folder
        crop_files = sorted(os.listdir(cfg.crops_path+self.videos[self.chosenVideoIdx]))#here are all the files we have detections for
        self.crop_files = [str(frame) for frame in crop_files]
        
        #get the fc6 features of the current video
        feat,frames,coords,vids = load_features('fc6',cfg.features_path,self.videos[self.chosenVideoIdx],progress_bar=False)
        #remove dublicates
        _,indices = np.unique(frames,return_index=True)
        feat,frames,coords,vids = feat[indices],frames[indices],coords[indices],vids[indices]
        
        self.current_feats = {'frames': frames, 'features': feat, 'coords': coords}
        
        #get the lstm features of the current video
        feat,frames,coords,vids = load_features('lstm',cfg.features_path,self.videos[self.chosenVideoIdx],progress_bar=False)
        
        #remove doubles
        _,indices = np.unique(frames,return_index=True,axis=0)
        feat,frames,coords,vids = feat[indices],frames[indices],coords[indices],vids[indices]
        self.current_feats_seqs = {'frames': frames, 'features': feat, 'coords': coords}
        
        #compute the middle frames for the sequences
        self.get_middle_frames()
        
        #set the first frame
        self.current_idx = 0 # for postures and sequences
        if self.repr_type==0:#postures
            self.current_frame_name = self.crop_files[self.current_idx]
            self.current_frame_number = int(''.join(c for c in self.current_frame_name if c.isdigit()))
        else:#sequences:
            self.current_frame_name = '%06d.jpg'%(self.middle_frames[self.current_idx])
            self.current_frame_number = self.middle_frames[self.current_idx]
        
        #write the chosen video in info label
        #print self.Label_info_video.geometry()
        self.Label_info_video.setText('Current Video: %s \n #Frames: %i \n #Sequences %i'%(self.videos[self.chosenVideoIdx],len(self.crop_files),feat.shape[0]))
        QtTest.QTest.qWait(0.1*1000)#otherwise the new geometry of Label_info_video is not updated
        
        #activate specific windows
        if self.repr_type==0:
            self.Frame_adjustBB.setEnabled(True)
        
        self.RadioButton_adjustBB.setChecked(False)
        self.Widget_changeBBSize.setEnabled(False)
        self.Frame_play.setEnabled(True)
        #self.Widget_moveFrames.setEnabled(True)
        self.Widget_buttons_result.setEnabled(True)
        self.Widget_buttons_frame.setEnabled(True)
        self.Widget_chooseReprType.setEnabled(True)
        self.Frame_save.setEnabled(True)
        self.Frame_vidInfo.setEnabled(True)
        self.Label_frame.setEnabled(True)
        self.Label_showResult_title.setEnabled(True)
        self.Label_save_currentFolder.setText(self.save_dir)
        self.Widget_buttons_result.setEnabled(True)
        self.doubleSpinBox_threshold.setEnabled(True)
        self.label_threshold.setEnabled(True)
        
        #show the first image
        self.draw_frame()
        
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    def draw_frame(self,pos=[]):
        self.clear_info_inbetween()
        self.Label_showResult.clear()
        
        if self.repr_type==0:
            if not pos:
                frame_idx = np.where(self.current_feats['frames']==self.current_frame_number)[0][0]
                pos = self.current_feats['coords'][frame_idx]
                self.default_BB = True
                penRectangle = QPen(Qt.red)
            else:
                penRectangle = QPen(Qt.blue)
                
                #self.SpinBox_BBSize_x.setValue(100)
                #self.SpinBox_BBSize_y.setValue(100)
            
            #get the image
            pixmap = QPixmap.fromImage(QImage(cfg.frames_path+self.chosenVideo+'/'+self.current_frame_name))
            #pixmap = QPixmap(cfg.frames_path+str(self.chosenVideo)+'/'+self.current_frame_name)
            self.frame_orig_size = [pixmap.width(),pixmap.height()]
            
            ## draw the detection on top
            recPainter = QPainter(pixmap)
            # set rectangle color and thickness
            #penRectangle = QPen(Qt.red)
            penRectangle.setWidth(3)
            # draw rectangle on painter
            recPainter.setPen(penRectangle)
            
            if self.default_BB:#update the position in case we want to use the whole image and therefore c[2]=c[3]=-1
                pos = self.update_coords(pos,[pixmap.width(),pixmap.height()])
                self.current_BB_pos = [pos[0]+(pos[2]-pos[0])/2,pos[1]+(pos[3]-pos[1])/2]#middle point
                self.BB_size = [pos[2]-pos[0],pos[3]-pos[1]]
                
            recPainter.drawRect(pos[0],pos[1],pos[2]-pos[0],pos[3]-pos[1])
            recPainter.end()
            ##
            
            ##Draw the frame number on top of the image
            #font = QFont()
            #font.setPointSizeF(20)
            #metrics = QFontMetricsF(font)
            #rect = metrics.boundingRect(str(self.current_frame_number))
            #position = -rect.topRight()
            
            
            frameNrPainter = QPainter(pixmap)
            frameNrPainter.setPen(QPen(Qt.white))
            frameNrPainter.setFont(QFont("Arial", 20))
            #frameNrPainter.setFont(font)
            frameNrPainter.drawText(pixmap.rect(),Qt.AlignTop | Qt.AlignRight,'Frame ID: '+str(self.current_frame_number))
            frameNrPainter.end()
            
            #adjust to the size of the widget
            #height = self.Widget_buttons_frame.geometry().y()+self.Widget_buttons_frame.geometry().height()-self.Label_frame.geometry().y()
            smaller_pixmap = pixmap.scaled(self.Label_showFrame.frameGeometry().width(),
                                        #height,
                                        self.Label_showFrame.frameGeometry().height(),
                                        Qt.KeepAspectRatio,
                                        transformMode = Qt.FastTransformation)
            
            self.frame_new_size = [smaller_pixmap.width(),smaller_pixmap.height()]
            self.Label_showFrame.setPixmap(smaller_pixmap)
            #self.Label_showFrame.setFixedWidth(smaller_pixmap.width())
            self.Label_showFrame.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            #since we align, we need to find out where pixmap start on QLabel for the mousepressevent
            self.pixmap_coords_start = [self.Label_showFrame.width()/2-smaller_pixmap.width()/2,
                                        self.Label_showFrame.height()/2-smaller_pixmap.height()/2]
            
            
        else:#sequences
            fig3 = plt.figure(3)
            
            self.default_BB = True#only default is possible
            
            frames = self.current_feats_seqs['frames'][self.current_idx]#all frames of current sequence
            coords = self.current_feats_seqs['coords'][self.current_idx]#all coords of current sequence
            
            pos = coords[len(frames)//2]#coords of middle frame
            
            gs = gridspec.GridSpec(2,cfg.seq_len,height_ratios = [5,1])
            gs.update(hspace=0.1,wspace=0)
            
            #first plot main image
            ax_main = plt.subplot(gs[0,:])
            ax_main.clear()
            image = Image.open(cfg.frames_path+self.chosenVideo+'/'+self.current_frame_name)
            rect = patches.Rectangle((pos[0],pos[1]),pos[2]-pos[0],pos[3]-pos[1],edgecolor='r',fill=False)
            _=ax_main.add_patch(rect)
            _=ax_main.imshow(image)
            _=ax_main.text(image.size[0],0.5,'Frame ID: '+str(self.current_frame_number),horizontalalignment='right',verticalalignment='top',color='white',fontsize=12)
            _=ax_main.axis('off')
            
            #then plot the whole sequence, but only the cropped version
            vid = self.videos[self.chosenVideoIdx]
            if vid[0]=='/': vid = vid[1:]
                
            for i,(f,c) in enumerate(zip(frames,coords)):
                image_patch = Image.open('%s/%s/%06d.jpg'%(cfg.crops_path,vid,f))
                
                if i==len(frames)//2:#update pos if default values (-1) and then current_BB_pos
                    pos = self.update_coords(c,image_patch.size)
                    self.current_BB_pos = [pos[0]+(pos[2]-pos[0])/2,pos[1]+(pos[3]-pos[1])/2]#middle point
                
                #image_patch = image.crop(c)
                
                ax = plt.subplot(gs[1,i])
                ax.clear()
                _=ax.imshow(image_patch)
                
                #show that this is the main one
                if i==len(frames)//2:
                    rect = patches.Rectangle((0,0),image_patch.size[0],image_patch.size[1],edgecolor='r',fill=False,linewidth=5)
                    _=ax.add_patch(rect)
                
                _=ax.axis('off')
            
            plt.savefig('frame.jpg',bbox_inches='tight',dpi=100)
            pixmap = QPixmap('frame.jpg')
            smaller_pixmap = pixmap.scaled(self.Label_showFrame.frameGeometry().width(),
                                        self.Label_showFrame.frameGeometry().height(),
                                        Qt.KeepAspectRatio,
                                        transformMode = Qt.FastTransformation)
            
            self.Label_showFrame.setPixmap(smaller_pixmap)
            self.Label_showFrame.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    def get_all_feats(self):
        #first get the features of the postures (fc6)
        feat,frames,coords,vids = load_features('fc6',cfg.features_path,self.videos.tolist(),progress_bar=True,progress_bar_text='Extract Posture Features')
        self.feats = {'frames':frames,'coords': coords,'vids': vids,'features': feat}
        self.nr_embedding_dataset = min(self.nr_embedding_dataset,feat.shape[0])#since 'Postures' is default
        #then get the features of the sequences (lstm)
        #feat,frames,coords,vids = np.zeros([10,128]),np.zeros([10]),np.zeros([10,4]),np.zeros([10])
        #feat,frames,coords,vids = load_features('lstm',cfg.features_path,self.videos.tolist(),progress_bar=True,progress_bar_text='Extract Sequence Features')
        #self.feats_seqs = {'frames':frames,'coords': coords,'vids': vids,'features': normalize(feat)}
        pos_time    =np.concatenate([self.video_time[self.videos==v] for v in vids])
        healthy, impaired = pos_time==0, pos_time==1
        self.healthy  = {'features':feat[healthy], 'vids': vids[healthy], 'frames':frames[healthy],'coords':coords[healthy]}
        self.impaired = {'features':feat[impaired], 'vids': vids[impaired], 'frames':frames[impaired],'coords':coords[impaired]}
    
    def compute(self):
        t1 = time.time()
        self.set_info_inbetween('Computing.........')
        vid_query = self.chosenVideo
        #get the features of the query image
        if self.default_BB:
            idx = np.where(self.current_frame_number==self.current_feats['frames'])[0][0]
            feat_query = np.expand_dims(self.current_feats['features'][idx],axis=0)
        else:
            if not hasattr(self,'feats_ROI'): self.extract_features()
            feat_query = self.feats_ROI
        
        q_image = Image.open(cfg.frames_path+self.chosenVideo+'/'+self.current_frame_name)
        
        sort_idx = cdist(feat_query,self.healthy['features'])[0].argsort()[:20]
        h_images = [load_image(cfg.crops_path,self.healthy['vids'][nn],self.healthy['frames'][nn]) for nn in sort_idx]
        healthy_res, impaired_res, magnified_res = self.generator.extrapolate_multiple(
                    h_images, self.healthy['features'][sort_idx],
                    [q_image],feat_query,
                    [0.0,1.0,2.5])
        diff_image,flow_filtered,X,Y = find_differences_cc(healthy_res,impaired_res,magnified_res,
                                                           Th=self.doubleSpinBox_threshold.value())
        self.plot_magnification(healthy_res, impaired_res, magnified_res,diff_image,flow_filtered,X,Y)
        self.set_info_inbetween('Magnification produced in %.2f seconds'%(time.time()-t1))
    
    def plot_magnification(self,healthy, impaired, extrapolated,diff_image,flow_filtered,X,Y):
        healthy  = draw_border(healthy, l=3,color=[0,1.0,0])
        R, C = 2, 3
        _=plt.close()
        fig = plt.figure(1,figsize=(2*C,2*R))
        gs = gridspec.GridSpec(R,C,width_ratios=[1]*3)
        #gs.update(hspace=0.5,wspace=0.2)
        ax = [plt.subplot(gs[0,j]) for j in range(3)]+[plt.subplot(gs[1,j+1]) for j in range(2)]
        _=ax[0].imshow(healthy); _=ax[0].axis('off')
        _=ax[0].set_title('Healthy',fontsize=14,color='green')
        _=ax[1].imshow(impaired); _=ax[1].axis('off')
        _=ax[1].set_title('Query',fontsize=14,color='blue')
        _=ax[2].imshow(extrapolated); _=ax[2].axis('off')
        _=ax[2].set_title('Magnified',fontsize=14,color='red')
        _=ax[3].imshow(diff_image); _=ax[3].axis('off')
        _=ax[3].set_title('Difference',fontsize=14,color='red')#,x=0.55)
        _=ax[4].imshow(impaired); _=ax[4].axis('Off')
        
        _=ax[4].quiver(X, Y, flow_filtered[:,0], flow_filtered[:,1], 
                width=0.1, headwidth=3, color='red',scale_units='width', scale=10,minlength=0.1)
        _=ax[4].set_title('Improvement',fontsize=14,color='red')#,x=0.70)
        
        _=plt.savefig('NN.jpg',bbox_inches='tight',dpi=150)
        _=plt.close()
        
        self.Label_showResult_title.setText('Enhancment of behavioral differences')
        self.Label_showResult_title.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap('NN.jpg')
        smaller_pixmap = pixmap.scaled(self.Label_showResult.frameGeometry().width(),
                                    self.Label_showResult.frameGeometry().height(),
                                    Qt.KeepAspectRatio,
                                    transformMode = Qt.FastTransformation)
        self.Label_showResult.setPixmap(smaller_pixmap)
    
    def init_network(self):
        self.net = CaffeNet(batchsize=1,
               input_shape=(3,cfg.input_img_size,cfg.input_img_size),
               seq_len = cfg.seq_len,
               gpu=False)
        self.net.to(torch.device("cpu"))
        #initialize the weights
        if cfg.final_weights is None:#load the last saved state from training
            try:
                checkpoints = sorted(glob(cfg.checkpoint_path+'*.pth.tar'))
                init_dict = torch.load(checkpoints[-1])['state_dict']
            except:
                print('Weights not found! Please set the path for the weights in the config file!')
                
        else:
            init_dict = torch.load(cfg.final_weights, map_location=lambda storage, loc: storage)['state_dict']
        
        self.net.load_weights(init_dict,show=False)
        self.net.eval()
        self.generator = Generator(z_dim=cfg.encode_dim,path_model=cfg.vae_weights_path,device='gpu')
    
    def extract_features(self):
        if self.repr_type==0:#postures
            #get patch
            image = Image.open(cfg.frames_path+self.chosenVideo+'/'+self.current_frame_name)
            c = [self.current_BB_pos[0]-self.BB_size[0]/2,
                self.current_BB_pos[1]-self.BB_size[1]/2,
                self.current_BB_pos[0]+self.BB_size[0]/2,
                self.current_BB_pos[1]+self.BB_size[1]/2]
            
            image_patch = image.crop(c)
            #transform patch for the network
            image_patch = np.array(image_patch.resize((cfg.input_img_size,cfg.input_img_size),Image.BILINEAR),dtype=np.float32)
            image_patch = np.reshape(image_patch,(1,cfg.input_img_size,cfg.input_img_size,3))
            image_patch = transform_batch(image_patch)
            image_patch = Variable(torch.from_numpy(image_patch))
            image_patch = image_patch.float()
            #extract features
            fc6,_ = self.net.extract_features_fc(image_patch)
            self.feats_ROI = fc6.detach().numpy()
        
        elif self.repr_type==1:#sequences
            raise NotImplementedError


def main():
    app = QApplication(sys.argv)
    form = NNInterface()
    form.show()
    app.exec_()
    _=plt.close()
    if os.path.isfile('NN.jpg'): os.remove('NN.jpg')
    if os.path.isfile('tsne.jpg'): os.remove('tsne.jpg')
    if os.path.isfile('frame.jpg'): os.remove('frame.jpg')
    
    
if __name__ == '__main__':  # if we're running file directly and not importing it
    main()   



#app = QApplication(sys.argv)
#form = NNInterface()
#form.show()
#_=plt.close()
#os.remove('NN.jpg')
#os.remove('tsne.jpg')
#os.remove('frame.jpg')