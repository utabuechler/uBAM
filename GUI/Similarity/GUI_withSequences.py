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
os.environ["CUDA_VISIBLE_DEVICES"]=""#make sure that it is run on the cpu
sys.path.append("features/pytorch")
from model import CaffeNet
sys.path.append('')#so that we can load modules from the current directory
from utils import load_table, load_features, transform_batch
sys.path.append('./GUI/Similarity/')
import design3 as design


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-cf", "--config",type=str,default='config_pytorch_human',############################################################
                    help="Define config file")
args = parser.parse_args()

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
        self.fill_comboBox()
        self.nr_embedding_dataset = 5000
        self.nr_embedding_dataset_seqs = 5000
        
        self.get_all_feats()#collect the features of all crops
        self.init_network()
        
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
        #self.RadioButton_embedding.clicked.connect(self.activate_emb_onlyVideo)
        self.RadioButton_showClusters.clicked.connect(self.activate_clusters)
        #self.PushButton_defaultBB.clicked.connect(self.draw_frame)
        
        #self.PushButton_compute.clicked.connect(self.compute)
        self.ComboBox_compute.activated.connect(self.method)
        self.SpinBox_BBSize_x.valueChanged.connect(self.adjust_BB_directly)
        self.SpinBox_BBSize_y.valueChanged.connect(self.adjust_BB_directly)
        #self.PushButton_computeNN.clicked.connect(self.show_NN)
        #self.PushButton_computeTSNE.clicked.connect(self.show_TSNE)
        
        self.PushButton_reloadResult.clicked.connect(self.compute)
        self.PushButton_clearResult.clicked.connect(self.clear_result_labels)
        self.PushButton_saveResult.clicked.connect(partial(self.save_result,True))
        self.PushButton_save_changeFolder.clicked.connect(self.change_dir)
        
        
        
        self.k = self.SpinBox_chooseNumberNN.value()
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
        
        #self.Label_showFrame.interface = self
        #self.is_pixmap = False
        
    #def bigger(self):
        #self.Label_showFrame.setGeometry(QRect(self.Label_showFrame.geometry().left(),
                                               #self.Label_showFrame.geometry().top(),
                                               #self.Label_showFrame.geometry().width()+(0.1*self.Label_showFrame.geometry().width()),
                                               #self.Label_showFrame.geometry().height()+(0.1*self.Label_showFrame.geometry().height())))
        #print(self.Label_showFrame.geometry())
        #self.draw_frame()
    
    
    #def smaller(self):
        #self.Label_showFrame.setGeometry(QRect(self.Label_showFrame.geometry().left(),
                                               #self.Label_showFrame.geometry().top(),
                                               #self.Label_showFrame.geometry().width()-(0.1*self.Label_showFrame.geometry().width()),
                                               #self.Label_showFrame.geometry().height()-(0.1*self.Label_showFrame.geometry().height())))
        #print(self.Label_showFrame.geometry())
        #self.draw_frame()
    
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
                self.SpinBox_chooseNumberNN.setValue(12)
                self.k = 12
            else:
                self.Frame_adjustBB.setEnabled(False)
                self.Label_frame.setText('3. Choose Middle Frame of Sequence')
                self.nr_embedding_dataset = min(self.nr_embedding_dataset,self.feats_seqs['features'].shape[0])
                
                #update current_idx since the idx for postures mean something else than for sequences
                self.current_idx = (np.abs(self.current_frame_number-np.array(self.middle_frames))).argmin()#find closest to this frame number
                self.current_frame_name = '%06d.jpg'%(self.middle_frames[self.current_idx])
                self.current_frame_number = self.middle_frames[self.current_idx]
                
                #set default value for number of nearest neighbors
                self.SpinBox_chooseNumberNN.setValue(5)
                self.k = 5
            
            self.draw_frame()
    
    def activate_clusters(self):
        if self.RadioButton_showClusters.isChecked():
            self.Widget_nrClusters.setEnabled(True)
        else:
            self.Widget_nrClusters.setEnabled(False)
    
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
    
    def method(self):
        if (self.last_result=='NN' and self.ComboBox_compute.currentIndex()==1) or \
            (self.last_result=='embedding' and self.ComboBox_compute.currentIndex()==0):#if it changed reset labels
            
            self.Label_showResult.clear()
            self.clear_info_inbetween()
            self.Label_showResult_title.setText('Result')
            self.Label_showResult_title.setAlignment(Qt.AlignCenter)
        
        #only enable for postures, not for sequences
        if self.repr_type==0:
            self.Frame_adjustBB.setEnabled(True)
        else:
            self.Frame_adjustBB.setEnabled(False)
        
        if self.ComboBox_compute.currentIndex()==1:#embedding
            self.Frame_nrNN.setEnabled(False)
            self.Frame_embSource.setEnabled(True)
            if not IS_UMAP:#if we want the low-dim embedding, but don't have UMAP, we cannot map the new detection
                self.Frame_adjustBB.setEnabled(False)
                self.set_info_inbetween('Adjusting Area of Interest not available with TSNE as Low Dimensionality Embedding, only with UMAP')
                
        else:#nearest neighbors
            self.Frame_nrNN.setEnabled(True)
            self.Frame_embSource.setEnabled(False)
    
    def compute(self):
        self.clear_info_inbetween()
        
        if self.ComboBox_compute.currentIndex()==0:
            self.last_result='NN'
            #update number of Nearets Neighbors
            self.k = self.SpinBox_chooseNumberNN.value()
            
            if self.repr_type==0:#postures
                self.show_NN()
            else:#sequences
                self.show_NN_seqs()
        else:
            self.last_result = 'embedding'
            
            if self.repr_type==0:#postures
                self.show_low_dim_embedding()
            else:#sequences
                self.show_low_dim_embedding_seqs()
                
        
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
            
    
    def compute_low_dim_embedding(self):
        
        if IS_UMAP:#use UMAP
            
            file_name = '%s/embedding_dataset_%i'%(self.save_dir,self.nr_embedding_dataset)
            
            #compute embedding for the dataset
            if not hasattr(self, 'low_dim_points_all'):
                #file_dir = '%s/GUI/Similarity'%(cfg.results_path)
                #if not os.path.isdir(file_dir):
                #    os.makedirs(file_dir)
                
                
                #set text
                self.set_info_inbetween('Compute Low Dimensionality Embedding with UMAP for Dataset.....')
                QtTest.QTest.qWait(1*1000)
                
                #compute embedding and save it
                if not os.path.isfile(file_name+'_umap.sav'):
                    self.plot_centroids = True
                    
                    #collect randomly self.nr_embedding_dataset frames for creating the low-dim embedding
                    #idx = np.linspace(0,len(self.feats['features'])-1,self.nr_embedding_dataset).astype(int)
                    idx = random.sample(range(len(self.feats['features'])),self.nr_embedding_dataset)
                    pickle.dump(idx,open(file_name+'_feat_idx.sav','wb'))
                    
                    low_dim_method = umap.UMAP(metric='cosine')#n_neighbors=10,metric='cosine')
                    
                    #fit model
                    feats_norm = self.feats['features'][idx ,:]
                    low_dim_method.fit(feats_norm)
                    save_umap(low_dim_method,open(file_name+'_umap.sav','wb'))
                    
                    #transform dataset points
                    low_dim_points_all = low_dim_method.transform(feats_norm)
                    pickle.dump(low_dim_points_all,open(file_name+'_umap_points.sav','wb'))
                    
                #load embedding
                self.low_dim_method = load_umap(open(file_name+'_umap.sav','rb'))
                self.low_dim_points_all = pickle.load(open(file_name+'_umap_points.sav','rb'))
                self.feats_idx = pickle.load(open(file_name+'_feat_idx.sav','rb'))
                
            if self.RadioButton_showClusters.isChecked() and (not hasattr(self,'kmeans') or self.nr_clusters!=self.SpinBox_nrClusters.value()):
                self.nr_clusters = self.SpinBox_nrClusters.value()
                
                self.set_info_inbetween('Compute Clustering.....')
                QtTest.QTest.qWait(1*1000)
                
                #do clustering
                if not os.path.isfile('%s_kmeans%04d.sav'%(file_name,self.nr_clusters)):
                    if 'feats_norm' not in locals():
                        feats_norm = self.feats['features'][self.feats_idx ,:]
                    
                    kmeans = KMeans(n_clusters=self.nr_clusters, random_state=0).fit(feats_norm)
                    pickle.dump(kmeans,open('%s_kmeans%04d.sav'%(file_name,self.nr_clusters),'wb'))
                    #save the centroids
                    centroid_idx = np.array(self.feats_idx)[np.argmin(cdist(feats_norm, kmeans.cluster_centers_, 'euclidean'), axis=0).astype(int)]
                    pickle.dump(centroid_idx,open('%s_kmeans%04d_centroids.sav'%(file_name,self.nr_clusters),'wb'))
                    
                
                #load clusters
                self.kmeans = pickle.load(open('%s_kmeans%04d.sav'%(file_name,self.nr_clusters),'rb'))
                self.low_dim_points_all_clusterlabel = self.kmeans.labels_
                self.centroids = pickle.load(open('%s_kmeans%04d_centroids.sav'%(file_name,self.nr_clusters),'rb'))
            
                #reset text
                self.clear_info_inbetween()
                
                    
            #get embedding for current video
            if not hasattr(self,'low_dim_points_vid'):
                #set text
                self.set_info_inbetween('Compute Low Dimensionality Embedding with UMAP for Current Video.....')
                QtTest.QTest.qWait(1*1000)
                
                #compute
                feats_vid = self.current_feats['features']
                self.low_dim_points_vid = self.low_dim_method.transform(feats_vid)
                
                self.low_dim_points_vid_clusterlabel = self.kmeans.predict(feats_vid)
                #self.low_dim_points_all_clusterlabel = self.kmeans.labels_[:self.low_dim_points_all.shape[0]]
                
                #reset label
                self.clear_info_inbetween()
                
                
            #get embedding for current frame
            if self.default_BB:
                idx = np.where(self.current_feats['frames']==self.current_frame_number)[0]
                self.low_dim_points_ROI = [self.low_dim_points_vid[idx,0],self.low_dim_points_vid[idx,1]]
                self.low_dim_points_ROI_clusterlabel = int(self.low_dim_points_vid_clusterlabel[idx])
            else:
                if not hasattr(self,'feats_ROI'): self.extract_features()
                feats_query = self.feats_ROI
                self.low_dim_points_ROI = self.low_dim_method.transform(feats_query)[0]
                self.low_dim_points_ROI_clusterlabel = int(self.kmeans.predict(feats_query))
            
        else:#use TSNE
            raise NotImplementedError
            #self.Label_info_inbetween.setText('Not Implemented Yet, only works with UMAP!')
            #self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
            #self.Label_info_inbetween.setStyleSheet("QLabel { background-color : blue; color : white; }")
     
     
    def compute_low_dim_embedding_seqs(self):
        if  not IS_UMAP:
            raise NotImplementedError
        
        file_name = '%s/embedding_dataset_%i_seq'%(self.save_dir,self.nr_embedding_dataset_seqs)
        
        #compute embedding for the dataset
        if not hasattr(self, 'low_dim_points_all_seqs'):
            
            #set text
            self.set_info_inbetween('Compute Low Dimensionality Embedding with UMAP for Dataset.....')
            QtTest.QTest.qWait(1*1000)
        
            #compute embedding and save it
            if not os.path.isfile(file_name+'_umap.sav'):
                self.plot_centroids = True
                
                #collect randomly self.nr_embedding_dataset_seqs frames for creating the low-dim embedding
                #idx = np.linspace(0,len(self.feats_seqs['features'])-1,self.nr_embedding_dataset_seqs).astype(int)
                idx = random.sample(range(len(self.feats_seqs['features'])),self.nr_embedding_dataset_seqs)
                pickle.dump(idx,open(file_name+'_feat_idx.sav','wb'))
                
                low_dim_method = umap.UMAP(metric='cosine')#n_neighbors=10,metric='cosine')
                
                #fit model
                feats_norm_seq = self.feats_seqs['features'][idx ,:]
                low_dim_method.fit(feats_norm_seq)
                save_umap(low_dim_method,open(file_name+'_umap.sav','wb'))
                
                #transform dataset points
                low_dim_points_all = low_dim_method.transform(feats_norm_seq)
                pickle.dump(low_dim_points_all,open(file_name+'_umap_points.sav','wb'))
                
            #load embedding
            self.low_dim_method_seqs = load_umap(open(file_name+'_umap.sav','rb'))
            self.low_dim_points_all_seqs = pickle.load(open(file_name+'_umap_points.sav','rb'))
            self.feats_idx_seqs = pickle.load(open(file_name+'_feat_idx.sav','rb'))
            
        if self.RadioButton_showClusters.isChecked() and (not hasattr(self,'kmeans_seqs') or self.nr_clusters!=self.SpinBox_nrClusters.value()):
            self.nr_clusters = self.SpinBox_nrClusters.value()
            
            self.set_info_inbetween('Compute Clustering.....')
            QtTest.QTest.qWait(1*1000)
            
            #do clustering
            if not os.path.isfile('%s_kmeans%04d.sav'%(file_name,self.nr_clusters)):
                if 'feats_norm_seq' not in locals():
                    feats_norm_seq = self.feats_seqs['features'][self.feats_idx_seqs ,:]
                
                kmeans = KMeans(n_clusters=self.nr_clusters, random_state=0).fit(feats_norm_seq)
                pickle.dump(kmeans,open('%s_kmeans%04d.sav'%(file_name,self.nr_clusters),'wb'))
                #save the centroids
                centroid_idx = np.array(self.feats_idx_seqs)[np.argmin(cdist(feats_norm_seq, kmeans.cluster_centers_, 'euclidean'), axis=0).astype(int)]
                pickle.dump(centroid_idx,open('%s_kmeans%04d_centroids.sav'%(file_name,self.nr_clusters),'wb'))
                
            
            #load clusters
            self.kmeans_seqs = pickle.load(open('%s_kmeans%04d.sav'%(file_name,self.nr_clusters),'rb'))
            self.low_dim_points_all_clusterlabel_seqs = self.kmeans_seqs.labels_
            self.centroids_seqs = pickle.load(open('%s_kmeans%04d_centroids.sav'%(file_name,self.nr_clusters),'rb'))
        
            #reset text
            self.clear_info_inbetween()
            
                
        #get embedding for current video
        if not hasattr(self,'low_dim_points_vid_seqs'):
            #set text
            self.set_info_inbetween('Compute Low Dimensionality Embedding with UMAP for Current Video.....')
            QtTest.QTest.qWait(2*1000)
            
            #compute
            feats_vid_seqs = self.current_feats_seqs['features']
            self.low_dim_points_vid_seqs = self.low_dim_method_seqs.transform(feats_vid_seqs)
            
            self.low_dim_points_vid_clusterlabel_seqs = self.kmeans_seqs.predict(feats_vid_seqs)
            #self.low_dim_points_all_clusterlabel = self.kmeans.labels_[:self.low_dim_points_all.shape[0]]
            
            #reset label
            self.clear_info_inbetween()
            
            
        #get embedding for current frame
        if self.default_BB:
            #idx = np.where(self.current_feats_seqs['frames']==self.current_frame_number)[0]
            #frames = self.current_feats_seqs['frames'][self.current_idx]
            #idx = np.where(self.current_feats_seqs['frames']==self.current_frame_number)[0]
            self.low_dim_points_ROI_seqs = [self.low_dim_points_vid_seqs[self.current_idx,0],self.low_dim_points_vid_seqs[self.current_idx,1]]
            self.low_dim_points_ROI_clusterlabel_seqs = int(self.low_dim_points_vid_clusterlabel_seqs[self.current_idx])
        #else:
        #    feat_ROI = normalize(self.extract_features())
        #    self.low_dim_points_ROI = self.low_dim_method.transform(feat_ROI)[0]
        #    self.low_dim_points_ROI_clusterlabel = self.kmeans.predict(feat_ROI) 
    
    def show_low_dim_embedding_seqs(self):
        self.compute_low_dim_embedding_seqs()
        
        #_=plt.close()
        fig2 = plt.figure(2,figsize=(25,self.nr_clusters+3))
        #fig2.clf()
        
        gs = gridspec.GridSpec(self.nr_clusters+3,cfg.seq_len+2,
                        height_ratios = [0.1,1,0.5]+np.ones((self.nr_clusters,),np.int).tolist(),
                        width_ratios=[15,1]+np.ones((cfg.seq_len,),np.int).tolist())
        #gs = gridspec.GridSpec(self.nr_clusters+4,cfg.seq_len+1,
                        #height_ratios = [0.1,1,0.5,0.5]+np.ones((self.nr_clusters,),np.int).tolist(),
                        #width_ratios=[15]+np.ones((cfg.seq_len,),np.int).tolist())
        gs.update(hspace=0.1,wspace=0.1)
        
        x_lim = [np.min(self.low_dim_points_all_seqs[:,0]),np.max(self.low_dim_points_all_seqs[:,0])]
        x_lim = [x_lim[0]-0.05*(x_lim[1]-x_lim[0]),x_lim[1]+0.05*(x_lim[1]-x_lim[0])]#+5%
        y_lim = [np.min(self.low_dim_points_all_seqs[:,1]),np.max(self.low_dim_points_all_seqs[:,1])]
        y_lim = [y_lim[0]-0.05*(y_lim[1]-y_lim[0]),y_lim[1]+0.05*(y_lim[1]-y_lim[0])]#+5%
        
        #with Clusters
        if self.RadioButton_showClusters.isChecked():
            #plt.figure(figsize=(15,10))
            
            #get the color for the clusters
            colors = [cm.jet(val) for val in np.linspace(0,1,self.nr_clusters)]
            
            ax_tsne = plt.subplot(gs[:,0])
            ax_tsne.clear()
            
            #plot the points of dataset, consider the clusters
            if not self.RadioButton_embedding.isChecked():
                h_dataset, = ax_tsne.plot(self.low_dim_points_all_seqs[0,0],self.low_dim_points_all_seqs[0,1],'.',color='k')#,markersize=1)#plot one dot for legend
                #h_dataset, = ax_tsne.plot(self.low_dim_points_all[0,0],self.low_dim_points_all[0,1],'o',color='k')#plot one dot for legend
                for i in range(self.nr_clusters):
                    idx = np.where(self.low_dim_points_all_clusterlabel_seqs==i)[0]
                    _=ax_tsne.plot(self.low_dim_points_all_seqs[idx,0],self.low_dim_points_all_seqs[idx,1],'.',color=colors[i])#,markersize=1)
                    #_=ax_tsne.plot(self.low_dim_points_all[idx,0],self.low_dim_points_all[idx,1],'o',markerfacecolor='k',markeredgewidth=1,markeredgecolor=colors[i])
            
            #plot the points of video, consider the clusters
            h_video, = ax_tsne.plot(self.low_dim_points_vid_seqs[0,0],self.low_dim_points_vid_seqs[0,1],'x',color='k')#plot one dot for legend
            #h_video, = ax_tsne.plot(self.low_dim_points_vid[0,0],self.low_dim_points_vid[0,1],'o',color=(0.6,0.6,0.6,1))#plot one dot for legend
            for i in range(self.nr_clusters):
                idx = np.where(self.low_dim_points_vid_clusterlabel_seqs==i)[0]
                if not self.RadioButton_embedding.isChecked():
                    _=ax_tsne.plot(self.low_dim_points_vid_seqs[idx,0],self.low_dim_points_vid_seqs[idx,1],'x',color='k')
                else:
                    _=ax_tsne.plot(self.low_dim_points_vid_seqs[idx,0],self.low_dim_points_vid_seqs[idx,1],'x',color=colors[i])
            
            #highlight the current frame/ROI
            clusterlabel = self.low_dim_points_ROI_clusterlabel_seqs#self.kmeans.predict(normalize(self.extract_features()))[0]
            h_frame,=ax_tsne.plot(self.low_dim_points_ROI_seqs[0],self.low_dim_points_ROI_seqs[1],'*',color=(1,0.08,0.58),markersize=40,markeredgewidth=1,markeredgecolor=colors[clusterlabel])
            
            #set legend
            if not self.RadioButton_embedding.isChecked():
                lgd=ax_tsne.legend((h_dataset,h_video,h_frame),
                                ('%i Sequences of Dataset'%(self.nr_embedding_dataset_seqs),'Sequences of Current Video','Query Sequence'),prop={'size': 22},loc='upper right')
                lgd.legendHandles[2]._legmarker.set_markersize(15)
            else:
                lgd=ax_tsne.legend((h_video,h_frame),
                            ('Sequences of Current Video','Query Sequence'),prop={'size': 22},loc='upper right')
                lgd.legendHandles[1]._legmarker.set_markersize(15)
            
            _=ax_tsne.set_xlim(x_lim)
            _=ax_tsne.set_ylim(y_lim)
            _=plt.axis('off')
            
            #show query
            vid = self.videos[self.chosenVideoIdx]
            frames = self.current_feats_seqs['frames'][self.current_idx]
            #coords = self.current_feats_seqs['coords'][self.current_idx]
            
            #put a rectangle around query
            ax_rect = plt.subplot(gs[1,2:])
            ax_rect.clear()
            pos = ax_rect.get_position()
            fig2.patches.extend([plt.Rectangle((pos.x0,pos.y0),pos.width,pos.height,
                                            fill=False, edgecolor=colors[clusterlabel],linewidth=self.Label_showResult.width()/150,
                                            transform=fig2.transFigure, figure=fig2)])
            
            for j,f in enumerate(frames):
                image_patch = Image.open('%s/%s/%06d.jpg'%(cfg.crops_path,vid,f))
                #image_patch = image.crop(c)
                
                ax = plt.subplot(gs[1,j+2])
                ax.clear()
                _=ax.imshow(image_patch)
                #rect = patches.Rectangle((0,0),image_patch.size[0],image_patch.size[1],edgecolor=colors[clusterlabel],fill=False,linewidth=self.Label_showResult.width()/150)
                #_=ax.add_patch(rect)
                _=ax.axis('off')
            
            ##plot the centroids and others and save the information about the centroids in meta data
            if self.plot_centroids:#only plot the clusters and titles if necessary, this takes time, so we can save it, if the plot has it already
                #set title for Query
                ax_title = plt.subplot(gs[0,2:])
                ax_title.clear()
                _=ax_title.set_title('Query Sequence',fontdict={'fontsize': 25})
                _=ax_title.axis('off')
                
                #Line after Query
                ax_query_line = plt.subplot(gs[2,2:])
                ax_query_line.clear()
                _=ax_query_line.annotate("", xy=(1, 1), xytext=(0,1),arrowprops=dict(arrowstyle='simple, head_width=0.5',linewidth=1))
                _=ax_query_line.text(1,0.2,"time",fontsize=18,horizontalalignment='right',verticalalignment='center',transform=ax_query_line.transAxes)
                _=ax_query_line.axis('off')
                
                #title for centroids
                ax_title = plt.subplot(gs[3:,1])
                ax_title.clear()
                _=ax_title.text(0,0.6,'Centroids',rotation=90,fontsize=25, verticalalignment='center')
                _=ax_title.axis('off')
                
                #plot centroids
                #fig2 = plt.figure(2,figsize=(25,self.nr_clusters+3))
                meta_data_seq = []
                for i,c in enumerate(self.centroids_seqs):
                    vid = self.feats_seqs['vids'][c]
                    if vid[0]=='/': vid = vid[1:]
                    frames = self.feats_seqs['frames'][c]
                    coords = self.feats_seqs['coords'][c]
                    
                    meta_data = []
                    vid_id = np.where(self.videos==self.feats_seqs['vids'][c])[0][0]
                    
                    #put a rectangle around one sequence
                    ax_rect = plt.subplot(gs[i+3,2:])
                    ax_rect.clear()
                    pos = ax_rect.get_position()
                    fig2.patches.extend([plt.Rectangle((pos.x0,pos.y0),pos.width,pos.height,
                                                    fill=False, edgecolor=colors[i],linewidth=self.Label_showResult.width()/150,
                                                    transform=fig2.transFigure, figure=fig2)])
                    
                    for j,(f,c) in enumerate(zip(frames,coords)):
                        image_patch = Image.open('%s%s/%06d.jpg'%(cfg.crops_path,vid,f))
                        
                        ax = plt.subplot(gs[i+3,j+2])
                        #ax.clear()
                        _=ax.imshow(image_patch)
                        _=ax.axis('off')
                        
                        #put a border around the image!
                        #rect = patches.Rectangle((0,0),image_patch.size[0],image_patch.size[1],edgecolor=colors[i],fill=False,linewidth=self.Label_showResult.width()/150)
                        #_=ax.add_patch(rect)
                        #_=ax.axis('off')
                        
                        meta_data.append([i,j,vid_id,vid,f,c])
                    
                    meta_data_seq.append(meta_data)
                
                self.meta_data = meta_data_seq
                
                self.plot_centroids = False
                
                
                
        
        #without clusters
        else:
            ax_tsne = plt.subplot(gs[:,0])
            ax_tsne.clear()
            #plot dataset points
            if not self.RadioButton_embedding.isChecked():
                h_dataset, = ax_tsne.plot(self.low_dim_points_all_seqs[:,0],self.low_dim_points_all_seqs[:,1],'.',color='k')
            
            #plot video points
            h_video, = ax_tsne.plot(self.low_dim_points_vid_seqs[:,0],self.low_dim_points_vid_seqs[:,1],'x',color='g')
            
            #highlight current frame/ROI
            h_frame,=ax_tsne.plot(self.low_dim_points_ROI_seqs[0],self.low_dim_points_ROI_seqs[1],'*',color='r',markersize=35)
            
            #set legend
            if not self.RadioButton_embedding.isChecked():
                lgd=ax_tsne.legend((h_dataset,h_video,h_frame),
                                ('%i Sequences of Dataset'%(self.nr_embedding_dataset_seqs),'Sequences of Current Video','Query Sequence'))#loc=2,prop={'size': 6})
                lgd.legendHandles[2]._legmarker.set_markersize(15)
            else:
                lgd=ax_tsne.legend((h_video,h_frame),
                            ('Sequences of Current Video','Query Sequence'))#loc=2,prop={'size': 6})
                lgd.legendHandles[1]._legmarker.set_markersize(15)
            
            _=ax_tsne.set_xlim(x_lim)
            _=ax_tsne.set_ylim(y_lim)
            _=plt.axis('off')  
        
        #rcParams.update({'font.size': 14})
        self.plot_embedding = fig2
        
        #_=plt.savefig('tsne.jpg',bbox_inches='tight')
        fig2.tight_layout()
        _=plt.savefig('tsne.jpg',bbox_extra_artists=(lgd,),bbox_inches='tight',dpi=150)
        #_=plt.close()
        
        #set info
        self.Label_showResult_title.setText('Low Dimensionality Embedding')
        self.Label_showResult_title.setAlignment(Qt.AlignCenter)
        
        #show the result on QLabel
        pixmap = QPixmap('tsne.jpg')
        smaller_pixmap = pixmap.scaled(self.Label_showResult.frameGeometry().width(),
                                       self.Label_showResult.frameGeometry().height(),
                                       Qt.KeepAspectRatio,
                                       transformMode = Qt.FastTransformation)
        
        
        #change position of Label_showResult, so that it is in the middle of the title
        #mid = self.Label_showResult_title.geometry().left()+self.Label_showResult_title.geometry().width()/float(2)
        #self.Label_showResult.setGeometry(QRect(mid-smaller_pixmap.width()/float(2),
                                               #self.Label_showResult.geometry().top(),
                                               #smaller_pixmap.width(),
                                               #self.Label_showResult.geometry().height()))
        
        
        self.Label_showResult.setPixmap(smaller_pixmap)
        
    
    def show_low_dim_embedding(self):
        self.compute_low_dim_embedding()
        
        #if plt.fignum_exists(2):
        #    fig2.clf()
        #else:
        #_=plt.close()
        fig2 = plt.figure(2,figsize=(15,10))
        #fig2.clf()
            #plt.figure(figsize=(15,7))
        
        gs = gridspec.GridSpec(3,self.nr_clusters+2,height_ratios = [5,0.05,1],width_ratios=[1,0.1]+np.ones((self.nr_clusters,),np.int).tolist())
        gs.update(hspace=0.1,wspace=0.1)
        
        #get the xlim and ylim coordinates, should be the same for all, no matter if we show dataset or if we show the clusters
        x_lim = [np.min(self.low_dim_points_all[:,0]),np.max(self.low_dim_points_all[:,0])]
        x_lim = [x_lim[0]-0.05*(x_lim[1]-x_lim[0]),x_lim[1]+0.05*(x_lim[1]-x_lim[0])]#+5%
        y_lim = [np.min(self.low_dim_points_all[:,1]),np.max(self.low_dim_points_all[:,1])]
        y_lim = [y_lim[0]-0.05*(y_lim[1]-y_lim[0]),y_lim[1]+0.05*(y_lim[1]-y_lim[0])]#+5%
        
        #with Clusters
        if self.RadioButton_showClusters.isChecked():
            #plt.figure(figsize=(15,10))
            
            #get the color for the clusters
            colors = [cm.jet(val) for val in np.linspace(0,1,self.nr_clusters)]
            
            ax_tsne = plt.subplot(gs[0,:])
            ax_tsne.clear()
            
            
            #plot the points of dataset, consider the clusters
            if not self.RadioButton_embedding.isChecked():
                h_dataset, = ax_tsne.plot(self.low_dim_points_all[0,0],self.low_dim_points_all[0,1],'.',color='k')#,markersize=1)#plot one dot for legend
                #h_dataset, = ax_tsne.plot(self.low_dim_points_all[0,0],self.low_dim_points_all[0,1],'o',color='k')#plot one dot for legend
                for i in range(self.nr_clusters):
                    idx = np.where(self.low_dim_points_all_clusterlabel==i)[0]
                    _=ax_tsne.plot(self.low_dim_points_all[idx,0],self.low_dim_points_all[idx,1],'.',color=colors[i])#,markersize=1)
                    #_=ax_tsne.plot(self.low_dim_points_all[idx,0],self.low_dim_points_all[idx,1],'o',markerfacecolor='k',markeredgewidth=1,markeredgecolor=colors[i])
            
            #plot the points of video, consider the clusters
            h_video, = ax_tsne.plot(self.low_dim_points_vid[0,0],self.low_dim_points_vid[0,1],'x',color='k')#plot one dot for legend
            #h_video, = ax_tsne.plot(self.low_dim_points_vid[0,0],self.low_dim_points_vid[0,1],'o',color=(0.6,0.6,0.6,1))#plot one dot for legend
            for i in range(self.nr_clusters):
                idx = np.where(self.low_dim_points_vid_clusterlabel==i)[0]
                if not self.RadioButton_embedding.isChecked():
                    _=ax_tsne.plot(self.low_dim_points_vid[idx,0],self.low_dim_points_vid[idx,1],'x',color='k')
                else:
                    _=ax_tsne.plot(self.low_dim_points_vid[idx,0],self.low_dim_points_vid[idx,1],'x',color=colors[i])
            
            #highlight the current frame/ROI
            clusterlabel = self.low_dim_points_ROI_clusterlabel#self.kmeans.predict(normalize(self.extract_features()))[0]
            h_frame,=ax_tsne.plot(self.low_dim_points_ROI[0],self.low_dim_points_ROI[1],'*',color=(1,0.08,0.58),markersize=40,markeredgewidth=1,markeredgecolor=colors[clusterlabel])
            
            #set legend
            if not self.RadioButton_embedding.isChecked():
                lgd=ax_tsne.legend((h_dataset,h_video,h_frame),
                                ('%i Postures of Dataset'%(self.nr_embedding_dataset),'Posture of Current Video','Query Posture'),prop={'size': 20})
                lgd.legendHandles[2]._legmarker.set_markersize(15)
            else:
                lgd=ax_tsne.legend((h_video,h_frame),
                            ('Postures of Current Video','Query Posture'),prop={'size': 20})
                lgd.legendHandles[1]._legmarker.set_markersize(15)
            
            
            ax_tsne.set_xlim(x_lim)
            ax_tsne.set_ylim(y_lim)
            _=plt.axis('off')
            
            ax_query = plt.subplot(gs[2,0])
            ax_query.clear()
            image = Image.open(cfg.frames_path+self.chosenVideo+'/'+self.current_frame_name)
            c = [self.current_BB_pos[0]-self.BB_size[0]/2,
                self.current_BB_pos[1]-self.BB_size[1]/2,
                self.current_BB_pos[0]+self.BB_size[0]/2,
                self.current_BB_pos[1]+self.BB_size[1]/2]
            image_patch = image.crop(c)
            rect = patches.Rectangle((0,0),image_patch.size[0],image_patch.size[1],edgecolor=colors[clusterlabel],fill=False,linewidth=self.Label_showResult.width()/150)
            _=ax_query.add_patch(rect)
            _=ax_query.imshow(image_patch)
            _=ax_query.axis('off')
            
            ax_query_line = plt.subplot(gs[2,1])
            _=ax_query_line.vlines(0.5,0,1,colors='k',linewidth=2)
            _=ax_query_line.axis('off')
            
            #plot the centroids and others and save the information about the centroids in meta data
            if self.plot_centroids:#only plot the clusters if necessary, this takes time, so we can save it, if the plot has it already
                
                #set title for the centroids
                ax_title = plt.subplot(gs[1,2:])
                ax_title.clear()
                _=ax_title.set_title('Centroids of Clusters',fontdict={'fontsize': 20})
                _=ax_title.axis('off')
                
                #show query
                ax_query_title = plt.subplot(gs[1,0])
                ax_query_title.clear()
                _=ax_query_title.set_title('Query',fontdict={'fontsize': 20})
                _=ax_query_title.axis('off')
                
                meta_data = []
                for i,c in enumerate(self.centroids):
                    vid = self.feats['vids'][c]
                    if vid[0]=='/': vid = vid[1:]
                    frame = self.feats['frames'][c]
                    coords = self.feats['coords'][c]
                    
                    image = Image.open('%s%s/%06d.jpg'%(cfg.crops_path,vid,frame))
                    
                    ax = plt.subplot(gs[2,i+2])
                    ax.clear()
                    _=ax.imshow(image)
                    #put a border around the image!
                    rect = patches.Rectangle((0,0),image.size[0],image.size[1],edgecolor=colors[i],fill=False,linewidth=self.Label_showResult.width()/150)
                    _=ax.add_patch(rect)
                    _=plt.axis('off')
                    #_=ax.set_title('%d. Centroid'%(i+1))
                    
                    vid_id = np.where(self.videos==self.feats['vids'][c])[0][0]
                    meta_data.append([vid_id,vid,frame,coords])
                    
                self.meta_data = meta_data
                
                self.plot_centroids = False
        
        #without clusters
        else:
            ax_tsne = plt.subplot(gs[0,:])
            ax_tsne.clear()
            #plot dataset points
            if not self.RadioButton_embedding.isChecked():
                h_dataset, = ax_tsne.plot(self.low_dim_points_all[:,0],self.low_dim_points_all[:,1],'.',color='k')
            
            #plot video points
            h_video, = ax_tsne.plot(self.low_dim_points_vid[:,0],self.low_dim_points_vid[:,1],'x',color='g')
            
            #highlight current frame/ROI
            h_frame,=ax_tsne.plot(self.low_dim_points_ROI[0],self.low_dim_points_ROI[1],'*',color='r',markersize=35)
            
            #set legend
            if not self.RadioButton_embedding.isChecked():
                lgd=ax_tsne.legend((h_dataset,h_video,h_frame),
                                ('%i Postures of Dataset'%(self.nr_embedding_dataset),'Posture of Current Video','Postures of Current ROI'))#loc=2,prop={'size': 6})
                lgd.legendHandles[2]._legmarker.set_markersize(15)
            else:
                lgd=ax_tsne.legend((h_video,h_frame),
                            ('Postures of Current Video','Postures of Current ROI'))#loc=2,prop={'size': 6})
                lgd.legendHandles[1]._legmarker.set_markersize(15)
                
            
            ax_tsne.set_xlim(x_lim)
            ax_tsne.set_ylim(y_lim)
            
            _=plt.axis('off')  
            
        #rcParams.update({'font.size': 18})
        self.plot_embedding = fig2
        
        fig2.tight_layout()
        #_=plt.savefig('tsne.jpg',bbox_inches='tight')
        _=plt.savefig('tsne.jpg',bbox_extra_artists=(lgd,),bbox_inches='tight',dpi=150)
        #_=plt.close()
        
        #set info
        self.Label_showResult_title.setText('Low Dimensionality Embedding')
        self.Label_showResult_title.setAlignment(Qt.AlignCenter)
        
        #show the result on QLabel
        pixmap = QPixmap('tsne.jpg')
        smaller_pixmap = pixmap.scaled(self.Label_showResult.frameGeometry().width(),
                                       self.Label_showResult.frameGeometry().height(),
                                       Qt.KeepAspectRatio,
                                       transformMode = Qt.FastTransformation)
        
        
        #change position of Label_showResult, so that it is in the middle of the title
        #mid = self.Label_showResult_title.geometry().left()+self.Label_showResult_title.geometry().width()/float(2)
        #self.Label_showResult.setGeometry(QRect(mid-smaller_pixmap.width()/float(2),
                                               #self.Label_showResult.geometry().top(),
                                               #smaller_pixmap.width(),
                                               #self.Label_showResult.geometry().height()))
        
        
        self.Label_showResult.setPixmap(smaller_pixmap)
            
    
    def init_network(self):
        
        #device = torch.device("cpu")
        self.net = CaffeNet(batchsize=1,
               input_shape=(3,cfg.input_img_size,cfg.input_img_size),
               seq_len = cfg.seq_len,
               gpu=False)
        
        #net.to(device)
        
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
    
    def extract_features(self):
        
        if self.repr_type==0:#postures
            #get patch
            image = Image.open(cfg.frames_path+self.chosenVideo+'/'+self.current_frame_name)
            c = [self.current_BB_pos[0]-self.BB_size[0]/2,
                self.current_BB_pos[1]-self.BB_size[1]/2,
                self.current_BB_pos[0]+self.BB_size[0]/2,
                self.current_BB_pos[1]+self.BB_size[1]/2]
            
            #if cfg.project=='humans':
                #c[1] = 200 #start from the middle, because that's how we trained the network for the humans
            
            image_patch = image.crop(c)
            
            #transform patch for the network
            image_patch = np.array(image_patch.resize((cfg.input_img_size,cfg.input_img_size),Image.BILINEAR),dtype=np.float32)
            image_patch = np.reshape(image_patch,(1,cfg.input_img_size,cfg.input_img_size,3))
            
            image_patch = transform_batch(image_patch)
            image_patch = Variable(torch.from_numpy(image_patch))
            image_patch = image_patch.float()
            
            #extract features
            fc6,_ = self.net.extract_features_fc(image_patch)
            
            self.feats_ROI = normalize(fc6.detach().numpy())
            
            #return fc6.detach().numpy()
        
        elif self.repr_type==1:#sequences
            raise NotImplementedError
            
    
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
            #click = self.Label_showFrame.mapFromParent(event.pos())
            #print click_adjusted[0]
            #print click_adjusted[1]
            
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
        #load the detections
        self.detections = load_table(cfg.detection_file,asDict=False)
        videos = np.unique(self.detections['videos'].values)
        self.videos = [vid.encode(encod) for vid in videos]
        self.videos = np.array(self.videos)
        
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
        #if cfg.project=='humans':#only keep the '2kmh' videos
            #keep = []
            #for i,v in enumerate(self.videos):
                #if v=="021_2kmh":
                    #continue
                #if v.find('2km')!=-1:
                    #keep.append(i)
                    
            #self.videos = self.videos[keep]
        
        self.ComboBox_chooseVideo.addItem('Random')
        for i,v in enumerate(self.videos):
            if v[0]=='/': v=v[1:]
            self.ComboBox_chooseVideo.addItem('%d) %s'%(i+1,v))
    
    def get_all_feats(self):
        #first get the features of the postures (fc6)
        feat,frames,coords,vids = load_features('fc6',cfg.features_path,self.videos.tolist(),progress_bar=True,progress_bar_text='Extract Posture Features')
        self.feats = {'frames':frames,'coords': coords,'vids': vids,'features': normalize(feat)}
        #(problem when saving the coordinates during the extraction of the features)
        #if cfg.project=='humans':#set it manually, fix later!###############################################################################################################################################################################
            #coords = np.array([[0,0,400,400],]*coords.shape[0])
        
        self.nr_embedding_dataset = min(self.nr_embedding_dataset,feat.shape[0])#since 'Postures' is default
        
        #then get the features of the sequences (lstm)
        feat,frames,coords,vids = load_features('lstm',cfg.features_path,self.videos.tolist(),progress_bar=True,progress_bar_text='Extract Sequence Features')
        #(problem when saving the coordinates during the extraction of the features)
        #if cfg.project=='humans':#set it manually, fix later!###############################################################################################################################################################################
            #coords = np.array([[[0,0,-1,-1],]*coords.shape[1]]*coords.shape[0])
        
        self.feats_seqs = {'frames':frames,'coords': coords,'vids': vids,'features': normalize(feat)}
            
    
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
                if hasattr(self,'centroids') and self.RadioButton_showClusters.isChecked():
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
                if hasattr(self,'centroids_seqs') and self.RadioButton_showClusters.isChecked():
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
                            
                    
                    #include the current frame if custom ROI
                    #if not self.default_BB:
                        #c = [self.current_BB_pos[0]-self.BB_size[0]/2,
                                #self.current_BB_pos[1]-self.BB_size[1]/2,
                                #self.current_BB_pos[0]+self.BB_size[0]/2,
                                #self.current_BB_pos[1]+self.BB_size[1]/2]
                        
                        #clust_info_vid = clust_info_vid.append({'cluster id': self.low_dim_points_ROI_clusterlabel,
                                                                    #'video id': self.chosenVideoIdx,
                                                                    #'video directory': vid,
                                                                    #'frame': self.current_frame_number,
                                                                    #'x1':c[0], 'y1':c[1], 'x2':c[2],'y2':c[3],
                                                                    #'default roi': 'No'}, ignore_index=True)
                    
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
    
    def compute_NN(self):
        
        vid_query = self.chosenVideo
        
        #get the features of the query image
        if self.default_BB:
            idx = np.where(self.current_frame_number==self.current_feats['frames'])[0][0]
            feat_query = np.expand_dims(self.current_feats['features'][idx],axis=0)
        else:
            if not hasattr(self,'feats_ROI'): self.extract_features()
            feat_query = self.feats_ROI
        
        nr = min(100*self.n_mean_nn,self.feats['features'].shape[0])
        
        #get the nearest neighbors
        D = np.zeros((1,nr),np.float32)
        I = np.zeros((1,nr),np.int)
        dist = 1-np.dot(feat_query,self.feats['features'].T)[0]
        #dist = cdist(feat_query,self.feats['features'],metric='cosine')[0]
        sort_idx = np.argsort(dist)
        dist = dist[sort_idx]
        I = sort_idx[:nr]
        D = dist[:nr]
            
            
        #sort out the ones which belong to the same video
        nn_idx = []
        nn_dist = []
        vids_all = self.feats['vids']
        for i,d in zip(I,D):
            if vids_all[i]!=vid_query:
                nn_idx.append(i)
                nn_dist.append(d)
            
            if len(nn_idx)>=self.n_mean_nn:
                break
            
        return nn_idx, nn_dist
        
    def compute_NN_seqs(self):
        
        vid_query = self.chosenVideo
        
        #get the features of the query sequence
        feat_query = np.expand_dims(self.current_feats_seqs['features'][self.current_idx],axis=0)
        
        #get nearest neighbors
        nr = min(100*self.k,self.feats_seqs['features'].shape[0])
        
        D = np.zeros((1,nr),np.float32)
        I = np.zeros((1,nr),np.int)
        dist = 1-np.dot(feat_query,self.feats_seqs['features'].T)[0]
        #dist = cdist(feat_query,self.feats['features'],metric='cosine')[0]
        sort_idx = np.argsort(dist)
        dist = dist[sort_idx]
        I = sort_idx[:nr]
        D = dist[:nr]
        
        #sort out the ones which belong to the same video
        nn_idx = []
        nn_dist = []
        vids_all = self.feats_seqs['vids']
        for i,d in zip(I,D):
            if vids_all[i]!=vid_query:
                nn_idx.append(i)
                nn_dist.append(d)
            
            if len(nn_idx)>=self.n_mean_nn:
                break
        
        return nn_idx, nn_dist
    
    def update_coords(self,c,size):
        if c[2]==-1: c[2] = size[0]
        if c[3]==-1: c[3] = size[1]
        
        return c
        
    def show_NN_seqs(self,saveNN=False):
        #print self.k
        nn_idx, nn_dist = self.compute_NN_seqs()
        
        #plot the image with matplotlib, save it as image and then show this image in the interface
        nr_cols = cfg.seq_len
        nr_rows = self.k+2 if saveNN else self.k
        
        _=plt.close()
        fig = plt.figure(1,figsize=(cfg.seq_len,self.k))
        
        #plot also query if we want to save it
        if saveNN:
            gs = gridspec.GridSpec(nr_rows,nr_cols,height_ratios = [1,0.5]+np.ones((self.k,),np.int).tolist())
            gs.update(hspace=0.5,wspace=0.1)
            
            frames = self.current_feats_seqs['frames'][self.current_idx]
            #coords = self.current_feats_seqs['coords'][self.current_idx]
            vid = self.videos[self.chosenVideoIdx]
            
            for j,f in enumerate(frames):
                image_patch = Image.open('%s%s/%06d.jpg'%(cfg.crops_path,vid,f))
                #image = Image.open('%s/%s/%06d.jpg'%(cfg.frames_path,self.chosenVideo,f))
                #c = self.update_coords(c,image.size)
                #image_patch = image.crop(c)
                
                ax = plt.subplot(gs[0,j])
                _=ax.imshow(image_patch)
                _=ax.axis('off')
                vid_id = self.chosenVideoIdx+1
                
                if j==0:
                    _=ax.set_title('Query, Vid %d'%(vid_id),loc='left',fontsize=12)
            
            #plot a line between the query and the NN
            ax_line = plt.subplot(gs[1,:])
            _=ax_line.axhline(y=1,linewidth=2, color='k')
            _=ax_line.axis('off')
        else:
            gs = gridspec.GridSpec(nr_rows,nr_cols)
            gs.update(hspace=0.5,wspace=0.1)
        
        meta_data = []
        for i,(nn,d) in enumerate(zip(nn_idx[:self.k],nn_dist[:self.k])):
            vid = self.feats_seqs['vids'][nn]
            if vid[0]=='/': vid = vid[1:]
            frames = self.feats_seqs['frames'][nn]
            coords = self.feats_seqs['coords'][nn]
            
            meta_data_seq = []
            for j,(f,c) in enumerate(zip(frames,coords)):
                image = Image.open('%s%s/%06d.jpg'%(cfg.crops_path,vid,f))
                row = i+2 if saveNN else i
                col = j
                ax = plt.subplot(gs[row,col])
                _=ax.imshow(image)
                _=ax.axis('off')
                
                vid_id = np.where(self.videos==self.feats_seqs['vids'][nn])[0][0]+1
                if j==0:#set a title
                    _=ax.set_title('No. %d, Vid %d'%(i+1,vid_id),loc='left',fontsize=12)
                
                meta_data_seq.append([i,j,vid_id,vid,f,c,d])
                #ax.tight_layout()
            
            meta_data.append(meta_data_seq)
        
        fig.tight_layout()
        _=plt.savefig('NN.jpg',bbox_inches='tight')
        _=plt.close()
        
        self.plot_NN = fig
        self.meta_data = meta_data
        
        if not saveNN:#only update QLabel if we don't want to save it
            self.Label_showResult_title.setText('Nearest Neighbors of Current Sequence given all Sequences in Dataset')
            self.Label_showResult_title.setAlignment(Qt.AlignCenter)
            
            pixmap = QPixmap('NN.jpg')
            smaller_pixmap = pixmap.scaled(self.Label_showResult.frameGeometry().width(),
                                            self.Label_showResult.frameGeometry().height(),
                                            Qt.KeepAspectRatio,
                                            transformMode = Qt.FastTransformation)
            
            #change position of Label_showResult, so that it is in the middle of the title
            #mid = self.Label_showResult_title.geometry().left()+self.Label_showResult_title.geometry().width()/float(2)
                #self.PushButton_computeNN.geometry().top()+self.PushButton_computeNN.geometry().height()/float(2)]
            #self.Label_showResult.setGeometry(QRect(mid-smaller_pixmap.width()/float(2),
                                                #self.Label_showResult.geometry().top(),
                                                #smaller_pixmap.width(),
                                                #self.Label_showResult.geometry().height()))
            
            self.Label_showResult.setPixmap(smaller_pixmap)
    
    def show_NN(self,saveNN=False):
        nn_idx, nn_dist = self.compute_NN()
        
        #plot the image with matplotlib, save it as image and then show this image in the interface
        nr_cols = 6 if saveNN else 5
        nr_rows = np.ceil((self.k)/float(nr_cols-(2 if saveNN else 1))).astype(int)
        #nr_rows = np.ceil((self.k)/float(nr_cols-1)).astype(int)
        mean_nn = np.zeros((200,200,3),np.float64)#init mean image
        
        #if plt.fignum_exists(1):
        #    fig.clf()
        #else
        _=plt.close()
        fig = plt.figure(1,figsize=(2*nr_cols,2*nr_rows))
        #fig.clf()
        
        #plt.figure(self.plot_NN.number)#set it as current figure
        gs = gridspec.GridSpec(nr_rows,nr_cols,width_ratios=np.ones((nr_cols-1,),np.int).tolist()+[1.5])
        gs.update(hspace=0.5,wspace=0.2)
        
        #plot also query if we want to save it
        if saveNN:
            #get patch
            image = Image.open(cfg.frames_path+self.chosenVideo+'/'+self.current_frame_name)
            c = [self.current_BB_pos[0]-self.BB_size[0]/2,
                self.current_BB_pos[1]-self.BB_size[1]/2,
                self.current_BB_pos[0]+self.BB_size[0]/2,
                self.current_BB_pos[1]+self.BB_size[1]/2]
            image_patch = image.crop(c)
            ax = plt.subplot(gs[nr_rows/2,0])
            _=ax.imshow(image_patch)
            _=ax.axis('off')
            vid_id = self.chosenVideoIdx+1
            _=ax.set_title('Query\nVid %d'%(vid_id),fontsize=14)
        
        meta_data = []
        for i,(nn,d) in enumerate(zip(nn_idx[:self.k],nn_dist[:self.k])):
            vid = self.feats['vids'][nn]
            if vid[0]=='/': vid = vid[1:]
            frame = self.feats['frames'][nn]
            coords = self.feats['coords'][nn]
            
            image = Image.open('%s%s/%06d.jpg'%(cfg.crops_path,vid,frame))
            mean_nn += np.asarray(image.resize((200,200),Image.BILINEAR),np.float64)
            
            #_=plt.subplot(nr_rows,nr_cols,i+1)
            row = int(i/(nr_cols-2)) if saveNN else int(i/(nr_cols-1))
            col = int(i%float(nr_cols-2))+1 if saveNN else int(i%float(nr_cols-1))
            ax = plt.subplot(gs[row,col])
            _=ax.imshow(image)
            _=ax.axis('off')
            vid_id = np.where(self.videos==self.feats['vids'][nn])[0][0]+1
            _=ax.set_title('No. %d\nVid %d'%(i+1,vid_id), fontsize=14)
            
            meta_data.append([vid_id,vid,frame,coords,d])
        
        #now go on with computing the mean of the 100NN
        ax = plt.subplot(gs[nr_rows/2,-1])
        nn_found = i+1
        for i,(nn,d) in enumerate(zip(nn_idx[nn_found:],nn_dist[nn_found:])):
            vid = self.feats['vids'][nn]
            if vid[0]=='/': vid = vid[1:]
            frame = self.feats['frames'][nn]
            coords = self.feats['coords'][nn]
            vid_id = np.where(self.videos==self.feats['vids'][nn])[0][0]+1
            
            image = Image.open('%s%s/%06d.jpg'%(cfg.crops_path,vid,frame))
            
            mean_nn += np.asarray(image.resize((200,200),Image.BILINEAR),np.float64)
            nn_found += 1
            
            meta_data.append([vid_id,vid,frame,coords,d])
            
            if nn_found>=self.n_mean_nn:
                break
        
        mean_nn /= nn_found
        _=ax.imshow(mean_nn.astype(np.uint8))
        _=ax.axis('off')
        _=ax.set_title('Mean of \n %d NN'%(self.n_mean_nn),fontsize=14)
        
        _=plt.savefig('NN.jpg',bbox_inches='tight',dpi=150)
        _=plt.close()
        
        self.plot_NN = fig
        self.meta_data = meta_data
        
        if not saveNN:#only update QLabel if we don't want to save it
            self.Label_showResult_title.setText('Nearest Neighbors of Current Frame given all Postures in Dataset')
            self.Label_showResult_title.setAlignment(Qt.AlignCenter)
            
            pixmap = QPixmap('NN.jpg')
            smaller_pixmap = pixmap.scaled(self.Label_showResult.frameGeometry().width(),
                                        self.Label_showResult.frameGeometry().height(),
                                        Qt.KeepAspectRatio,
                                        transformMode = Qt.FastTransformation)
            
            
            #change position of Label_showResult, so that it is in the middle of the title
            #mid = self.Label_showResult_title.geometry().left()+self.Label_showResult_title.geometry().width()/float(2)
                #self.PushButton_computeNN.geometry().top()+self.PushButton_computeNN.geometry().height()/float(2)]
            #self.Label_showResult.setGeometry(QRect(mid-smaller_pixmap.width()/float(2),
                                                #self.Label_showResult.geometry().top(),
                                                #smaller_pixmap.width(),
                                                #self.Label_showResult.geometry().height()))
            
            
            self.Label_showResult.setPixmap(smaller_pixmap)
    
    def go_right(self):
        self.clear_info_inbetween()
        if self.last_result=='NN':
            self.Label_showResult.clear()
        
        #self.Label_info_inbetween.setText(' ')
        #self.Label_info_inbetween.setStyleSheet("")
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
                self.current_frame_name = self.crop_files[self.current_idx].encode(encod)
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
            
    
    
    #def clear_label_inbetween(self):
    #    self.Label_info_inbetween.setText('')
    #    self.Label_info_inbetween.setStyleSheet("")
    
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
            
            #self.label_showFrame.setFixedWidth(smaller_pixmap.width())
            
            #change position of label_showFrame, so that it is in the middle of the buttons
            #mid = self.Widget_chooseFrame.geometry().left()+(self.Widget_chooseFrame.geometry().right()-self.Widget_chooseFrame.geometry().left())/2
            #mid = self.PushButton_goLeft.geometry().right()+(self.PushButton_goRight.geometry().left()-self.PushButton_goLeft.geometry().right())/float(2)
            #mid = self.PushButton_computeNN.geometry().left()+self.PushButton_computeNN.geometry().width()/float(2)
            #self.Label_showFrame.setGeometry(QRect(mid-smaller_pixmap.width()/float(2),
                                                #self.Label_showFrame.geometry().top(),
                                                #smaller_pixmap.width(),
                                                #self.Label_showFrame.geometry().height()))
            
            #show the frame
            #self.Label_showFrame.setFixedWidth(smaller_pixmap.width())
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
            
            pos = coords[len(frames)/2]#coords of middle frame
            
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
                
                if i==len(frames)/2:#update pos if default values (-1) and then current_BB_pos
                    pos = self.update_coords(c,image_patch.size)
                    self.current_BB_pos = [pos[0]+(pos[2]-pos[0])/2,pos[1]+(pos[3]-pos[1])/2]#middle point
                
                #image_patch = image.crop(c)
                
                ax = plt.subplot(gs[1,i])
                ax.clear()
                _=ax.imshow(image_patch)
                
                #show that this is the main one
                if i==len(frames)/2:
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
    
    def get_middle_frames(self):
        middle_frames = []
        for i,f in enumerate(self.current_feats_seqs['frames']):
            middle_frames.append(f[len(f)/2])
            
        self.middle_frames = middle_frames
        
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
        
        self.current_feats = {'frames': frames, 'features': normalize(feat), 'coords': coords}
        
        #get the lstm features of the current video
        feat,frames,coords,vids = load_features('lstm',cfg.features_path,self.videos[self.chosenVideoIdx],progress_bar=False)
        
        #remove doubles
        _,indices = np.unique(frames,return_index=True,axis=0)
        feat,frames,coords,vids = feat[indices],frames[indices],coords[indices],vids[indices]
        self.current_feats_seqs = {'frames': frames, 'features': normalize(feat), 'coords': coords}
        
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
        self.Widget_method.setEnabled(True)
        #self.Widget_moveFrames.setEnabled(True)
        self.Widget_buttons_result.setEnabled(True)
        self.Widget_buttons_frame.setEnabled(True)
        self.Widget_chooseReprType.setEnabled(True)
        self.Frame_save.setEnabled(True)
        self.Frame_vidInfo.setEnabled(True)
        self.Label_frame.setEnabled(True)
        self.Label_showResult_title.setEnabled(True)
        if self.ComboBox_compute.currentIndex()==0:
            self.Frame_nrNN.setEnabled(True)
        elif self.ComboBox_compute.currentIndex()==1:
            self.Frame_embSource.setEnabled(True)
            
        self.Label_save_currentFolder.setText(self.save_dir)
        
        
        #show the first image
        self.draw_frame()
        
def main():
    app = QApplication(sys.argv)
    form = NNInterface()
    form.show()
    app.exec_()
    _=plt.close()
    if os.path.isfile('NN.jpg'): os.remove('NN.jpg')
    if os.path.isfile('tsne.jpg'): os.remove('tsne.jpg')
    if os.path.isfile('frame.jpg'): os.remove('frame.jpg')
    
    
if __name__ == '__main__':              # if we're running file directly and not importing it
    main()   



#app = QApplication(sys.argv)
#form = NNInterface()
#form.show()
#_=plt.close()
#os.remove('NN.jpg')
#os.remove('tsne.jpg')
#os.remove('frame.jpg')