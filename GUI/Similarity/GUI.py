import sys
import os
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

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont, QMouseEvent, QImage
from PyQt5.QtCore import Qt,QRect, QCoreApplication
from PyQt5 import QtTest

import torch
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"]=""#make sure that it is run on the cpu
sys.path.append("features/pytorch")
from model import CaffeNet

from utils import load_table, load_features, transform_batch
#from glob import glob
sys.path.append('./GUI/Similarity/')
import design3 as design

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-cf", "--config",type=str,default='config_pytorch',
                    help="Define config file")
args = parser.parse_args()

import importlib

cfg = importlib.import_module(args.config)

def save_umap(umap,file):
    for attr in ["_tree_init", "_search", "_random_init"]:
        if hasattr(umap, attr):
            delattr(umap, attr)
    #return 
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
        #scaledSize = self.originalPixmap.size()
        #scaledSize.scale(self.size(), Qt.KeepAspectRatio)
        #scaledSize.scale(self.size(), Qt.KeepAspectRatio)
        #if not self.pixmap() or scaledSize != self.pixmap().size():
        #    self.updateLabel()
       
    #def updateLabel(self):
    #    self.setPixmap(self.originalPixmap.scaled(
    #        self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class NNInterface(QMainWindow, design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(NNInterface, self).__init__(parent)
        print('Initialize Interface...')
        self.setupUi(self)
        #self.centralwidget.showMaximized()
        self.fill_comboBox()
        self.nr_embedding_dataset = 5000
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
        self.nr_clusters = 5
        self.repr_type = str(self.ComboBox_chooseReprType.currentText())
        
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
        if str(self.ComboBox_chooseReprType.currentText) != self.repr_type:
            self.repr_type = str(self.ComboBox_chooseReprType.currentText())
            self.clear_result_labels()
            
            if self.repr_type=='Sequences':
                self.Frame_adjustBB.setEnabled(False)
                self.Label_frame.setText('3. Choose Middle Frame of Sequence')
                self.nr_embedding_dataset = min(self.nr_embedding_dataset,self.feats_sequences['features'].shape[0])####################################################
            else:
                self.Frame_adjustBB.setEnabled(True)
                self.Label_frame.setText('3. Choose Frame and Area of Interest (ROI)')
                self.nr_embedding_dataset = min(self.nr_embedding_dataset,self.feats['features'].shape[0])################################################################
                
            self.Label_frame.setStyleSheet("font-size:12pt; font-weight:600;")
            
    def activate_clusters(self):
        if self.RadioButton_showClusters.isChecked():
            self.Widget_nrClusters.setEnabled(True)
        else:
            self.Widget_nrClusters.setEnabled(False)
    
    def clear_result_labels(self):
        self.Label_showResult.clear()
        self.Label_info_inbetween.clear()
        
    def clear_all_labels(self):
        self.Label_showFrame.clear()
        self.Label_showResult.clear()
        self.Label_info_inbetween.clear()
    
    def change_dir(self):
        #vid = str(self.chosenVideo).replace('/','_').replace(' ','_')
        #tempfilename = '%s_%s_frame%06d'%(self.last_result,vid,self.current_frame_number)
        fileDialog = QFileDialog()
        self.save_dir = str(fileDialog.getExistingDirectory(directory=self.save_dir))
        self.Label_save_currentFolder.setText(self.save_dir)
            #self.save_dir = str(fileDialog.getExistingDirectory(directory=os.sep.join((os.path.expanduser('~'), 'Documents'))))
        
    def adjust_BB_directly(self):
        self.BB_size = [self.SpinBox_BBSize_x.value(),self.SpinBox_BBSize_y.value()]
        self.draw_frame(pos=[self.current_BB_pos[0]-self.BB_size[0]/2,
                            self.current_BB_pos[1]-self.BB_size[1]/2,
                            self.current_BB_pos[0]+self.BB_size[0]/2,
                            self.current_BB_pos[1]+self.BB_size[1]/2])
    
    #def activate_emb_onlyVideo(self):
    #    if hasattr(self,'low_dim_points_vid'):
    #        self.show_low_dim_embedding()#update the embedding
    
    def method(self):
        self.Label_showResult.clear()
        self.Label_info_inbetween.clear()
        self.Label_showResult_title.setText('Result')
        self.Label_showResult_title.setAlignment(Qt.AlignCenter)
        
        if self.ComboBox_compute.currentIndex()==1:
            self.Frame_nrNN.setEnabled(False)
            self.Frame_embSource.setEnabled(True)
            if not IS_UMAP:#if we want the low-dim embedding, but don't have UMAP, we cannot map the new detection
                self.Frame_adjustBB.setEnabled(False)
                self.Label_info_inbetween.setText('Adjusting Area of Interest not available with TSNE as Low Dimensionality Embedding, only with UMAP')
                self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
                self.Label_info_inbetween.setStyleSheet("QLabel { background-color : blue; color : white; }")
                
            
        else:
            self.Frame_adjustBB.setEnabled(True)
            self.Frame_nrNN.setEnabled(True)
            self.Frame_embSource.setEnabled(False)
    
    def compute(self):
        if self.ComboBox_compute.currentIndex()==0:
            self.show_NN()
        else:
            self.show_low_dim_embedding()
        
    def pause(self):
        self.goOn = False
        self.Frame_adjustBB.setEnabled(True)
        
    def play(self):
        self.default_BB = True
        self.goOn = True
        self.RadioButton_adjustBB.setChecked(False)
        self.Frame_adjustBB.setEnabled(False)
        
        while self.goOn:
            if self.current_frame_idx == len(self.crop_files)-1:
                #start again from the beginning
                self.current_frame_idx = 0
            else:
                self.current_frame_idx += 1
                
            self.current_frame_name = self.crop_files[self.current_frame_idx]
            self.current_frame_number = int(''.join(c for c in self.current_frame_name if c.isdigit()))
            
            self.draw_frame()
            
            t1 = time.time()
            self.compute()
            if self.RadioButton_automaticSaving.isChecked():
                self.save_result(showInfo = False)#don't show it every single time
            
            t2 = time.time()
            t = t2-t1
            
            QtTest.QTest.qWait((float(1)/self.DoubleSpinBox_secPerFrame.value()-t)*1000)
            
            #stop if we are at the end of the video, if the user wants that
            if self.RadioButton_play_automaticStop.isChecked() and self.current_frame_idx == len(self.crop_files)-1:
                self.goOn = False
            
            QCoreApplication.processEvents()
        
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
                self.Label_info_inbetween.setText('Compute Low Dimensionality Embedding with UMAP for Dataset.....')
                self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
                self.Label_info_inbetween.setStyleSheet("QLabel { background-color : blue; color : white; }")
                QtTest.QTest.qWait(2*1000)
            
                #compute embedding and save it
                if not os.path.isfile(file_name+'_umap.sav'):
                    
                    #collect randomly self.nr_embedding_dataset frames for creating the low-dim embedding
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
                
            if self.RadioButton_showClusters.isChecked() and self.nr_clusters!=self.SpinBox_nrClusters.value():
                self.nr_clusters = self.SpinBox_nrClusters.value()
                
                self.Label_info_inbetween.setText('Compute Clustering.....')
                self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
                QtTest.QTest.qWait(2*1000)
                
                #do clustering
                if not os.path.isfile('%s_kmeans%04d.sav'%(file_name,self.nr_clusters)):
                    if 'feats_norm' not in locals():
                        feats_norm = self.feats['features'][self.feats_idx ,:]
                    
                    kmeans = KMeans(n_clusters=self.nr_clusters, random_state=0).fit(feats_norm)
                    pickle.dump(kmeans,open('%s_kmeans%04d.sav'%(file_name,self.nr_clusters),'wb'))
                    #save the centroids
                    centroid_idx = np.array(self.feats_idx)[np.argmin(cdist(feats_norm, self.kmeans.cluster_centers_, 'euclidean'), axis=0).astype(int)]
                    pickle.dump(centroid_idx,open('%s_kmeans%04d_centroids.sav'%(file_name,self.nr_clusters),'wb'))
                    
                
                #load clusters
                self.kmeans = pickle.load(open('%s_kmeans%04d.sav'%(file_name,self.nr_clusters),'rb'))
                self.low_dim_points_all_clusterlabel = self.kmeans.labels_
                self.centroids = pickle.load(open('%s_kmeans%04d_centroids.sav'%(file_name,self.nr_clusters),'rb'))
            
                #reset text
                self.Label_info_inbetween.clear()
                
                    
            #get embedding for current video
            if not hasattr(self,'low_dim_points_vid'):
                #set text
                self.Label_info_inbetween.setText('Compute Low Dimensionality Embedding with UMAP for Current Video.....')
                self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
                self.Label_info_inbetween.setStyleSheet("QLabel { background-color : blue; color : white; }")
                QtTest.QTest.qWait(2*1000)
                
                #compute
                feats_vid = self.current_feats['features']
                self.low_dim_points_vid = self.low_dim_method.transform(feats_vid)
                
                self.low_dim_points_vid_clusterlabel = self.kmeans.predict(feats_vid)
                #self.low_dim_points_all_clusterlabel = self.kmeans.labels_[:self.low_dim_points_all.shape[0]]
                
                #reset label
                self.Label_info_inbetween.clear()
                
                
            #get embedding for current frame
            if self.default_BB:
                idx = np.where(self.current_feats['frames']==self.current_frame_number)[0]
                self.low_dim_points_ROI = [self.low_dim_points_vid[idx,0],self.low_dim_points_vid[idx,1]]
                self.low_dim_points_ROI_clusterlabel = int(self.low_dim_points_vid_clusterlabel[idx])
            else:
                feat_ROI = normalize(self.extract_features())
                self.low_dim_points_ROI = self.low_dim_method.transform(feat_ROI)[0]
                self.low_dim_points_ROI_clusterlabel = self.kmeans.predict(feat_ROI)
            
        else:#use TSNE
            self.Label_info_inbetween.setText('Not Implemented Yet, only works with UMAP!')
            self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
            self.Label_info_inbetween.setStyleSheet("QLabel { background-color : blue; color : white; }")
        
    def show_low_dim_embedding(self):
        self.last_result = 'embedding'
        self.compute_low_dim_embedding()
        
        if plt.fignum_exists(2):
            fig2.clf()
        else:
            fig2 = plt.figure(2,figsize=(15,10))
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
            h_frame,=ax_tsne.plot(self.low_dim_points_ROI[0],self.low_dim_points_ROI[1],'*',color=(0.6,0.6,0.6,1),markersize=35,markeredgewidth=1,markeredgecolor=colors[clusterlabel])
            
            #set legend
            if not self.RadioButton_embedding.isChecked():
                lgd=ax_tsne.legend((h_dataset,h_video,h_frame),
                                ('%i Postures of Dataset'%(self.nr_embedding_dataset),'Posture of Current Video','Posture of Current ROI'))#loc=2,prop={'size': 6})
                lgd.legendHandles[2]._legmarker.set_markersize(15)
            else:
                lgd=ax_tsne.legend((h_video,h_frame),
                            ('Postures of Current Video','Posture of Current ROI'))#loc=2,prop={'size': 6})
                lgd.legendHandles[1]._legmarker.set_markersize(15)
            
            
            ax_tsne.set_xlim(x_lim)
            ax_tsne.set_ylim(y_lim)
            _=plt.axis('off')
            
            #set title for the centroids
            ax_title = plt.subplot(gs[1,2:])
            _=ax_title.set_title('Centroids of Clusters')
            _=ax_title.axis('off')
            
            #show query
            ax_query_title = plt.subplot(gs[1,0])
            _=ax_query_title.set_title('Query')
            _=ax_query_title.axis('off')
            
            ax_query = plt.subplot(gs[2,0])
            image = Image.open(cfg.frames_path+str(self.chosenVideo)+'/'+self.current_frame_name)
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
            
            #plot the centroids and save the information about the centroids in meta data
            meta_data = []
            for i,c in enumerate(self.centroids):
                vid = str(self.feats['vids'][c])
                if vid[0]=='/': vid = vid[1:]
                frame = self.feats['frames'][c]
                coords = self.feats['coords'][c]
                
                image = Image.open('%s%s/%06d.jpg'%(cfg.crops_path,str(vid),frame))
                
                ax = plt.subplot(gs[2,i+2])
                _=ax.imshow(image)
                #put a border around the image!
                rect = patches.Rectangle((0,0),image.size[0],image.size[1],edgecolor=colors[i],fill=False,linewidth=self.Label_showResult.width()/150)
                _=ax.add_patch(rect)
                _=plt.axis('off')
                #_=ax.set_title('%d. Centroid'%(i+1))
                
                vid_id = np.where(self.videos==self.feats['vids'][c])[0][0]
                meta_data.append([vid_id,vid,frame,coords])
                
            self.meta_data = meta_data
        
        #without clusters
        else:
            ax_tsne = plt.subplot(gs[0,:])
            
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
            else:
                lgd=ax_tsne.legend((h_video,h_frame),
                            ('Postures of Current Video','Postures of Current ROI'))#loc=2,prop={'size': 6})
                
            
            ax_tsne.set_xlim(x_lim)
            ax_tsne.set_ylim(y_lim)
            
            _=plt.axis('off')  
            
        rcParams.update({'font.size': 18})
        self.plot_embedding = fig2
        
        #_=plt.savefig('tsne.jpg',bbox_inches='tight')
        _=plt.savefig('tsne.jpg',bbox_extra_artists=(lgd,),bbox_inches='tight',dpi=150)
        _=plt.close()
        
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
        
        if self.repr_type=='Postures':
            #get patch
            image = Image.open(cfg.frames_path+str(self.chosenVideo)+'/'+self.current_frame_name)
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
            
            #self.feats_ROI = fc6.detach().numpy()
            
            return fc6.detach().numpy()
        
        elif self.repr_type=='Sequences':
            pass
            
    
    def activate_adjust_BB(self):
        if self.RadioButton_adjustBB.isChecked():
            #self.Frame_adjustBB.setEnabled(True)
            self.Label_info_inbetween.setText('Please click on the desired middle point for the Bounding Box')
            self.Label_info_inbetween.setStyleSheet("QLabel { background-color : blue; color : white; }")
            self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
            self.default_BB = False#set it to false, so that we know that we have to compute the features from scratch
            self.Widget_changeBBSize.setEnabled(True)
        else:
            self.Label_info_inbetween.clear()
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
    
    
    def fill_comboBox(self):
        #load the detections
        self.detections = load_table(cfg.detection_file,asDict=False)
        self.videos = np.unique(self.detections['videos'].values)
        
        self.ComboBox_chooseVideo.addItem('Random')
        for i,v in enumerate(self.videos):
            if v[0]=='/': v=v[1:]
            self.ComboBox_chooseVideo.addItem('%d) %s'%(i+1,str(v)))
    
    def get_all_feats(self):
        #first get the features of the postures (fc6)
        feat,frames,coords,vids = load_features('fc6',cfg.features_path,self.videos.tolist(),progress_bar=True,progress_bar_text='Extract Posture Features')
        self.feats = {'frames':frames,'coords': coords,'vids': vids,'features': normalize(feat)}
        self.nr_embedding_dataset = min(self.nr_embedding_dataset,feat.shape[0])#since 'Postures' is default
        
        #then get the features of the sequences (lstm)
        feat,frames,coords,vids = load_features('lstm',cfg.features_path,self.videos.tolist(),progress_bar=True,progress_bar_text='Extract Sequence Features')
        self.feats_sequences = {'frames':frames,'coords': coords,'vids': vids,'features': feat,'features_norm': normalize(feat)}
            
    
    def save_result(self,showInfo = True):
        if (hasattr(self, 'plot_NN') and self.last_result == 'NN') or (hasattr(self, 'plot_embedding') and self.last_result == 'embedding'):
            vid = str(self.videos[self.chosenVideoIdx])#.replace('/','_').replace(' ','_')
            if vid[0]=='/': vid = vid[1:]
            folder = self.save_dir+'/'+vid
            if not os.path.isdir(folder):
                os.makedirs(folder)
            
            
            #tempfilename = '%s_%s_frame%06d'%(self.last_result,vid,self.current_frame_number)
            #fileDialog = QFileDialog()
            #filename,suffix = fileDialog.getSaveFileName(directory=os.sep.join((os.path.expanduser('~'), 'Documents',tempfilename)),filter="png;;jpg")
            
            if self.last_result=='NN':
                filename = '%s/%s_frame%06d'%(folder,self.last_result,self.current_frame_number)
                #filename = str("%s.%s"%(filename,suffix))
                
                #first draw the plot with query
                self.show_NN(saveNN=True)
                #then save it
                self.plot_NN.savefig(filename+'.png',bbox_inches='tight')
                
                #then save meta data in a csv file
                NN_info = pd.DataFrame(index=None, columns=['No. NN','video id','video name','frame','x1','y1','x2','y2'])
                
                for i,(v_id,vn,f,c) in enumerate(self.meta_data):
                    NN_info = NN_info.append({'No. NN': i+1, 'video id': v_id,'video directory': vn,'frame': f,'x1':c[0], 'y1':c[1], 'x2':c[2],'y2':c[3]}, ignore_index=True)
                
                NN_info.to_csv(filename+'.csv')
                
                if showInfo:
                    self.Label_info_inbetween.setText('Plot and Meta Data saved as ".png" and ".csv", respectively under %s'%(folder))
                
            
            elif self.last_result=='embedding':
                #1. save the plot
                filename_plot = '%s/embedding_dataset_%i_kmeans%04d_frame%06d.png'%(folder,self.nr_embedding_dataset,self.nr_clusters,self.current_frame_number)
                self.plot_embedding.savefig(filename_plot,bbox_inches='tight')
                
                #2. save the cluster information with all necessary information
                #3. save the cluster labels of all detections of the current video and also of the frame if custom bounding box
                if hasattr(self,'centroids'):
                    filename_clust_info = '%s/embedding_dataset_%i_kmeans%04d_clusters_info.csv'%(self.save_dir,self.nr_embedding_dataset,self.nr_clusters)
                    #####2.
                    if not os.path.isfile(filename_clust_info):
                        clust_info = pd.DataFrame(index=None, columns=['cluster id','video id','video directory','frame','x1','y1','x2','y2'])
                        
                        
                        for i,(v_id,vid,f,c) in enumerate(self.meta_data):
                            clust_info = clust_info.append({'cluster id': i,
                                                            'video id': v_id,
                                                            'video directory': vid,
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
                            clust_info_vid = clust_info_vid.append({'cluster id': l,
                                                                  'video id': self.chosenVideoIdx,
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
                        
                        clust_info_vid = clust_info_vid.append({'cluster id': self.low_dim_points_ROI_clusterlabel,
                                                                    'video id': self.chosenVideoIdx,
                                                                    'video directory': vid,
                                                                    'frame': self.current_frame_number,
                                                                    'x1':c[0], 'y1':c[1], 'x2':c[2],'y2':c[3],
                                                                    'default roi': 'No'}, ignore_index=True)
                    
                    #save
                    #clust_info_vid = clust_info_vid.set_index('frame')
                    clust_info_vid.to_csv(filename_clust_info_vid, index=False)
                        
                    #####End 3.
                    
                    if showInfo:
                        self.Label_info_inbetween.setText(
                            'Saved the following information: \n - General cluster information under %s \n - Plot under %s \n - Video cluster information %s'
                            %(filename_plot,filename_clust_info,filename_clust_info_vid))
                        
                elif showInfo:
                    self.Label_info_inbetween.setText('Result saved as ".png" under %s'%(filename))
                
            
            if showInfo:
                self.Label_info_inbetween.setStyleSheet("QLabel { background-color : blue; color : white; }")
                self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
        else:
            self.Label_info_inbetween.setText('No results produced so far!')
            self.Label_info_inbetween.setStyleSheet("QLabel { background-color : blue; color : white; }")
            self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
            
            QtTest.QTest.qWait((2)*1000)
            
            self.Label_info_inbetween.clear()
            
    
    def compute_NN(self):
        
        if self.default_BB:
            #update number of Nearets Neighbors
            self.k = self.SpinBox_chooseNumberNN.value()
            
            #get the features of the query image
            idx = np.where(self.current_frame_number==self.current_feats['frames'])[0][0]
            feat_query = np.expand_dims(self.current_feats['features'][idx],axis=0)
        else:
            feat_query = normalize(self.extract_features())
        
        vid_query = self.chosenVideo
        
        nr = min(100*self.n_mean_nn,self.feats['features'].shape[0])
        
        #get the nearest neighbors
        D = np.zeros((1,nr),np.float32)
        I = np.zeros((1,nr),np.int)
        dist = 1-np.dot(feat_query,self.feats['features_norm'].T)[0]
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
    
    
        
    def show_NN(self,saveNN=False):
        self.last_result = 'NN'
        nn_idx, nn_dist = self.compute_NN()
        
        #plot the image with matplotlib, save it as image and then show this image in the interface
        nr_cols = 6 if saveNN else 5
        nr_rows = np.ceil((self.k)/float(nr_cols-1)).astype(int)
        mean_nn = np.zeros((200,200,3),np.float64)#init mean image
        
        if plt.fignum_exists(1):
            fig.clf()
        else:
            fig = plt.figure(1)
        
        #plt.figure(self.plot_NN.number)#set it as current figure
        gs = gridspec.GridSpec(nr_rows,nr_cols,width_ratios=np.ones((nr_cols-1,),np.int).tolist()+[1.5])
        gs.update(hspace=0.5,wspace=0.2)
        
        #plot also query if we want to save it
        if saveNN:
            #get patch
            image = Image.open(cfg.frames_path+str(self.chosenVideo)+'/'+self.current_frame_name)
            c = [self.current_BB_pos[0]-self.BB_size[0]/2,
                self.current_BB_pos[1]-self.BB_size[1]/2,
                self.current_BB_pos[0]+self.BB_size[0]/2,
                self.current_BB_pos[1]+self.BB_size[1]/2]
            image_patch = image.crop(c)
            ax = plt.subplot(gs[nr_rows/2,0])
            _=ax.imshow(image_patch)
            _=ax.axis('off')
            vid_id = self.chosenVideoIdx
            _=ax.set_title('Query\nVid %d'%(vid_id))
            
        
        meta_data = []
        for i,nn in enumerate(nn_idx[:self.k]):
            vid = str(self.feats['vids'][nn])
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
            vid_id = np.where(self.videos==self.feats['vids'][nn])[0][0]
            _=ax.set_title('No. %d\nVid %d'%(i+1,vid_id))
            
            meta_data.append([vid_id,vid,frame,coords])
        
        #now go on with computing the mean of the 100NN
        ax = plt.subplot(gs[nr_rows/2,-1])
        nn_found = i+1
        for i,nn in enumerate(nn_idx[nn_found:]):
            vid = str(self.feats['vids'][nn])
            if vid[0]=='/': vid = vid[1:]
            frame = self.feats['frames'][nn]
            coords = self.feats['coords'][nn]
            vid_id = np.where(self.videos==self.feats['vids'][nn])[0][0]
            
            image = Image.open('%s%s/%06d.jpg'%(cfg.crops_path,vid,frame))
            
            mean_nn += np.asarray(image.resize((200,200),Image.BILINEAR),np.float64)
            nn_found += 1
            
            meta_data.append([vid_id,vid,frame,coords])
            
            if nn_found>=self.n_mean_nn:
                break
        
        mean_nn /= nn_found
        _=ax.imshow(mean_nn.astype(np.uint8))
        _=ax.axis('off')
        _=ax.set_title('Mean of %d NN'%(self.n_mean_nn))
        
        _=plt.savefig('NN.jpg',bbox_inches='tight')
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
            mid = self.Label_showResult_title.geometry().left()+self.Label_showResult_title.geometry().width()/float(2)
                #self.PushButton_computeNN.geometry().top()+self.PushButton_computeNN.geometry().height()/float(2)]
            self.Label_showResult.setGeometry(QRect(mid-smaller_pixmap.width()/float(2),
                                                self.Label_showResult.geometry().top(),
                                                smaller_pixmap.width(),
                                                self.Label_showResult.geometry().height()))
            
            
            self.Label_showResult.setPixmap(smaller_pixmap)
        
    
    def go_right(self):
        self.Label_info_inbetween.clear()
        if self.last_result=='NN':
            self.Label_showResult.clear()
        
        #self.Label_info_inbetween.setText(' ')
        #self.Label_info_inbetween.setStyleSheet("")
        self.RadioButton_adjustBB.setChecked(False)
        
        if (self.current_frame_idx+1)<len(self.crop_files):
            self.current_frame_idx += 1
            self.current_frame_name = str(self.crop_files[self.current_frame_idx])
            self.current_frame_number = int(''.join(c for c in self.current_frame_name if c.isdigit()))
            
            self.draw_frame()
        else:
            self.Label_info_inbetween.setText('Reached the end of the video')
            self.Label_info_inbetween.setStyleSheet("QLabel { background-color : blue; color : white; }")
            self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
    
    def go_left(self):
        self.Label_info_inbetween.clear()
        if self.last_result=='NN':
            self.Label_showResult.clear()
        
        #self.Label_info_inbetween.setText(' ')
        #self.Label_info_inbetween.setStyleSheet("")
        self.RadioButton_adjustBB.setChecked(False)
        
        if (self.current_frame_idx-1)>=0:
            self.current_frame_idx -= 1
            self.current_frame_name = str(self.crop_files[self.current_frame_idx])
            self.current_frame_number = int(''.join(c for c in self.current_frame_name if c.isdigit()))
            
            self.draw_frame()
        else:
            self.Label_info_inbetween.setText('Reached the beginning of the video')
            self.Label_info_inbetween.setStyleSheet("QLabel { background-color : blue; color : white; }")
            self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
        
    def go_rightright(self):
        self.Label_info_inbetween.clear()
        if self.last_result=='NN':
            self.Label_showResult.clear()
        
        #self.Label_info_inbetween.setText(' ')
        #self.Label_info_inbetween.setStyleSheet("")
        self.RadioButton_adjustBB.setChecked(False)
        
        if self.current_frame_idx!=(len(self.crop_files)-1):
            add = min(5,len(self.crop_files)-1-self.current_frame_idx)
            self.current_frame_idx += add
            self.current_frame_name = str(self.crop_files[self.current_frame_idx])
            self.current_frame_number = int(''.join(c for c in self.current_frame_name if c.isdigit()))
            
            self.draw_frame()
        else:
            self.Label_info_inbetween.setText('Reached the end of the video')
            self.Label_info_inbetween.setStyleSheet("QLabel { background-color : blue; color : white; }")
            self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
        
        
    def go_leftleft(self):
        self.Label_info_inbetween.clear()
        if self.last_result=='NN':
            self.Label_showResult.clear()
        
        #self.Label_info_inbetween.setText(' ')
        #self.Label_info_inbetween.setStyleSheet("")
        self.RadioButton_adjustBB.setChecked(False)
        
        if self.current_frame_idx!=0:
            subtract = min(self.current_frame_idx,5)
            self.current_frame_idx -= subtract
            self.current_frame_name = str(self.crop_files[self.current_frame_idx])
            self.current_frame_number = int(''.join(c for c in self.current_frame_name if c.isdigit()))
            
            self.draw_frame()
        else:
            self.Label_info_inbetween.setText('Reached the beginning of the video')
            self.Label_info_inbetween.setStyleSheet("QLabel { background-color : blue; color : white; }")
            self.Label_info_inbetween.setAlignment(Qt.AlignCenter)
    
    
    #def clear_label_inbetween(self):
    #    self.Label_info_inbetween.setText('')
    #    self.Label_info_inbetween.setStyleSheet("")
        
    def draw_frame(self,pos=[]):
        
        self.Label_info_inbetween.clear()
        self.Label_showResult.clear()
        
        if not pos:
            #find the detection
            det_idx = np.where(self.current_dets['frames'].values==int(self.current_frame_name[:self.current_frame_name.rfind('.')]))[0][0]
            #if len(det_idx)>1: det_idx = det_idx[0]
            pos = [self.current_dets.at[det_idx,'x1'],
                self.current_dets.at[det_idx,'y1'],
                self.current_dets.at[det_idx,'x2'],
                self.current_dets.at[det_idx,'y2']]
            
            self.current_BB_pos = [pos[0]+(pos[2]-pos[0])/2,pos[1]+(pos[3]-pos[1])/2]#middle point
            self.default_BB = True
            penRectangle = QPen(Qt.red)
        else:
            penRectangle = QPen(Qt.blue)
            
            #self.SpinBox_BBSize_x.setValue(100)
            #self.SpinBox_BBSize_y.setValue(100)
        
        #get the image
        pixmap = QPixmap.fromImage(QImage(cfg.frames_path+str(self.chosenVideo)+'/'+self.current_frame_name))
        #pixmap = QPixmap(cfg.frames_path+str(self.chosenVideo)+'/'+self.current_frame_name)
        self.frame_orig_size = [pixmap.width(),pixmap.height()]
        
        ## draw the detection on top
        recPainter = QPainter(pixmap)
        # set rectangle color and thickness
        #penRectangle = QPen(Qt.red)
        penRectangle.setWidth(3)
        # draw rectangle on painter
        recPainter.setPen(penRectangle)
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
        #self.Label_showFrame.setStyleSheet("")
        
        #self.is_pixmap = True
            
    def load_video(self):
        
        self.clear_all_labels()
        
        if hasattr(self,'low_dim_points_vid'): delattr(self,'low_dim_points_vid')#becaues the embedding points need to be computed again for the new video
        
        #first get the chosen video
        self.chosenVideoIdx = self.ComboBox_chooseVideo.currentIndex() 
        if self.chosenVideoIdx==0:
            self.chosenVideoIdx = random.sample(range(len(self.videos)),1)[0]
            self.ComboBox_chooseVideo.setCurrentIndex(self.chosenVideoIdx+1)
        else:
            self.chosenVideoIdx-=1#because "Random" is the first position
        
        self.chosenVideo = self.videos[self.chosenVideoIdx]
        if os.path.isdir(cfg.frames_path+str(self.chosenVideo)+'/deinterlaced'):
            self.chosenVideo = self.chosenVideo+'/deinterlaced'
        
        #save the files available in the folder
        self.crop_files = sorted(os.listdir(cfg.crops_path+self.videos[self.chosenVideoIdx]))#here are all the files we have detections for
        #self.frame_files = sorted(os.listdir(cfg.frames_path+self.chosenVideo))
        
        #save the detections belonging to these files
        idx_vid = np.where(self.detections['videos'].values==self.videos[self.chosenVideoIdx])[0]
        
        self.current_dets = pd.DataFrame({'frames':self.detections['frames'].values[idx_vid],
                                          'x1':self.detections['x1'].values[idx_vid],
                                          'y1':self.detections['y1'].values[idx_vid],
                                          'x2':self.detections['x2'].values[idx_vid],
                                          'y2':self.detections['y2'].values[idx_vid],
                                          })
        
        #self.current_dets = self.current_dets.set_index('frames',inplace=True)
        
        #set the first frame
        self.current_frame_idx = 0
        self.current_frame_name = self.crop_files[self.current_frame_idx]
        self.current_frame_number = int(''.join(c for c in self.current_frame_name if c.isdigit()))
        #self.current_frame_name = str(self.crop_files[self.current_frame_idx])#'%06d'%(self.current_dets['frames'].values[self.current_frame_idx])
        
        #load the first image
        #image = Image.open(cfg.frames_path+self.chosenVideo+'/'+self.current_frame_name)
        
        #show the first image
        self.draw_frame()
        
        
        #get the fc6 features of the current video
        feat,frames,coords,vids = load_features('fc6',cfg.features_path,str(self.videos[self.chosenVideoIdx]),progress_bar=False)
        self.current_feats = {'frames': frames, 'features': normalize(feat), 'coords': coords}
        
        #write the chosen video in info label
        self.Label_info_video.setText('Current Video: %s \n #Frames: %i'%(str(self.videos[self.chosenVideoIdx]),len(self.crop_files)))
        
        #activate all widgets
        #self.centralwidget.setEnabled(True)
        #for child in self.centralwidget.children():
        #    if child.objectName().find('Layout'):
        #        for childchild in child.children():
        #            if hasattr(childchild,'setEnabled'):
        #                childchild.setEnabled(True)
        #    
        #    if hasattr(child,'setEnabled'):
        #        child.setEnabled(True)
        
        #activate specific windows
        if self.repr_type=='Postures':
            self.Frame_adjustBB.setEnabled(True)
        
        self.Widget_changeBBSize.setEnabled(False)
        self.Frame_play.setEnabled(True)
        self.Widget_method.setEnabled(True)
        #self.Widget_moveFrames.setEnabled(True)
        self.Widget_buttons_result.setEnabled(True)
        self.Widget_buttons_frame.setEnabled(True)
        self.Widget_chooseReprType.setEnabled(True)
        #self.PushButton_goRight.setEnabled(True)
        #self.PushButton_goRightRight.setEnabled(True)
        #self.PushButton_goLeft.setEnabled(True)
        #self.PushButton_goLeftLeft.setEnabled(True)
        if self.ComboBox_compute.currentIndex()==0:
            self.Frame_nrNN.setEnabled(True)
        elif self.ComboBox_compute.currentIndex()==1:
            self.Frame_embSource.setEnabled(True)
            
        self.Label_save_currentFolder.setText(self.save_dir)
        
        #except of these widgets
        #self.Widget_changeBBSize.setEnabled(False)
        #if self.ComboBox_compute.currentIndex()==0:
        #    self.Frame_embSource.setEnabled(False)
        #else:
        #    self.Frame_nrNN.setEnabled(False)
        
        
#def main():
    #app = QApplication(sys.argv)
    #form = NNInterface()
    #form.show()
    #app.exec_()
    
    
#if __name__ == '__main__':              # if we're running file directly and not importing it
    #main()   



app = QApplication(sys.argv)
form = NNInterface()
form.show()
os.remove('NN.jpg')
os.remove('tsne.jpg')
#app.exec_()


#import sys
#from PyQt5.QtWidgets import QApplication, QWidget
#from PyQt5.QtGui import QIcon
#import firstTrial


#app = QApplication(sys.argv)

#w = QWidget()
#w.setGeometry(50,50,1000,500)
#w.setWindowTitle('Get Closest Neighbors')
#w.setWindowIcon(QIcon("hciLogoSmall.png"))


#w.show()

#sys.exit(app.exec_())


   #script_dir = os.path.dirname(os.path.realpath('__file__'))
        #videos_path = script_dir+'/../../input_output/rats/preprocessing/crops/'
        #videos,videos_small = [],[]#one so that we know the location, and "videos_small" for showing in the ComboBox
        #for root, subFolders,_ in os.walk(videos_path):
        #    if len(videos)>10:######################################################################################################################
        #        break
        #    
        #    if not subFolders:
        #        videos.append(root)
        #        root_small = root[len(videos_path):]
        #        videos_small.append(root_small)
        
        #self.videos = videos
        #self.videos_small = videos_small
        
        #self.videos_small = [self.comboBox_chooseVideo.itemText(i) for i in range(self.comboBox_chooseVideo.count())][1:]#the first one is random
        