#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 2.7
@author: Uta Büchler 
uta.buechler@iwr.uni-heidelberg.de
Last Update: 22.10.2018

Use FAISS library and the extracted Features to compute Nearest Neighbor
"""

import os
import sys
import numpy as np
from sklearn import preprocessing
import argparse
from time import time
from tqdm import tqdm, trange
from utils import load_table, load_features
import matplotlib.pyplot as plt
from PIL import Image

import config_pytorch as cfg

try:
    import faiss
    IS_FAISS=True
except ImportError:
    from scipy.spatial.distance import cdist
    IS_FAISS=False
    print('FAISS not installed, a simple cosine distance will be used (takes more time).')

parser = argparse.ArgumentParser()
parser.add_argument("-ft", "--feature_type",type=str,default='fc6',
                    help="type of features, 'fc6','fc7' or 'fc6fc7' for postures or 'lstm' for sequences")
parser.add_argument("-nq", "--number_of_queries",type=int,default=100,
                    help="number of queries used for showing the nearest neighbor")                    
parser.add_argument("-nn", "--nn_per_query",type=int,default=10,
                    help="nearest neighbor shown per query")
args = parser.parse_args()


def compute_NN(feat,k,idx):
    n,d = feat.shape
    if IS_FAISS:
        #euclidean:
        #index = faiss.IndexFlatL2(d)#compute the distance
        #index.add(np.ascontiguousarray(feat))
        #D,I = index.search(feat[idx[:nr],:],100*k)#100*k because later we want to sort out the NN from the same video
        #cosine:
        index = faiss.IndexFlatIP(d)#computes the similarity AND NOT DISTANCE
        feat = preprocessing.normalize(feat, norm='l2')
        index.add(np.ascontiguousarray(feat))
        D,I = index.search(feat[idx[:nr],:],100*k)
        #nlist=10
        #nprobe = 100 #default is 1
        #quantizer = faiss.IndexFlatL2(d)  # the other index
        #index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            # here we specify METRIC_L2, by default it performs inner-product search
        #assert not index.is_trained
        #index.train(fc6)
        #assert index.is_trained
        #index.nprobe = nprobe
        
    else:
        D = np.zeros((len(idx),100*k),np.float32)
        I = np.zeros((len(idx),100*k),np.int)
        dist = cdist(feat[idx,:],feat,metric='cosine')
        for j,d in enumerate(dist):
            sort_idx = np.argsort(d)
            d = d[sort_idx]
            I[j,:] = sort_idx[:100*k]
            D[j,:] = d[:100*k]
        
    return D,I
    
# if __name__ == "__main__":
############################################
# 1. Load Features
############################################
results_fold = '%s/similarity/nearestNeighbor/%s/'%(cfg.results_path,args.feature_type)
if not os.path.exists(results_fold): os.makedirs(results_fold)

#videos = load_table(cfg.video_path+'/vid_list.csv',index=None,asDict=True).values()
#if len(videos)>1: videos = videos[1].values()
detections = load_table(cfg.detection_file,asDict=False)
uni_videos = np.unique(detections['videos'].values)

if 'fc6' or 'fc7' in args.feature_type:
    print('Chosen features: "%s". Compute %i Nearest Neighbors of %i randomly chosen postures. The Results will be saved in "%s".'%(args.feature_type,args.nn_per_query,args.number_of_queries,results_fold))
elif 'lstm' in args.feature_type:
    print('Chosen features: "%s". Compute %i Nearest Neighbors of %i randomly chosen sequences. The Results will be saved in "%s".'%(args.feature_type,args.nn_per_query,args.number_of_queries,results_fold))
else:
    raise ValueError('Chosen Features (%s) are not available. Please choose "fc6", "fc7" or "fc6fc7" for posture features or "lstm" for sequence features.'%args.feature_type)
    
print('Load features...')
feat,frames,coords,vids = load_features(args.feature_type,cfg.features_path,uni_videos.tolist())

############################################
# 2. compute NN and plot it
############################################
k = args.nn_per_query #number of nearest neighbor per query
nr = min(args.number_of_queries,len(feat))#number of queries
idx = np.random.permutation(len(feat))[:nr]#choose randomly queries

#plot queries and NN
if 'fc6' in args.feature_type or 'fc7' in args.feature_type:
    n_mean_nn = 100#we also want to plot the mean over the 100 nearest neighbor of the queries
    print('Compute %i Nearest Neighbor for %i queries'%(k,nr))
    D,I = compute_NN(feat,min(100*n_mean_nn,feat.shape[0]),idx)
    nr_rows,r,fig_nr=10,0,1
    fig = plt.figure(figsize=(k+2*2,nr_rows*1.5))
    for i,idx_i in enumerate(tqdm(idx)):
        mean_nn = np.zeros((200,200,3),np.float64)#init mean image
        
        ##get the current query
        vid = vids[idx_i]
        frame = '%s%s/%06d.jpg'%(cfg.crops_path,vid,frames[idx_i])
        image = Image.open(frame)
        
        ##plot the current query
        _=plt.subplot(nr_rows,k+2,r*(k+2)+1)
        _=plt.imshow(image)
        _=plt.axis('off')
        _=plt.title(str(idx_i))
        
        ##plot the nearest neighbor
        nn_found,j=0,0
        while nn_found<k:
            nn = I[i,j]
            dist = D[i,j]
            if vids[nn]==vid:#don't show NN of the same video
                j+=1
                continue
            
            frame = '%s%s/%06d.jpg'%(cfg.crops_path,vids[nn],frames[nn])
            image = Image.open(frame)
            mean_nn += np.asarray(image.resize((200,200),Image.BILINEAR),np.float64)
            _=plt.subplot(nr_rows,k+2,r*(k+2)+nn_found+2)
            _=plt.imshow(image)
            _=plt.axis('off')
            _=plt.title('%.3f'%dist)
            j+=1
            nn_found+=1
        
        ##collect n_mean_nn Nearest Neighbor and plot the mean
        while nn_found<n_mean_nn and j<I.shape[1]:
            nn = I[i,j]
            dist = D[i,j]
            if vids[nn]==vid:#don't show NN of the same video
                j+=1
                continue
            
            frame = '%s%s/%06d.jpg'%(cfg.crops_path,vids[nn],frames[nn])
            image = Image.open(frame)
            mean_nn += np.asarray(image.resize((200,200),Image.BILINEAR),np.float64)
            
            j+=1
            nn_found+=1
        
        mean_nn /= nn_found
        _=plt.subplot(nr_rows,k+2,r*(k+2)+k+2)
        _=plt.imshow(mean_nn.astype(np.uint8))
        _=plt.axis('off')
        _=plt.title('%iNN'%n_mean_nn)
        
        ##check if the figure is full, if yes, save it
        if (r+1)<nr_rows:
            r+=1
        else:
            _=plt.savefig(results_fold+str(fig_nr)+'.jpg',bbox_inches='tight')#,dpi=50)
            r = 0
            fig_nr+=1
            fig.clf()
    
    #save the last figure
    _=plt.savefig(results_fold+str(fig_nr)+'.jpg',bbox_inches='tight')

else:
    D,I = compute_NN(feat,min(100*k,feat.shape[0]),idx)
    nr_rows=k+1
    fig = plt.figure(figsize=(cfg.seq_len*2,nr_rows*2))
    for i,idx_i in enumerate(tqdm(idx)):
        vid = vids[idx_i]
        
        ##plot the current query
        for m,f in enumerate(frames[idx_i]):
            frame = '%s%s/%06d.jpg'%(cfg.crops_path,vid,f)
            image = Image.open(frame)
            _=plt.subplot(nr_rows,cfg.seq_len,m+1)
            _=plt.imshow(image)
            _=plt.axis('off')
            if m==0: _=plt.title(str(idx_i))
            
        
        ##plot the nearest neighbor
        nn_found,j=0,0
        while nn_found<k:
            nn = I[i,j]
            dist = D[i,j]
            if vids[nn]==vid:#don't show NN of the same video
                j+=1
                continue
            
            for m,f in enumerate(frames[nn]):
                frame = '%s%s/%06d.jpg'%(cfg.crops_path,vids[nn],f)
                image = Image.open(frame)
                _=plt.subplot(nr_rows,cfg.seq_len,(nn_found+1)*cfg.seq_len+m+1)
                _=plt.imshow(image)
                _=plt.axis('off')
                if m==0: _=plt.title('%.3f'%dist)
            
            j+=1
            nn_found+=1
            
        _=plt.savefig(results_fold+str(idx_i)+'.jpg',bbox_inches='tight')
        fig.clf()
        
    