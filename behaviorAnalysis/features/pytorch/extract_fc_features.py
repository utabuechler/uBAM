#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 2.7
@author: Uta Buechler
Last Update: 28.8.2018

Code for extracting the features of a trained model using pytorch
"""
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.autograd import Variable
from glob import glob

sys.path.append("features/pytorch")
from model import CaffeNet
sys.path.append("features")
from utils import load_table, adjust_coordinates, get_vid_info, transform_batch, set_up_vid_path

import config_pytorch as cfg

################################
# 1. Prepare Data
################################

if isinstance(cfg.crops_path,str):
    use_jpg_crops = True
    cfg.frames_path = cfg.crops_path

#for the fc features we only need the frames itself
detections = load_table(cfg.detection_file,asDict=False)
videos = np.unique(list(detections['videos']))
N_vids = len(videos)


################################
# 2. Prepare Network
################################

net = CaffeNet(batchsize=cfg.batchsize_train,
               input_shape=(3,cfg.input_img_size,cfg.input_img_size),
               seq_len = cfg.seq_len,
               gpu=True if cfg.gpu_id is not None else False)

#initialize the weights
if cfg.final_weights is None:#load the last saved state from training
    checkpoints = sorted(glob(cfg.checkpoint_path+'*.pth.tar'))
    init_dict = torch.load(checkpoints[-1])['state_dict']
else:
    init_dict = torch.load(cfg.final_weights)['state_dict']

net.load_weights(init_dict)

#set GPU option
use_gpu=0
if cfg.gpu_id is not None:
    use_gpu=1
    #you can also use USE CUDA_VISIBLE_DEVICES={gpu_id} before calling python, but still gpu_id has to be set
    #torch.cuda.device(cfg.gpu_id)
    net.cuda()
    

################################
# 3. Extract Features
################################
net.eval()
batchsize = cfg.batchsize_train*cfg.seq_len#number of frames per batch
batch_iter = 0#to know when the batch is full
for vid in tqdm(videos):
    frames, coords, groups, orig_size = get_vid_info(detections,vid,use_jpg_crops)
    
    n = len(frames)
    
    #set up path for saving the features
    save_path,vid_name = set_up_vid_path(cfg.features_path+'fc6_fc7_feat/',vid)
    
    #initialize arrays
    fc6_vid = np.empty((n,4096),np.float32)
    fc7_vid = np.empty((n,4096),np.float32)
    images = np.empty((np.min((batchsize,n)),cfg.input_img_size,cfg.input_img_size,3),np.float32)
    
    #start feature extraction
    for j,(f,ci) in enumerate(zip(frames,coords)):
        c = [int(i) for i in ci]#make sure that it is int
        image = Image.open('%s%s/%06d.jpg'%(cfg.frames_path,vid,f))#self.frames_path is either the path to the crops or the path to the original frames
        os = orig_size if use_jpg_crops else image.size #original size of the frames
        
        #change c if we saved the crops
        c = adjust_coordinates(c,os,image.size,use_jpg_crops)
        
        #crop the original detection
        image = image.crop(c)
        
        #resize the image to the input size of the network
        image = np.array(image.resize((cfg.input_img_size,cfg.input_img_size),Image.BILINEAR),dtype=np.float32)
        image = np.reshape(image,(1,cfg.input_img_size,cfg.input_img_size,3))
        
        #save the image in batch
        images[batch_iter] = image
        
        #run the batch through the network if it is filled or if we are at the end of the video
        if batch_iter==batchsize-1 or j==n-1:
            
            images = transform_batch(images)
            
            images = Variable(torch.from_numpy(images))
            
            if use_gpu:
                images = images.float().cuda()
            
            fc6,fc7 = net.extract_features_fc(images)
            
            fc6_vid[j-batch_iter:j+1,:] = fc6.cpu().detach().numpy()
            fc7_vid[j-batch_iter:j+1,:] = fc7.cpu().detach().numpy()
            
            #reset iterator and array
            batch_iter = 0
            images = np.empty((np.min((batchsize,n-(j+1))),cfg.input_img_size,cfg.input_img_size,3),np.float32)
                
        else:
            batch_iter+=1
            
    #save the features of vid with all of its information
    np.savez(save_path+'/'+vid_name+'.npz',video=vid,groups=groups,frames=frames,coords = coords,orig_size=orig_size,fc6=fc6_vid,fc7=fc7_vid)
        
        
    