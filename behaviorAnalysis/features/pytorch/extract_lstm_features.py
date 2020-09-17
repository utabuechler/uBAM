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
import torch
from torch.autograd import Variable
from glob import glob

sys.path.append("features/pytorch")
from model import CaffeNet
sys.path.append("features")
from dataset import Dataset
from utils import set_up_vid_path

import config_pytorch as cfg

################################
# 0. Define functions
################################
def init_arrays(seq_len):
    
    lstm_vid = np.array([],dtype=np.float32).reshape(0,seq_len,1024)
    frames = np.array([],dtype=np.int).reshape(0,seq_len)
    coords = np.array([],dtype=np.int).reshape(0,seq_len,4)
    group = np.array([],dtype=np.int).reshape(0,seq_len)
    orig_size = np.array([],dtype=np.int).reshape(0,2)
    
    return lstm_vid,frames,coords,group,orig_size

def extract_vid_info_from_batch(batch_info):
    
    batch_info = np.array(batch_info,dtype=object)
    
    info = [batch_info[0][0]]
    for i in range(1,len(batch_info[0])):
        info.append(np.array([item for item in batch_info[:,i]]))
        #1: frames, 2: coords, 3: group, 4: orig_size
    
    return info


################################
# 1. Prepare Data
################################

if isinstance(cfg.crops_path,str):
    use_jpg_crops = True
    cfg.frames_path = cfg.crops_path

#get all dense sequences
data_loader = Dataset(cfg.detection_file,
               cfg.crops_path,
               crops_saved=use_jpg_crops,
               phase='features',
               quantity = 1,#take all data
               batchsize=cfg.batchsize_train,
               skip_frames = cfg.skip_frames,
               seq_len = cfg.seq_len)


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
    #use USE CUDA_VISIBLE_DEVICES={gpu_id} before calling python
    #os.environ['CUDA_VISIBLE_DEVICES']=str(cfg.gpu_id)
    #torch.cuda.device(cfg.gpu_id)
    net.cuda()
    

################################
# 3. Extract Features
################################
net.eval()
vid_old = []

#initialize arrays for saving all seqs of one video
lstm_vid,frames_vid,coords_vid,group_vid,orig_size_vid = init_arrays(cfg.seq_len)

for i, (data, batch_info) in enumerate(tqdm(data_loader)):
    
    vid,frames,coords,group,orig_size = extract_vid_info_from_batch(batch_info)
    
    
    #check if we collected all sequences of a specific video
    if ((vid is not vid_old) and (i is not 0)) or i==len(data)-1:
        
        #set up path for saving the features
        save_path,vid_name = set_up_vid_path(cfg.features_path+'lstm_feat/', vid_old)
        
        #save the features with all of its information
        np.savez(save_path+'/'+vid_name+'.npz',video=vid_old,group = group_vid,frames=frames_vid,coords = coords_vid,orig_size=orig_size_vid,lstm=lstm_vid)
        
        #initialize all arrays
        lstm_vid,frames_vid,coords_vid,group_vid,orig_size_vid = init_arrays(cfg.seq_len)
        
    
    vid_old = vid
    
    data = Variable(torch.from_numpy(data.copy()))
    
    if use_gpu: data = data.float().cuda()
    
    #clear out the hidden state of the LSTM, detaching it from its history on the last instance
    net.hidden = net.init_hidden(data.size()[0]/cfg.seq_len)
    
    #forward
    lstm = net.extract_features_lstm(data).transpose(1,0).contiguous()
    
    #concat
    lstm_vid = np.vstack([lstm_vid,lstm.cpu().detach().numpy()])
    group_vid = np.vstack([group_vid,group])
    frames_vid = np.vstack([frames_vid,frames])
    coords_vid = np.vstack([coords_vid,coords])
    orig_size_vid = np.vstack([orig_size_vid,orig_size])
        
    