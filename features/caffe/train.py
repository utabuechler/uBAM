#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 2.7
@author: Uta Buechler
Last Update: 22.8.2018

Code for training a model with caffe
"""

import os
import sys
sys.path.append(os.getcwd())
import config as cfg 
sys.path.append(cfg.caffe_path)
import caffe
import numpy as np
import time
from tqdm import tqdm

#from utils import check_gpu
sys.path.append('features')
from dataset import Dataset
sys.path.append('features/caffe')
from model import write_network, write_solver_SGD, write_solver_Adam,get_lr

################################
# 1. Prepare Data
################################

#get two objects for training and testing data
use_jpg_crops = True if isinstance(cfg.crops_path,str) else False
data_train = Dataset(cfg.detection_file,
                     cfg.crops_path,
                     crops_saved=use_jpg_crops,
                     train=True,
                     batchsize=cfg.batchsize_train)

data_test = Dataset(cfg.detection_file,
                    cfg.crops_path,
                    crops_saved=use_jpg_crops,
                    train=False,
                    idx_test=data_train.idx_test,
                    batchsize=cfg.batchsize_test)

################################
# 2. Prepare Network
################################
snapshot_path = cfg.snapshot_name[:cfg.snapshot_name.rfind('/')]
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)


#compute total number of iterations
train_iter = cfg.train_nr_epochs*len(data_train)

#create and write the network and solver
write_network(cfg.net_file)
solver_file    = cfg.net_file[:cfg.net_file.rfind('/')+1]+'solver.prototxt'
if cfg.solver=='SGD':
    write_solver_SGD(solver_file,train_iter)
    solver = caffe.SGDSolver(solver_file)
elif cfg.solver=='Adam':
    write_solver_Adam(solver_file,train_iter)
    solver = caffe.AdaGradSolver(solver_file)


#set GPU option
if isinstance(cfg.gpu_id,int) or cfg.gpu_id:
    caffe.set_mode_gpu()
    #gpu_id = map(int, str(cfg.gpu_id)) if isinstance(cfg.gpu_id,int) else cfg.gpu_id#transfrom cfg.gpu_id to list (or tuple)
    #gpu_id = check_gpu(gpu_id)#if several gpus are given as input, output the best one (least memory usage)
    caffe.set_device(gpu_id)
else:
    caffe.set_mode_cpu()

#restore the weights
#if cfg.init_weights is not None:
    #solver.net.copy_from(cfg.init_weights)

#get the networks
net = solver.net
testnet = solver.test_nets[0]
testnet.share_with(net)


#define clip marker (input to the network)
cm = np.ones((cfg.seq_len,cfg.batchsize_train))
cm[0,:] = 0
cm_test = np.ones((cfg.seq_len,cfg.batchsize_test))
cm_test[0,:] = 0

#save accuracy and loss in a txt file
text_file = open(snapshot_path+('/train_lr%.6f.log'%cfg.lr), 'a')

################################
# 3. Training and Testing
################################
for epoch in range(cfg.train_nr_epochs):
    for i,(data,labels) in enumerate(data_train):
        #input the data
        net.blobs['data'].data[...] = data
        net.blobs['label'].data[...] = labels
        net.blobs['cm'].data[...] = cm
        
        #forward and backward step
        solver.step(1)
        
        #extract loss and accuracy of current batch
        loss = net.blobs['loss'].data
        acc  = net.blobs['accuracy'].data
        
        if np.mod(i,cfg.train_display_freq)==0 and i!=0:
            text_file.write('Epoch %d[%d]) Accuracy %.3f Loss %.3f LR %.6f\n'%(epoch,i,acc,loss,get_lr(solver.iter)))
            #print('Epoch %d[%d] Accuracy %.3f Loss %.3f LR %.6f\n'%(epoch,i,acc,loss,get_lr(solver.iter)))
    
    #shuffle the training data for the next epoch
    data_train.shuffle()
    
    #do a testing every epoch
    print(cfg.RED+'Testing...'+cfg.END)
    loss = np.empty(len(data_test),np.float32)
    acc = np.empty(len(data_test),np.float32)
    acc_last_frame = np.empty(len(data_test))
    for i,(data,labels) in enumerate(tqdm(data_test)):
        testnet.blobs['data'].data[...] = data
        testnet.blobs['label'].data[...] = labels
        testnet.blobs['cm'].data[...] = cm_test
        
        out = testnet.forward()
        
        loss[i] = out['loss']
        acc[i]  = out['accuracy']
        
        #compute accuracy based only on the prediction of the last frame
        plabel = out['probs'].argmax(axis=2)[-1,:]#prediction of last frame
        tlabel = labels[:cfg.batchsize_test]#true label
        acc_last_frame[i] = sum(plabel==tlabel)/float(cfg.batchsize_test)#compute accuracy
    
    loss = np.mean(loss)
    acc = np.mean(acc)
    acc_last_frame = np.mean(acc_last_frame)
    text_file.write('---Testing: Epoch %d Accuracy %.5f, Accuracy Last Frame %.3f, Loss %.3f---\n'%(epoch,acc,acc_last_frame,loss))
    print('%sEpoch %d Testing Result: Accuracy %.5f, Accuracy Last Frame %.3f, Loss %.3f%s'%(cfg.RED,epoch,acc,acc_last_frame,loss,cfg.END))
        

text_file.close()

#equal = []
#for name in net.params.keys():
    #for name2 in net.params.keys():
        #if name==name2:
            #equal.append(np.array_equal(net.params[name][0].data,testnet.params[name][0].data))
    