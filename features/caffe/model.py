#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 2.7
@author: Uta Buechler
Last Update: 22.8.2018

Module for defining the network structure and creating the prototxt files
"""
import config as cfg
import sys
sys.path.append(cfg.caffe_path)
import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np


def _create_network():
    
    #batch_size = .train_batch if is_training else options.test_batch
    N_train = cfg.batchsize_train*cfg.seq_len
    N_test = cfg.batchsize_test*cfg.seq_len

    n = caffe.NetSpec()

    #define the input
    n.data = L.Input(name='data',input_param = dict(shape=dict(dim=[N_train,3,cfg.input_img_size,cfg.input_img_size])),ntop=1,include=dict(phase=caffe.TRAIN))
    n.data_test = L.Input(name='data',input_param = dict(shape=dict(dim=[N_test,3,cfg.input_img_size,cfg.input_img_size])),ntop=0,include=dict(phase=caffe.TEST),top='data')

    n.label = L.Input(name='label',input_param = dict(shape=dict(dim=N_train)),ntop=1,include=dict(phase=caffe.TRAIN))
    n.label_test = L.Input(name='label',input_param = dict(shape=dict(dim=N_test)),ntop=0,include=dict(phase=caffe.TEST),top='label')

    n.cm = L.Input(name='clip_marker',input_param = dict(shape=dict(dim=[cfg.seq_len,cfg.batchsize_train])),ntop=1,include=dict(phase=caffe.TRAIN))
    n.cm_test = L.Input(name='clip_marker',input_param = dict(shape=dict(dim=[cfg.seq_len,cfg.batchsize_test])),ntop=0,include=dict(phase=caffe.TEST),top='cm')

    #define network
    n.conv1 = L.Convolution(bottom='data',
                            param=[dict(lr_mult=0.1,decay_mult=1),dict(lr_mult=0.2,decay_mult=0)],#first dict for weights, second for bias,
                            convolution_param=dict(num_output=96,kernel_size=11,stride=4,
                                                weight_filler=dict(type="gaussian",std=0.01),
                                                bias_filler=dict(type="constant",value=0)))
    n.relu1 = L.ReLU(n.conv1)
    n.pool1 = L.Pooling(n.relu1, pooling_param=dict(pool=P.Pooling.MAX,kernel_size=3,stride=2))
    n.norm1 = L.LRN(n.pool1,lrn_param=dict(local_size=5,alpha=0.0001,beta=0.75))

    #
    n.conv2 = L.Convolution(n.norm1,
                            param=[dict(lr_mult=0.1,decay_mult=1),dict(lr_mult=0.2,decay_mult=0)],
                            convolution_param=dict(num_output=256,pad=2,kernel_size=5,group=2,
                                                weight_filler=dict(type="gaussian",std=0.01),
                                                bias_filler=dict(type="constant",value=0)))
                            
    #n.conv2 = L.Convolution(n.norm1,
                            #param=[dict(lr_mult=0.1,decay_mult=1),dict(lr_mult=0.2,decay_mult=0)],
                            #convolution_param=dict(num_output=256,pad=2,kernel_size=5,
                                                #weight_filler=dict(type="gaussian",std=0.01),
                                                #bias_filler=dict(type="constant",value=0)))
                            
    n.relu2 = L.ReLU(n.conv2)
    n.pool2 = L.Pooling(n.relu2, pooling_param=dict(pool=P.Pooling.MAX,kernel_size=3,stride=2))
    n.norm2 = L.LRN(n.pool2,lrn_param=dict(local_size=5,alpha=0.0001,beta=0.75))

    #
    n.conv3 = L.Convolution(n.norm2,
                            param=[dict(lr_mult=0.1,decay_mult=1),dict(lr_mult=0.2,decay_mult=0)],
                            convolution_param=dict(num_output=384,pad=1,kernel_size=3,
                                                weight_filler=dict(type="gaussian",std=0.01),
                                                bias_filler=dict(type="constant",value=0)))
    n.relu3 = L.ReLU(n.conv3)

    #
    n.conv4 = L.Convolution(n.relu3,
                            param=[dict(lr_mult=0.1,decay_mult=1),dict(lr_mult=0.2,decay_mult=0)],
                            convolution_param=dict(num_output=384,pad=1,kernel_size=3,group=2,
                                                weight_filler=dict(type="gaussian",std=0.01),
                                                bias_filler=dict(type="constant",value=0)))
                            
    #n.conv4 = L.Convolution(n.relu3,
                            #param=[dict(lr_mult=0.1,decay_mult=1),dict(lr_mult=0.2,decay_mult=0)],
                            #convolution_param=dict(num_output=384,pad=1,kernel_size=3,
                                                #weight_filler=dict(type="gaussian",std=0.01),
                                                #bias_filler=dict(type="constant",value=0)))
    
    n.relu4 = L.ReLU(n.conv4)

    #
    n.conv5 = L.Convolution(n.relu4,
                            param=[dict(lr_mult=1,decay_mult=1),dict(lr_mult=2,decay_mult=0)],
                            convolution_param=dict(num_output=256,pad=1,kernel_size=3,group=2,
                                                weight_filler=dict(type="gaussian",std=0.01),
                                                bias_filler=dict(type="constant",value=0)))
                            
    #n.conv5 = L.Convolution(n.relu4,
                            #param=[dict(lr_mult=1,decay_mult=1),dict(lr_mult=2,decay_mult=0)],
                            #convolution_param=dict(num_output=256,pad=1,kernel_size=3,
                                                #weight_filler=dict(type="gaussian",std=0.01),
                                                #bias_filler=dict(type="constant",value=0)))
    
    n.relu5 = L.ReLU(n.conv5)
    n.pool5 = L.Pooling(n.relu5, pooling_param=dict(pool=P.Pooling.MAX,kernel_size=3,stride=2))

    #
    n.fc6 = L.InnerProduct(n.pool5,
                        param=[dict(lr_mult=1,decay_mult=1),dict(lr_mult=2,decay_mult=0)],
                        inner_product_param=dict(num_output=4096,
                                                    weight_filler=dict(type="gaussian",std=0.005),
                                                    bias_filler=dict(type="constant",std=0.1)))
    n.relu6 = L.ReLU(n.fc6)
    n.drop6 = L.Dropout(n.relu6,dropout_param=dict(dropout_ratio=0.5))

    #
    n.fc7 = L.InnerProduct(n.drop6,
                        param=[dict(lr_mult=1,decay_mult=1),dict(lr_mult=2,decay_mult=0)],
                        inner_product_param=dict(num_output=4096,
                                                    weight_filler=dict(type="gaussian",std=0.005),
                                                    bias_filler=dict(type="constant",std=0.1)))
    n.relu7 = L.ReLU(n.fc7)
    n.drop7 = L.Dropout(n.relu7,dropout_param=dict(dropout_ratio=0.5))

    #
    n.fc7_reshape = L.Reshape(n.drop7,reshape_param=dict(shape=dict(dim=[cfg.seq_len,cfg.batchsize_train,4096])),include=dict(phase=caffe.TRAIN))
    n.fc7_reshape_test = L.Reshape(n.drop7,name='fc7_reshape',reshape_param=dict(shape=dict(dim=[cfg.seq_len,cfg.batchsize_test,4096])),ntop=0,include=dict(phase=caffe.TEST),top='fc7_reshape')
    n.label_reshape = L.Reshape(bottom='label',reshape_param=dict(shape=dict(dim=[cfg.seq_len,cfg.batchsize_train])),include=dict(phase=caffe.TRAIN))
    n.label_reshape_test = L.Reshape(bottom='label',name='label_reshape',reshape_param=dict(shape=dict(dim=[cfg.seq_len,cfg.batchsize_test])),include=dict(phase=caffe.TEST),ntop=0,top='label_reshape')

    #
    n.lstm1 = L.LSTM(bottom=['fc7_reshape','cm'],recurrent_param=dict(num_output=1024,
                                                                   weight_filler=dict(type="uniform",min=-0.01,max=0.01),
                                                                   bias_filler=dict(type="constant",value=0)))
    n.drop_lstm1 = L.Dropout(n.lstm1,dropout_param=dict(dropout_ratio=0.5))

    #
    n.fc8_permute = L.InnerProduct(n.drop_lstm1,
                        param=[dict(lr_mult=10,decay_mult=1),dict(lr_mult=20,decay_mult=0)],
                        inner_product_param=dict(num_output=2,
                                                    weight_filler=dict(type="gaussian",std=0.01),
                                                    bias_filler=dict(type="constant",value=0),
                                                    axis=2))
    #
    n.loss = L.SoftmaxWithLoss(bottom=['fc8_permute','label_reshape'],softmax_param=dict(axis=2))
    n.probs = L.Softmax(bottom='fc8_permute',softmax_param=dict(axis=2),include=dict(phase=caffe.TEST))
    n.accuracy = L.Accuracy(bottom=['fc8_permute','label_reshape'],accuracy_param=dict(axis=2))
    
    return 'name: "'+cfg.net_name+'"\n' + str(n.to_proto())

def write_network(net_file):
    network = _create_network()
    with open(net_file,'w') as f:
        f.write(network)
    
def write_solver_SGD(solver_file,train_iter):
    solver_mode='GPU' if isinstance(cfg.gpu_id,int) else 'CPU'
    with open(solver_file,"w") as f:
        f.write('net: "'+ cfg.net_file+'"\n')
        
        f.write('test_iter: 1\n')
        f.write('test_interval: 1000000000\n')#We execute the testing that by our own, we don't want caffe to run a testing by itself
        
        f.write('base_lr: %f\n'%cfg.lr)
        f.write('lr_policy: "step"\n')
        f.write('gamma: 0.1\n')
        f.write('stepsize: %d\n'%(cfg.train_stepsize))
        f.write('momentum: %f\n'%(0.9))
        f.write('weight_decay: %f\n'%(0.005))
        
        f.write('snapshot: %d\n'%(cfg.train_snapshot_freq))
        f.write('snapshot_prefix: "'+cfg.snapshot_name+'"\n')
        
        f.write('display: %d\n'%cfg.train_display_freq)
        f.write('max_iter: %d\n'%train_iter)
        f.write('solver_mode: %s\n'%solver_mode)
        if isinstance(cfg.gpu_id,int): f.write('device_id: %d\n'%cfg.gpu_id)
        
def write_solver_Adam(solver_file,train_iter):
    solver_mode='GPU' if isinstance(cfg.gpu_id,int) else 'CPU'
    with open(solver_file,"w") as f:
        f.write('net: "'+ cfg.net_file+'"\n')
        
        f.write('test_iter: 1\n')
        f.write('test_interval: 1000000000\n')#We execute the testing that by our own, we don't want caffe to run a testing by itself
        
        f.write('base_lr: %f\n'%cfg.lr)
        f.write('lr_policy: "%s"\n'%('fixed'))
        f.write('momentum: %f\n'%(0.9))
        f.write('momentum2: %f\n'%(0.999))
        
        f.write('snapshot: %d\n'%(cfg.train_snapshot_freq))
        f.write('snapshot_prefix: "'+cfg.snapshot_name+'"\n')
        
        f.write('display: %d\n'%cfg.train_display_freq)
        f.write('max_iter: %d\n'%train_iter)
        f.write('solver_mode: %s\n'%solver_mode)
        f.write('type: "Adam"\n')
        if isinstance(cfg.gpu_id,int): f.write('device_id: %d\n'%cfg.gpu_id)

def get_lr(curr_iter):
    #lr policy is 'step' and gamma=0.1, change of learning rate: cfg.train_stepsize
    gamma = 0.1
    nr_changes = np.floor(float(curr_iter)/cfg.train_stepsize)
    
    return cfg.lr*(nr_changes*gamma) if nr_changes>0 else cfg.lr