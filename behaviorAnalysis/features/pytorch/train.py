#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 2.7
@author: Uta Buechler
Last Update: 28.8.2018

Code for training a model with pytorch
"""
import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.optim as optim
import torch.nn as nn
from torch import max as tmax
from torch.autograd import Variable
from tqdm import tqdm

#only use 'Logger' if tensorflow is installed
try:
    sys.path.append("dependencies")
    from logger import Logger
    tf_found=True
except ImportError:
    tf_found = False
    print('Tensorflow not installed, for saving Accuracy etc a txt file is used instead of tensorboard')
    
#if __import__('imp').find_module('tensorflow')[1] is not None:
    #tf_found = True
    #sys.path.append("dependencies")
    #from logger import Logger
#else:
    #tf_found = False

sys.path.append("features/pytorch")
from model import CaffeNet
sys.path.append("features")
from dataset import Dataset

import config_pytorch as cfg###############################################################################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu",type=int,default=0,
                    help="ID of the GPU to use for training. Set to -1 for cpu mode.")
args = parser.parse_args()

################################
# 0. Define some functions
################################
def save_checkpoint(checkpoint_path,epoch,iterTotal,net_state,optim_state):
    state = {
                'epoch': epoch+1,
                'iter': iterTotal,
                'state_dict': net_state,
                'optimizer' : optim_state,
            }
    filename = '%s/checkp_epoch_%03i_iter_%04i.pth.tar'%(checkpoint_path,epoch,iterTotal)
    torch.save(state, filename)

def exp_lr_scheduler(optimizer,iterTrain, lr=0.01, lr_decay_iter=10000):
    """Decay learning rate by a factor of 0.1 every lr_decay_iter"""
    
    if iterTrain!=0 and iterTrain%lr_decay_iter==0:
        lr = lr*0.1
        print('Learning rate is set to %d'%lr)
        optimizer.param_groups[0]['lr'] = lr
    
    return optimizer,lr

def log_images(logger,images,iter_total):
    
    #transpose back
    images = images.permute(0,1,3,4,2)
    #from BGR to RGB
    images = images[:,:,:,:,[2,1,0]]
    info = { 'images': images.view(-1,cfg.input_img_size,cfg.input_img_size,3)}
    for tag, images in info.items():
        logger.image_summary(tag, images, iter_total)


def init_weights(net):
    #initialize all weights using Xavier
    if type(net) in [nn.Conv2d,nn.Linear]:
        nn.init.xavier_normal_(net.weight.data)
        nn.init.constant_(net.bias.data,0.1)
        
    if type(net) in [nn.LSTM]:
        nn.init.xavier_normal_(net.weight_hh_l0)
        nn.init.xavier_normal_(net.weight_ih_l0)
        nn.init.constant_(net.bias_hh_l0,0.1)
        nn.init.constant_(net.bias_ih_l0,0.1)
################################
# 1. Prepare Data
################################

#get two objects for training and testing data
use_jpg_crops = True if isinstance(cfg.crops_path,str) else False
data_train = Dataset(cfg.detection_file,
                     cfg.crops_path if use_jpg_crops else cfg.frames_path,
                     crops_saved=use_jpg_crops,
                     phase='train',
                     batchsize=cfg.batchsize_train,
                     seq_len=cfg.seq_len,
                     skip_frames = cfg.skip_frames,
                     augment=cfg.augment)

data_test = Dataset(cfg.detection_file,
                    cfg.crops_path if use_jpg_crops else cfg.frames_path,
                    crops_saved=use_jpg_crops,
                    phase='test',
                    idx_test=data_train.idx_test,
                    batchsize=cfg.batchsize_test,
                    seq_len=cfg.seq_len,
                    skip_frames = cfg.skip_frames)

################################
# 2. Prepare Network
################################
if not os.path.exists(cfg.checkpoint_path):
    os.makedirs(cfg.checkpoint_path)


net = CaffeNet(batchsize=cfg.batchsize_train,
               input_shape=(3,cfg.input_img_size,cfg.input_img_size),
               seq_len = cfg.seq_len,
               gpu=True if args.gpu is not -1 else False)

net.apply(init_weights)#initialize the weights with Xavier

#initialize the weights
#iter_old = 0
epoch_old = 1
if cfg.init_weights is not None:
    init_dict = torch.load(cfg.init_weights)
    net.load_weights(init_dict['state_dict'])
    #try:
        ##iter_old = init_dict['iter']
        #epoch_old = init_dict['epoch']
    #except:
        ##iter_old = 0
        #epoch_old = 1

#set GPU option
use_gpu=0
if args.gpu is not -1:
    use_gpu=1
    #you can also use USE CUDA_VISIBLE_DEVICES={gpu_id} before calling python, but still gpu_id has to be set
    torch.cuda.device(args.gpu)
    net.cuda()


lr = cfg.lr
criterion = nn.CrossEntropyLoss()

if cfg.optimizer=='SGD':
    optimizer = optim.SGD(net.parameters(),
                        lr=lr,
                        momentum=0.9,
                        weight_decay = 0.0005)
elif cfg.optimizer=='Adam':
    optimizer = optim.Adam(net.parameters(),
                        lr=lr)
else:
    raise ValueError("Optimizer type %s unknown, please choose 'SGD' or 'Adam'"%cfg.optimizer)
        


#set the tensorboard logger
if tf_found:
    logger = Logger(cfg.checkpoint_path+'/logs')
else:
    text_file = open('%s/logs/train_lr%.6f.log'%(cfg.checkpoint_path,cfg.lr), 'a')
        

################################
# 3. Training and Testing
################################
for epoch in range(epoch_old-1,cfg.train_nr_epochs):
    net.train()
    for i, (data,labels) in enumerate(data_train):
        iter_total = epoch*len(data_train)+i+1
        
        #check if the learning rate needs to be changed
        optimizer,lr = exp_lr_scheduler(optimizer,iter_total,lr,cfg.train_stepsize)
        
        #'labels' has a label for every frame, but with pytorch we only need one label per sequence
        #labels = labels[:cfg.batchsize_train]
        
        data = Variable(torch.from_numpy(data))
        labels = Variable(torch.from_numpy(labels))
        
        if use_gpu:
            data, labels = data.float().cuda(), labels.long().cuda()
            
        optimizer.zero_grad()
        
        #clear out the hidden state of the LSTM, detaching it from its history on the last instance.
        net.hidden = net.init_hidden(cfg.batchsize_train)
        
        #forward pass
        out = net(data)
        
        #compute accuracy
        _, predicted = tmax(out.data,1)
        acc = float(sum(labels.cpu().data.numpy()==predicted.cpu().numpy())/float(labels.size(0)))
        
        #backprop
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        #save checkpoint
        if iter_total%cfg.train_checkpoint_freq==0:
            save_checkpoint(cfg.checkpoint_path,epoch,iter_total,net.state_dict(),optimizer.state_dict())
            
        #print result and save for tensorboard
        if iter_total%cfg.train_display_freq==0:
            print('%2d/%2d (iter %i), LR %.5f, Loss: % 1.3f, Accuracy %2.2f%%'\
                %(epoch+1, cfg.train_nr_epochs, iter_total, lr, loss.data.item(),100*acc))
            
            #log the values for tensorboard or save it in a txt file
            if tf_found:
                info = {'loss/train': loss.data.item(), 'accuracy/train': 100*acc, 'learning rate': lr}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, iter_total)
                
                #log some of the sequences, choose randomly which
                #idx = [range(idxi,len(data),cfg.batchsize_train) for idxi in random.sample(range(cfg.batchsize_train),5)]
                #idx = [range(idxi,len(data),cfg.batchsize_train) for idxi in range(min(5,len(labels)))]
                idx = [range(idxi,len(data),cfg.batchsize_train) for idxi in range(5)]
                log_images(logger,data[idx,:,:,:].cpu(),iter_total)
            else:
                text_file.write('Epoch %d[%d]) Accuracy %.3f Loss %.3f LR %.6f\n'%(epoch,i,100*acc,loss.data.item(),lr))
            
    #shuffle the training data for the next epoch
    data_train.shuffle()
    
    #Do a testing after every epoch
    net.eval()
    accT,lossT = torch.zeros(len(data_test),dtype=torch.float),torch.zeros(len(data_test),dtype=torch.float)
    for i,(data,labels) in enumerate(tqdm(data_test)):
        
        #'labels' has a label for every frame, but with pytorch we only need one label per sequence
        #labels = labels[:cfg.batchsize_test]
        
        data = Variable(torch.from_numpy(data))
        labels = Variable(torch.from_numpy(labels))
        
        if use_gpu:
            data, labels = data.float().cuda(), labels.long().cuda()
            
        net.hidden = net.init_hidden(cfg.batchsize_test)#creates new initial states for new sequences
        
        #forward
        out = net(data)
        
        #get accuracy and loss
        _, predicted = tmax(out.data,1)
        acc = float(sum(labels.cpu().data.numpy()==predicted.cpu().numpy())/float(labels.size(0)))
        accT[i] = acc
        
        loss = criterion(out, labels)
        lossT[i] = float(loss.cpu().data.numpy())
        
    print('Testing Average: Loss: %.4f, Accuracy %.1f%%'\
        %(lossT.mean(),100*accT.mean()))
    
    #log the values for tensorboard or save it in a txt file
    if tf_found:
        info = {'loss/test': lossT.mean(), 'accuracy/test': 100*accT.mean()}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, iter_total)
    else:
        text_file.write('---Testing: Epoch %d Accuracy %.5f, Loss %.3f---\n'%(epoch,100*accT.mean(),lossT.mean()))



























