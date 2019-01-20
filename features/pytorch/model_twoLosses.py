# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 18:22:00 2017

@author: ubuechle
"""
import torch
import torch.nn as nn
from torch import cat
from torch import from_numpy
import numpy as np
from torch.autograd import Variable

#class LRN(nn.Module):
    #def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        #super(LRN, self).__init__()
        #self.ACROSS_CHANNELS = ACROSS_CHANNELS
        #if ACROSS_CHANNELS:
            #self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    #stride=1,
                    #padding=(int((local_size-1.0)/2), 0, 0))
        #else:
            #self.average=nn.AvgPool2d(kernel_size=local_size,
                    #stride=1,
                    #padding=int((local_size-1.0)/2))
        #self.alpha = alpha
        #self.beta = beta


    #def forward(self, x):
        #if self.ACROSS_CHANNELS:
            #div = x.pow(2).unsqueeze(1)
            #div = self.average(div).squeeze(1)
            #div = div.mul(self.alpha).add(1.0).pow(self.beta)
        #else:
            #div = x.pow(2)
            #div = self.average(div)
            #div = div.mul(self.alpha).add(1.0).pow(self.beta)
        #x = x.div(div)
        #return x

class CaffeNet(nn.Module):

    def __init__(self,
                 batchsize=64,
                 input_shape=(3,227,227),
                 seq_len = 8,
                 gpu=True):
        super(CaffeNet, self).__init__()
        
        self.batchsize = batchsize
        self.seq_len = seq_len
        self.gpu = gpu
        self.hidden = self.init_hidden(batchsize)#creates new initial states for new sequences
        
        self.conv = nn.Sequential()
        self.conv.add_module('conv1',nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0))
        self.conv.add_module('relu1',nn.ReLU(inplace=True))
        self.conv.add_module('pool1',nn.MaxPool2d(kernel_size=3, stride=2))
        #self.conv.add_module('LRN1',LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.conv.add_module('lrn1',nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75))
        #self.features.add_module('bn1',nn.BatchNorm2d(96))
        
        self.conv.add_module('conv2',nn.Conv2d(96, 256, kernel_size=5, padding=2,groups=2),)
        self.conv.add_module('relu2',nn.ReLU(inplace=True))
        self.conv.add_module('pool2',nn.MaxPool2d(kernel_size=3, stride=2))
        #self.conv.add_module('LRN2',LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.conv.add_module('lrn2',nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75))
        #self.features.add_module('bn2',nn.BatchNorm2d(256))
        
        self.conv.add_module('conv3',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('relu3',nn.ReLU(inplace=True))
        
        self.conv.add_module('conv4',nn.Conv2d(384, 384, kernel_size=3, padding=1,groups=2))
        self.conv.add_module('relu4',nn.ReLU(inplace=True))
        
        self.conv.add_module('conv5',nn.Conv2d(384, 256, kernel_size=3, padding=1,groups=2))
        self.conv.add_module('relu5',nn.ReLU(inplace=True))
        self.conv.add_module('pool5',nn.MaxPool2d(kernel_size=3, stride=2))
        
        
        n_size = self._get_conv_output(input_shape)
            
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6',nn.Linear(n_size, 4096))
        self.fc6.add_module('relu6',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6',nn.Dropout(p=0.5))
        
        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7',nn.Linear(4096,4096))
        self.fc7.add_module('relu7',nn.ReLU(inplace=True))
        #self.fc7.add_module('bn7',nn.BatchNorm1d(4096))
        self.fc7.add_module('drop7',nn.Dropout(p=0.5))
        
        self.lstm = nn.LSTM(4096,1024,1)
        #self.fc7.add_module('bn7',nn.BatchNorm1d(4096))
        self.drop = nn.Dropout(p=0.5)
        
        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8_permute',nn.Linear(1024, 2))
        
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_conv(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size
    
    def _forward_conv(self, x):
        x = self.conv(x)
        return x
        
    def forward(self, x):
        self.batchsize = x.size(0)/self.seq_len#for variable batchsize
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.fc7(x)
        x = x.view(self.seq_len,self.batchsize,-1)#reshape data
        
        
        ## first output branch (fc7):
        #normalize (compute the mean over the sequence and subtract it from the chosen ones)
        x_mean = x.mean(0)
        x_mean = torch.stack((x_mean,x_mean),0)
        #choose the first and middle frame
        x2 = x[[0,int(self.seq_len/2)],:,:]-x_mean
        
        
        ## second output branch (lstm):
        xlstm,self.hidden = self.lstm(x,self.hidden)#get the lstm features
        xlstm = self.drop(xlstm)
        x = self.classifier(xlstm[-1,:,:].view(self.batchsize, -1))#take the features of the last frame and classify
        
        
        
        return x,x2
    
    def extract_features_fc(self,x):
        #self.batchsize = x.size(0)/self.seq_len#for variable batchsize
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        #x = x.view(self.seq_len,self.batchsize,-1)#reshape data
        
        return x,self.fc7(x)
    
    def extract_features_lstm(self,x):
        self.batchsize = x.size(0)/self.seq_len#for variable batchsize
        _,x = self.extract_features_fc(x)
        x = x.view(self.seq_len,self.batchsize,-1)
        xlstm,self.hidden = self.lstm(x,self.hidden)
        
        return xlstm
        
    def init_hidden(self,batchsize=256):
        # creates new initial states for new sequences
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.gpu:
            return (Variable(torch.zeros(1, batchsize, 1024),requires_grad = False).cuda(),
                    Variable(torch.zeros(1, batchsize, 1024),requires_grad = False).cuda())
        else:
            return (Variable(torch.zeros(1, batchsize, 1024),requires_grad = False),
                    Variable(torch.zeros(1, batchsize, 1024),requires_grad = False))
        
        
        
        
        
        #print 'Init weights %s'(self.)
        #if type(model) in [nn.Conv2d,nn.Linear]:
            #nn.init.xavier_normal(model.weight.data)
            #nn.init.constant(model.bias.data, 0.1)
    
    def load_weights(self,pretrained_dict):
        #model.load_state_dict(torch.load('mytraining.pt'))
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.load_state_dict(model_dict)
        
        print 'Updated the following weights:'
        for key,value in pretrained_dict.iteritems() :
            print key
    
    #def loadCaffeNetWeights(self,pretrained_dict):
        
        #model_dict = self.state_dict()
        
        #for k,v in pretrained_dict.items():
            #if 'features' in k:
                #if '.0.' in k:
                    #if 'weight' in k:
                        #pretrained_dict['conv.conv1.weight'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                    #elif 'bias' in k:
                        #pretrained_dict['conv.conv1.bias'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                
                #elif '.4.' in k:
                    #if 'weight' in k:
                        #pretrained_dict['conv.conv2.weight'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                    #elif 'bias' in k:
                        #pretrained_dict['conv.conv2.bias'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                
                #elif '.8.' in k:
                    #if 'weight' in k:
                        #pretrained_dict['conv.conv3.weight'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                    #elif 'bias' in k:
                        #pretrained_dict['conv.conv3.bias'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                
                #elif '.10.' in k:
                    #if 'weight' in k:
                        #pretrained_dict['conv.conv4.weight'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                    #elif 'bias' in k:
                        #pretrained_dict['conv.conv4.bias'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                
                #elif '.12.' in k:
                    #if 'weight' in k:
                        #pretrained_dict['conv.conv5.weight'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                    #elif 'bias' in k:
                        #pretrained_dict['conv.conv5.bias'] = pretrained_dict[k]
                        #del pretrained_dict[k]
            
            #elif 'classifier' in k:
                #if '.0.' in k:
                    #if 'weight' in k:
                        #pretrained_dict['fc6.fc6.weight'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                    #if 'bias' in k:
                        #pretrained_dict['fc6.fc6.bias'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                
                #if '.3.' in k:
                    #if 'weight' in k:
                        #pretrained_dict['fc7.fc7.weight'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                    #if 'bias' in k:
                        #pretrained_dict['fc7.fc7.bias'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                        
                #if '.6.' in k:
                    #if 'weight' in k:
                        #pretrained_dict['fc8.fc8.weight'] = pretrained_dict[k]
                        #del pretrained_dict[k]
                    #if 'bias' in k:
                        #pretrained_dict['fc8.fc8.bias'] = pretrained_dict[k]
                        #del pretrained_dict[k]
        
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #model_dict.update(pretrained_dict) 
        ## 3. load the new state dict
        #self.load_state_dict(model_dict)
        
        #print 'Updated the following weights:'
        #for key,value in pretrained_dict.items() :
            #print key