#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 2.7
@author: Uta Buechler
Last Update: 22.8.2018

Class for creating the training and testing data
"""

from utils import load_table, adjust_coordinates, get_vid_info, transform_batch
import random
import numpy as np
#from scipy.misc import imread
from PIL import Image #consider with PIL the first axis is x and the second is y!!!



class Dataset:
    
    def __init__(self,
                 det_file,
                 frames_path,
                 crops_saved = True,#in this case: frames_path should be =crops_path
                 phase='train',
                 augment=True,
                 seq_len=8,
                 skip_frames = 0,
                 input_img_size=227,
                 quantity = 0.95,#default for training (testing = 1-quantity)
                 split_video = False,
                 batchsize=48,
                 idx_test=None):
        '''
        Class for preparing the dataset for training and testing the CNN.
        One Object for training and one object for testing is required.
        
        Input:  det_file:           <str>, name and path to the '.csv' file with all detections (required filed names: {'videos','frames','x1','y1','x2','y2'})
                frames_path:        <str>, path of the frames, if crops_saved==True frames_path should be the path to the cropped frames, saved after getting the detections
                crops_saved:        <boolean>, if True the crops, if False the full frames are loaded
                phase:              <str>, 'train' if the Object outputs training data, 'test' if testing data, 'features' if we want to extract the features of the sequences
                seq_len:            <int>, length of the dense sequences
                skip_frames:        <int>, in case there is only little movement between consecutive frames, this option allows to set how many frames should be skipped,
                                    i.e. if skip_frames=1, 1 frame is skipped, so only every second frame is chosen (frame 0,2,4,6 etc)
                input_image_size:   <int>, width and height of the images before inputting them into the network
                quantity:           <float>, defines how much of the available data is (1-quantity: for testing)
                split_video:        <boolean>, if True the training and test data is split within a video, if False we use 100*split_train of the videos as training and the rest as testing
                                    (the network has never seen the testing videos; only useful if the behavior changes significantly during one video)
                batchsize:          <int>, size of one batch (number of sequences per batch), where batchsize/2 the number of 'real' sequences and batchsize/2 the number of shuffled sequences is
                idx_test:           <array of int>, only necessary for the testing data Object;
                                    in case we split the videos in training and testing we need to know which videos are already used for training and which are left for testing
                                    
        Example:    data_train = Dataset('./Preprocessing/detections.csv','./Preprocessing/crops/',crops_saved=True,train=True,batchsize=48)
                    data_test = Dataset('./Preprocessing/detections.csv','./Preprocessing/crops/',crops_saved=True,train=False,idx_test=data_train.idx_test,batchsize=6)
        '''
        
        if np.mod(batchsize,2)!=0:
            raise ValueError("Please choose an even batchsize!")
        
        self.quantity = quantity
        self.batchsize=batchsize
        self.skip_frames = skip_frames+1
        self.frames_path = frames_path
        self.F = seq_len
        self.input_img_size = input_img_size
        self.crops_saved = crops_saved
        self.phase = phase
        self.augment = augment
        
        self._compute_denseSeq(det_file)
        if phase is 'test':
            self.idx_test = idx_test
            if idx_test is None and not split_video:
                raise ValueError("Please provide the video indizes for the testing videos given by the 'Dataset' object of the training data")
        
        if self.quantity==1:#if we want all data
            self.seqs = self._flatten(self.seq_dense)
            self.N_seqs = len(self.seqs)
        else:
            self._split_trainTest(split_video)
        
        if phase is 'train': random.shuffle(self.seqs)#shuffle only the training data
        
        
        self._construct_batches()
        
    def _compute_denseSeq(self,det_file):
        '''
        Function for computing all dense sequences given the detections. Is only used during initialization (__init__)
        
        Input:      det_file:           <str>, name and path to the '.csv' file with all detections
                                        (required filed names: {'videos','frames','x1','y1','x2','y2'})  
        Output:     self.seq_dense:     <list of dictionaries, size: #videos> contains one dictionary per video containing all necessary information for loading the
                                        sequences during the training/testing phase
        '''
        detections = load_table(det_file,asDict=False)#the detections are given in an excel or csv file

        self.videos = np.unique(detections['videos'].values)
        self.N_vids = len(self.videos)

        self.seq_dense = []#initialize dictionary
        for v in range(self.N_vids):
            
            frames, coords, group, orig_size = get_vid_info(detections,self.videos[v],self.crops_saved)
            
            #it might be that some detections inbetween are missing, so find out where we have a gap
            gap = np.where(((frames[1:]-frames[:-1])>1)*1)[0]
            if len(gap)==0:#if all frames are consecutive
                connectSeq = [range(len(frames))]
            else:#get all connected sequences
                connectSeq = [range(gap[0]+1)]#first sequence
                connectSeq.extend([range(gap[i]+1,gap[i+1]+1) for i in range(len(gap)-1)])#all sequences inbetween
                connectSeq.extend([range(gap[-1]+1,len(frames))])#last sequence
            
            #create the dense sequences
            idx_dense = []
            for i in range(len(connectSeq)):
                seq = connectSeq[i]
                #idx_dense.extend([range(seq[i],seq[i]+self.F) for i in range(len(seq)-self.F+1)])
                idx_dense.extend([range(seq[i],seq[i]+self.skip_frames*self.F,self.skip_frames) for i in range(len(seq)-(self.F*self.skip_frames)+self.skip_frames)])
            
            #save all information for the dense sequences
            idx_dense = np.array(idx_dense)
            frames_dense = frames[idx_dense]
            coords_dense = coords[idx_dense]
            group_dense = group[idx_dense]
            seq_dense_v = {'video': self.videos[v],'group': group_dense,'frames':frames_dense,'coords':coords_dense,'orig_size':orig_size}
            self.seq_dense.append(seq_dense_v)
        
    
    def _split_trainTest(self,split_video):
        '''
        Function for splitting the dictionary in training and testing data,
        depending if the current Object is for training or testing data (self.phase)
        
        Input:      split_video:    <boolean>, if True the training and test data is split within a video,
                                    if False we use 100*quantity of the videos as training and the rest as testing
        Output:     self.seqs:      <list of lists, size: #dense seqs> contains per sequence a list with all necessary
                                    information for loading the sequences
                    self.N_seqs:    <int> number of sequences
        '''
        
        if split_video:#split the videos itself (per video use the first self.quantity sequences as training and the rest of the video as testing)
            #useful in case the behavior changes within one video significantly and/or not enough separate videos are available
            
            self.seqs = []
            for v in range(self.N_vids):
                n = len(self.seq_dense[v]['frames'])#number of sequences
                if self.phase is 'train':
                    idx = range(int(np.round(self.quantity*n)))#the first self.quantity percentage of sequences are for training
                elif self.phase is 'test':
                    idx = range(int(np.round(self.quantity*n)),n)#the last ones for testing
                
                vid = self.seq_dense[v]['video']
                orig_size = self.seq_dense[v]['orig_size']
                frames = self.seq_dense[v]['frames'].tolist()
                coords = self.seq_dense[v]['coords'].tolist()
                
                _=[self.seqs.append([vid,frames[i],coords[i],orig_size]) for i in idx]
                
        
        else:#split between the videos (take self.quantity of the videos (randomly) for training and the rest for testing)
            if self.phase is 'train':
                idx_all = np.arange(self.N_vids)
                random.shuffle(idx_all)
                idx = idx_all[:int(np.round(self.quantity*self.N_vids))]
                self.idx_test = idx_all[int(np.round(self.quantity*self.N_vids)):]#we need to save it somehow for later when creating an object for the testing data
            else:
                idx = self.idx_test
            
            ## save, so that one entry in seqs corresponds to one sequence
            self.seqs = self._flatten([self.seq_dense[j] for j in idx])#only use the chosen videos (idx)
        
        self.N_seqs = len(self.seqs)#total number of sequences
        
    def _flatten(self,seq_dense):
        '''
        Function for flattening the list seq_dense
        
        Input:      seq_dense       <list of dictionaries> contains one dictionary per video containing all necessary information for loading the
                                    sequences
        
        Output:     seqs:           <list of lists, size: #dense_seqs_of_given_videos*len(seq_dense)> contains per sequence a list with all necessary
                                    information for loading the sequences
        '''
        
        seqs = []
        for v_info in seq_dense:
            n = len(v_info['frames'])
            vid = v_info['video']
            orig_size = v_info['orig_size']
            frames = v_info['frames'].tolist()
            coords = v_info['coords'].tolist()
            group = v_info['group']
            if self.phase is 'features':
                #one entry per video
                seqs.append([[vid,frames[i],coords[i],group[i],orig_size] for i in range(n)])
            else:
                #all together in one list
                _=[seqs.append([vid,frames[i],coords[i],group[i],orig_size]) for i in range(n)]
            
        
        return seqs
        
        
        
    def _construct_batches(self):
        '''
        Divides all given dense sequences in N_seqs/(batchsize/2) batches.
        For training: 'self.seqs' is shuffled before to get a random combination per batch.
        For feature extraction: one batch should only contain the sequences of the same video (for convenience)
        
        Output:     self.batches    <list of lists, size: #batches> contains all information for loading the sequences of all batches
                    self.N          <int> number of batches     
        '''
        
        if self.phase is 'features':
            #first, one list per video
            batches = []
            for seqs_v in self.seqs:
                Nv = int(np.ceil(len(seqs_v)/float(self.batchsize)))
                batches.append([seqs_v[i*(self.batchsize):min((i+1)*(self.batchsize),len(seqs_v))] for i in range(Nv)])
            
            #then flatten the list (all batches together in one list, not spearated anymore per video)
            self.batches = [seq for batch in batches for seq in batch]
            self.N = len(self.batches)
        else:
            self.N = int(np.floor(self.N_seqs/float(self.batchsize/2)))#'self.batchsize/2' because the batch is filled half with the real sequences and half with its shuffled version
            self.batches = [self.seqs[i*(self.batchsize/2):(i+1)*(self.batchsize/2)] for i in range(self.N)]
            #self.batches = self.batches[:100]########################################################################################################################################################
            #self.N = len(self.batches)############################################################################################################################################################
    
    def shuffle(self):
        '''
        Function for shuffeling the training data after 1 epoch.
        After shuffeling, the batches are constructed again.
        '''
        random.shuffle(self.seqs)###################################################################################################################################################################
        self._construct_batches()
    
    def __len__(self):
        return self.N
    
    def augment_img(self,image,c):
        '''
        Function for augmenting an image. Two augmentations are applied:
            1. a random cropping around the bounding box (+-5%)
            2. rgb jittering
        
        Input:      image:  <PIL Image> either the full frame or its cropped version
                    c:      <list> coordinates of the detection
        Output:     image:  <PIL Image> augmented image
        '''
        
        #1. crop randomly around the given bounding box
        s = image.size
        w = c[2]-c[0] #if c[2]!=-1 else s[0]
        h = c[3]-c[1] #if c[3]!=-1 else s[1]
        
        #in case the bounding box is the image itself, still crop randomly (random cropping essential for training)
        if w==s[0]:
            c[0],c[2] = int(0.05*w),s[0]-int(0.05*w)
            w = c[2]-c[0]#update w
        if h==s[1]: 
            c[1],c[3] = int(0.05*h),s[1]-int(0.05*h)
            h = c[3]-c[1]#update h
            
        
        #we need a translation which is valid, i.e. it cannot go beyond the image (in case the bounding box c is at the border of the image)
        transl_w = random.randint(max(-c[0],int(-0.05*w)),min(s[0]-c[2],int(0.05*w)))
        transl_h = random.randint(max(-c[1],int(-0.05*h)),min(s[1]-c[3],int(0.05*h)))
        
        image = np.asarray(image.crop((c[0]+transl_w,c[1]+transl_h,c[2]+transl_w,c[3]+transl_h)),np.float32)
        
        #2. rgb jittering
        for ch in range(3):
            thisRand = random.uniform(0.8, 1.2)
            image[:,:,ch] *= thisRand
        
        shiftVal = random.randint(0,6)
        if random.randint(0,1) == 1:
            shiftVal = -shiftVal
        image += shiftVal
        
        image[image<0] = 0
        image[image>255] = 255
        
        image = Image.fromarray(image.astype(np.uint8))
        
        
        return image
    
    def __getitem__(self,index):
        '''
        Function for loading all sequences of a new batch. It is called during the training/testing phase (e.g. using enumerate)
        
        Input:      index:  <int> index of the batch which need to be loaded
        Output:     batch:  <float array> contains all images/sequences for the next training iteration (shuffled and not shuffled)
                    labels: <int array> contains all labels given the sequences in 'batch' (1: not shuffled, 0: shuffled)
        '''
        
        
        #get all the information for the current batch (videos, frames and coords)
        batch_info = self.batches[index]
        B = len(batch_info) if self.phase is 'features' else 2*len(batch_info)#for training/testing because half of the sequences are the real ones and half the shuffled ones
        
        #mult = 1 if self.phase is 'features' else 2
        #initialize
        #batch_init = np.empty((self.F*self.batchsize,self.input_img_size,self.input_img_size,3))
        batch_init = np.empty((self.F*B,self.input_img_size,self.input_img_size,3))
        #labels_init = np.empty((self.F*B,))#for caffe (one label per frame)
        labels_init = np.empty((B,))#for pytorch (one label per sequence)
        
        #Bs = B if self.phase is 'features' else B*2#'/2' for training/testing because half of the sequences are the real ones and half the shuffled ones
        
        #build the batch
        for s in range(len(batch_info)):
            imgSeq_pos = np.empty((self.F,self.input_img_size,self.input_img_size,3))#initialize
            
            #get all information for current sequence
            vid = batch_info[s][0]
            frames = batch_info[s][1]
            coords = batch_info[s][2]
            orig_size = batch_info[s][4]
            
            #for loop over the frames per sequence
            for i,(f,ci) in enumerate(zip(frames,coords)):
                c = [int(j) for j in ci]#make sure that it is int
                try:
                    image = Image.open('%s%s/%06d.jpg'%(self.frames_path,vid,f))#self.frames_path is either the path to the crops or the path to the original frames
                except:#in case the images were not saved with leading zeros
                    image = Image.open('%s%s/%i.jpg'%(self.frames_path,vid,f))
                    image = image.crop((0,image.size[1]/2,image.size[0],image.size[1]))###########################################################################################################################
                
                os = orig_size if self.crops_saved else image.size #original size of the frames
                
                #change c if we saved the crops
                c = adjust_coordinates(c,os,image.size,self.crops_saved)
                
                #do augmentation
                if self.augment and self.phase is 'train':#only augment for training and when augment=True
                    image = self.augment_img(image,c)
                else:
                    #crop the given area without using augmentation
                    #if c[2]!=-1:
                    image = image.crop(c)
                    
                #resize the image to the final size
                image = np.array(image.resize((self.input_img_size,self.input_img_size),Image.BILINEAR))
                
                #add image to sequence
                imgSeq_pos[i,:,:,:] = np.reshape(image,(1,self.input_img_size,self.input_img_size,3))
            
            if self.phase is 'features':
                batch_init[s*self.F:(s+1)*self.F,:,:,:] = imgSeq_pos
                labels_init[s] = 1
            else:#shuffle
                #randomly permut for the negative
                imgSeq_neg = np.copy(imgSeq_pos)
                #imgSeq_neg = imgSeq_neg[[5,1,0,6,3,7,4,2],:,:,:]
                #imgSeq_neg = imgSeq_neg[[5,1,9,6,3,8,0,4,7,2],:,:,:]
                np.random.shuffle(imgSeq_neg)###############################################################################################################################################
            
                #concatenate the positives and negatives
                batch_init[s*2*self.F:(s+1)*2*self.F,:,:,:] = np.concatenate((imgSeq_pos,imgSeq_neg),axis=0)
                
                #save the labels (for caffe we need one label per frame)
                #labels_init[s*2*self.F:(s+1)*2*self.F] = np.concatenate((np.ones(self.F,),np.zeros(self.F,)))
                
                #save the labels (one label per sequence)
                labels_init[s*2:(s+1)*2] = [1,0]
        
        #change order (as input in CNN; we need at first all first frames of the sequences, then every second and so on)
        newOrder = np.zeros((batch_init.shape[0]),np.int)
        for i in range(self.F):
            newOrder[i*B:(i+1)*B] = np.arange(i,self.F*B,self.F).astype(np.int)
            
        batch = batch_init[newOrder,:,:,:]
        #labels = labels_init[newOrder]#only necessary for caffe, if we have a label per frame
        labels = labels_init
        
        return transform_batch(np.array(batch, dtype=np.float32)), batch_info if self.phase is 'features' else labels