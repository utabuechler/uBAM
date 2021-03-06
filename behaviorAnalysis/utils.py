#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 2.7
@author: Uta Buechler
Last Update: 19.2.2019

Module for collecting some functions used in various scripts
"""
import os
import pandas as pd
import subprocess
import sys
#import GPUtil as GPUInfo
import warnings
import numpy as np
from tqdm import tqdm, trange
from PIL import Image

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s%s: %s%s\n' %('\033[91m',category.__name__,message,'\033[0m')
    #return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)

def run_on_system(command):
    with open(os.devnull, "w") as f:#suppress the output of ffmpeg
        subprocess.call(command,shell=True,stdout=f,stderr=f)
        


######To do: also allow matlab file for the coordinates#########
def load_table(file,index=None,asDict=True):
    if isinstance(file,str):
        if os.path.splitext(file)[1]=='.xlsx':
            try:
                df = pd.read_excel(io=file,encoding='utf-16')
            except:
                df = pd.read_excel(io=file,encoding='utf-8')
        elif os.path.splitext(file)[1]=='.csv':
            try:
                df = pd.read_csv(file,encoding='utf-16')
            except:
                df = pd.read_csv(file,encoding='utf-8')
    
        df.columns = [x.lower() for x in df.columns]#to be invariant to column headers with capital letter
        
        if index is not None:
            df =  df.set_index(index).T
            
        if asDict:
            return df.to_dict()
        else:
            return df
        
    else:
        return file

def adjust_coordinates(c,os,s,use_jpg_crops):
    if c[2]!=-1:
        c[0],c[1],c[2],c[3] = max(0,c[0]),max(0,c[1]),min(os[0],c[2]),min(os[1],c[3])#make sure that the bounding box is not outside of the original image
    else:#in case no tracking approach is necessary, i.e. no detections so we used the default values (0,0,-1,-1)
        c[2],c[3] = os[0],os[1]
        
    if use_jpg_crops:
        #we cropped 5% more on every side (when saving the croppings in preprocessing), so define where we need to crop now based on the original coordinates
        w,h = c[2]-c[0],c[3]-c[1]
        c_new = [0]*len(c)
        c_new[0] = int(0.05*w) if (c[0]-int(0.05*w))>0 else c[0] #the original cropping starts either at the 5% or c[0]
        c_new[1] = int(0.05*h) if (c[1]-int(0.05*h))>0 else c[1] #the original cropping starts either at the 5% or c[1]
        c_new[2] = s[0]-int(0.05*w) if (c[2]+int(0.05*w))<os[0] else s[0]#the original cropping ends either at 5% more, or at the end of the image
        c_new[3] = s[1]-int(0.05*h) if (c[3]+int(0.05*h))<os[1] else s[1]#the original cropping ends either at 5% more, or at the end of the image
        return c_new
    else:
        return c

def get_vid_info(detections,vid,use_jpg_crops):
    idx = [i for i,v in enumerate(detections['videos'].values) if v==vid]#get all entries for the current video
    frames = detections['frames'][idx].values#get all frames where we have detections
    coords = np.array([detections['x1'][idx].values,detections['y1'][idx].values,#get all detections
            detections['x2'][idx].values,detections['y2'][idx].values]).T
    orig_size = [detections['original_size_x'][idx[0]],detections['original_size_y'][idx[0]]] if use_jpg_crops else [0,0]
    groups = detections['group'][idx].values
    
    return frames,coords,groups,orig_size,

def transform_batch(batch):
    
    #change from RGB to BGR
    batch = batch[:, :, :, ::-1]
            
    #normalize between [0,1]
    if batch.max()>1: batch = np.divide(batch,255)
            
    #change from [N,y,x,ch] to [N,ch,y,x]
    batch = batch.transpose((0,3,1,2))
    
    return batch

def set_up_vid_path(features_path,vid):
    
    if vid.find('/') is not -1:
        vid_path = vid[:vid.rfind('/')]
        vid_name = vid[vid.rfind('/')+1:]
    else:
        vid_path = ''
        vid_name = vid
    
    save_path = features_path+vid_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    return save_path,vid_name


def load_features(feature_type,features_path,videos,progress_bar=True,progress_bar_text = 'Extract Features'):
    if not os.path.exists(features_path):
        print('Error! No features found at "%s"'%features_path)
        return -1
    
    if not isinstance(videos,list):
        videos = [videos]
    
    if 'fc6' in feature_type or 'fc7' in feature_type:
        fc6,fc7,frames,coords,vids= [],[],[],[],[]
        for i,vid in (enumerate(tqdm(videos,desc=progress_bar_text)) if progress_bar else enumerate(videos)):
            feat_vid_file = features_path+'fc6_fc7_feat/'+vid+'.npz'
            #print(feat_vid_file)
            if not os.path.exists(feat_vid_file):
                print('Not there')
                continue
            
            feat = np.load(feat_vid_file)
            fc6.extend(feat['fc6'])
            fc7.extend(feat['fc7'])
            frames.extend(feat['frames'])
            coords.extend(feat['coords'])
            _ = [vids.append(vid) for _ in range(len(feat['frames']))]
            
        fc6 = np.array(fc6)
        fc7 = np.array(fc7)
        
        if feature_type=='fc6':
            feat = fc6
        elif feature_type=='fc7':
            feat = fc7
        else:
            feat = np.concatenate((fc6,fc7),1)
        
    elif 'lstm' in feature_type:
        lstm,frames,coords,vids= [],[],[],[]
        for i,vid in (enumerate(tqdm(videos,desc=progress_bar_text)) if progress_bar else enumerate(videos)):
            vid = videos[i]
            feat_vid_file = features_path+'lstm_feat/'+vid+'.npz'
            if not os.path.exists(feat_vid_file):
                continue
            
            feat = np.load(feat_vid_file)
            lstm.extend(feat['lstm'])
            frames.extend(feat['frames'])
            coords.extend(feat['coords'])
            _ = [vids.append(vid) for _ in range(len(feat['frames']))]
            
        lstm = np.array(lstm)
        np.stack(lstm, axis=2).shape
        #feat = lstm[:,-1,:]#take the features of the last frame, since this also encodes all previous frames
        n,l,d = lstm.shape
        feat = np.reshape(lstm,(n,l*d))#or take the features of all frames and concat
        
    vids = np.array(vids)
    frames = np.array(frames)
    coords = np.array(coords)
    return feat,frames,coords,vids

def draw_box(image,coord,l=2,color=[255,0,0]):
    c = np.array([(coord[1]+coord[3])/2,
                  (coord[0]+coord[2])/2]).astype(int)
    size = (coord[3]-coord[1])/2
    color   = np.array(color).reshape([1,1,3])
    color_r = np.repeat(np.repeat(color,size*2,axis=0),4,axis=1)
    color_c = np.repeat(np.repeat(color,size*2,axis=1),4,axis=0)
    # image = images[i].copy()
    try:
        image[c[0]-size:c[0]+size,c[1]-size-l:c[1]-size+l,:] = color_r
    except:
        pass
    try:
        image[c[0]-size:c[0]+size,c[1]+size-l:c[1]+size+l,:] = color_r
    except:
        pass
    try:
        image[c[0]-size-l:c[0]-size+l,c[1]-size:c[1]+size,:] = color_c
    except:
        pass
    try:
        image[c[0]+size-l:c[0]+size+l,c[1]-size:c[1]+size,:] = color_c
    except:
        pass
    return image


def draw_border(image,l=5,color=[255,0,0]):
    color = np.array(color)
    #if not np.any(color>1):
        #color = (255*color).astype('uint8')
    
    for c in range(3):
        image[:,:l,c] = color[c]
        image[:,-l:,c]= color[c]
        image[:l,:,c] = color[c]
        image[-l:,:,c]= color[c]
    
    return image


def fig2data(fig):
    fig.tight_layout()
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


def load_image(path,v,f,c=None):
    im_path = (path+v+'/%06d.jpg'%f) 
    if not os.path.exists(im_path):
        im_path = (path+v+'/%d.jpg'%f)
    if not os.path.exists(im_path):
        return 128*np.ones([128,128,3],'uint8')
    im = Image.open(im_path)
    return np.array(im.resize((128,128)))


