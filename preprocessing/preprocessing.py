# --------------------------------------------------------
# Code for Python 2.7
# Written by Uta Buechler
# --------------------------------------------------------

import os
import config_pytorch as cfg
from glob import glob
from tqdm import tqdm,trange
import subprocess
import warnings
from utils import run_on_system, load_table
from PIL import Image
import pandas as pd
import numpy as np

############################################
# 0. Create the list of videos we want to utilize
############################################
vid_list = pd.DataFrame(index=None, columns=['videos'])
video_files = []
for root, dirnames, filenames in os.walk(cfg.video_path):
    video_files.extend(glob(root + "/*."+cfg.video_format))
    
for vid in video_files:
    vid_list = vid_list.append({'videos': vid[len(cfg.video_path):-(len(cfg.video_format)+1)]}, ignore_index=True)
    

vid_list.to_csv(cfg.video_path+'/vid_list.csv')

############################################
# 1. Extract the frames
############################################
frames_crop = load_table(cfg.frames_crop,index='videos')#load the cropping coordinates in dictionary format

#find all video files in the given directory
video_files = []
for root, dirnames, filenames in os.walk(cfg.video_path):
    video_files.extend(glob(root + "/*."+cfg.video_format))
    
N = len(video_files)
print('Number of found videos: %i \n Start extracting frames...'%N)

for v in trange(N):
    vid_fullPath = video_files[v]
    vid = vid_fullPath[len(cfg.video_path):-(len(cfg.video_format)+1)]
    
    #define output folder
    frames_path_vid = cfg.frames_path+vid
    #create output folder
    if not os.path.isdir(frames_path_vid):
        os.makedirs(frames_path_vid)
    else:#check if we even need to extract the frames here or if all frames are already available
        nrFramesTotal = int(subprocess.check_output('ffprobe -show_packets %s 2>/dev/null | grep video | wc -l'%vid_fullPath.replace(" ","\ "), shell=True))#get number of frames in video
        l = len([name for name in os.listdir(frames_path_vid) if os.path.isfile(name)])#check how many frames are already extracted
        if nrFramesTotal==l:
            continue
    
    
    ### extract the frames as jpg in best quality ###
    command = 'ffmpeg -i %s -qscale 0 %s'%(vid_fullPath.replace(" ", "\ "),frames_path_vid.replace(" ","\ ")+'/%06d.jpg')#default frame extraction without cropping
    if isinstance(frames_crop,dict) and (vid in frames_crop):#get the cropping from a dictionary
        crop = [frames_crop[vid]['x2']-frames_crop[vid]['x1'],frames_crop[vid]['y2']-frames_crop[vid]['y1'],frames_crop[vid]['x1'],frames_crop[vid]['y1']]
        command = 'ffmpeg -i %s -qscale 0 -vf "crop=%i:%i:%i:%i" %s'%(vid_fullPath.replace(" ", "\ "),crop[0],crop[1],crop[2],crop[3],frames_path_vid.replace(" ","\ ")+'/%06d.jpg')
    elif isinstance(frames_crop,list):#all frames have the same cropping
        command = 'ffmpeg -i %s -qscale 0 -vf "crop=%i:%i:%i:%i" %s'%(vid_fullPath.replace(" ", "\ "),frames_crop[0],frames_crop[1],frames_crop[2],frames_crop[3],frames_path_vid.replace(" ","\ ")+'/%06d.jpg')
    elif frames_crop is not None:#frames_crop is not empty, but still not in the correct format or no cropping for vid is available
        warnings.warn('Video %s[%i/%i]: Cropping cannot be read, full frame will be extraced'%(vid,v,N), Warning)
    
    run_on_system(command)
    
############################################
# 2. Extract Optical Flow
############################################

############################################
# 3. Apply a tracking approach
############################################

#Here comes your tracking approach

#### in case the full frames (saved in cfg.frames_path) should be used, the following code creates the file 'detections.csv' with default values

#create file
detections = pd.DataFrame(index=None, columns=['videos','frames','x1','y1','x2','y2','group'])
frames_format = 'jpg'

#find all video files in the given directory
video_files = []
for root, dirnames, filenames in os.walk(cfg.video_path):
    video_files.extend(glob(root + "/*."+cfg.video_format))

N = len(video_files)
for v in trange(N):
    vid_fullPath = video_files[v]
    vid = vid_fullPath[len(cfg.video_path):-(len(cfg.video_format)+1)]
    frames_path_vid = cfg.frames_path+vid
    
    if not os.path.isdir(frames_path_vid):
        continue
    
    
    #find all frames
    frame_files = glob(frames_path_vid+"/*."+frames_format)
    frame_files.sort()
    if len(frame_files[0])!=len(frame_files[2]):#if the frames are not saved with leading 0 the sorting does not work out of the box (it is sorted lexicographically (in this way: 0,1,10,100,101,102 etc))
        for f in range(len(frame_files)):
            frame = int(frame_files[f][len(frames_path_vid)+1:-(len(frames_format)+1)])
            frame_files[f] = '%s/%06i.%s'%(frames_path_vid,frame,frames_format)
        frame_files.sort()
    
    for f in range(len(frame_files)):
        frame = int(frame_files[f][len(frames_path_vid)+1:-(len(frames_format)+1)])
        detections = detections.append({'videos': vid,'frames': frame,'x1':0, 'y1':0, 'x2':-1,'y2':-1,'group':0}, ignore_index=True)
        
detections.to_csv(cfg.detection_file)


##################################################
# 4. Save the jpgs as cropped versions,
#    so that the training process of the CNN is faster
#################################################

detections = load_table(cfg.detection_file,asDict=False)

#include new column to save the size of the original frames (comment out if already saved)
#detections = detections.assign(original_size_x=pd.Series(np.zeros(len(detections),np.int)).values)
#detections = detections.assign(original_size_y=pd.Series(np.zeros(len(detections),np.int)).values)

videos = np.unique(list(detections['videos']))
N_vids = len(videos)

for v in range(N_vids):
    idx = [i for i,vid in enumerate(detections['videos'].values) if vid==videos[v]]#get all entries for the current video
    frames = detections['frames'][idx].values#get all frames where we have detections
    coords = np.array([detections['x1'][idx].values,detections['y1'][idx].values,#get all detections
            detections['x2'][idx].values,detections['y2'][idx].values]).T
    
    crops_path_vid = cfg.crops_path+videos[v]
    if not os.path.isdir(crops_path_vid):
       os.makedirs(crops_path_vid)
    
    frames_path_vid = cfg.frames_path+videos[v]
    if os.path.isdir(frames_path_vid+'/deinterlaced'):
        frames_path_vid = '%s/deinterlaced/'%(frames_path_vid)
    
    
    for j,(f,ci) in enumerate(tqdm(zip(frames,coords))):
        c = ci.astype(int)#make sure that it is int
        
        #image = Image.open('%s%s/%06d.jpg'%(cfg.frames_path,videos[v],f))#axis changed: x:0, y:1
        image = Image.open('%s/%06d.jpg'%(frames_path_vid,f))#axis changed: x:0, y:1
        s = image.size
        c[0],c[1],c[2],c[3] = max(0,c[0]),max(0,c[1]),min(s[0],c[2]),min(s[1],c[3])#make sure that the bounding box is not outside of the image
        
        #crop 5% more on every side for augmentation later during training
        w = c[2]-c[0]
        h = c[3]-c[1]
        
        c_new = np.copy(c)
        c_new[0],c_new[1] = max(0,c[0]-int(0.05*w)),max(0,c[1]-int(0.05*h))
        c_new[2],c_new[3] = min(c[2]+int(0.05*w),s[0]),min(c[3]+int(0.05*h),s[1])
        
        img_crop = image.crop(c_new)
        img_crop.save('%s/%06d.jpg'%(crops_path_vid,f))
        
        #we need to save the original size of the frame for later (comment out if already saved; there appears a 'SettingWithCopyWarning' which can be ignored)
        #detections['original_size_x'][idx[j]] = s[0]
        #detections['original_size_y'][idx[j]] = s[1]
        
#save the detections as csv file, so that it is updated with the new columns (comment out if already saved)
#detections.to_csv(cfg.detection_file)

