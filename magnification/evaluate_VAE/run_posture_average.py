#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 2.7
@author: Biagio Brattoli 
biagio.brattoli@iwr.uni-heidelberg.de
Last Update: 23.8.2018

Use Generative Model for posture extrapolation
"""
from datetime import datetime
import os, sys, numpy as np, argparse
from time import time
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

#from skimage.transform import resize
from scipy.io import loadmat
from scipy.misc import imread
from scipy.spatial.distance import euclidean, cdist
from PIL import Image
import imageio
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import normalize
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
except:
    from sklearn.lda import LDA

from sklearn.svm import LinearSVC
import peakutils

from utils import load_table,load_features, draw_border
sys.path.append('./magnification/')
from Generator import Generator, find_differences

import config_pytorch as cfg
#import config_pytorch_human as cfg

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pages",type=int,default=10,
                    help="Number of plots to save")
parser.add_argument("-q", "--queries",type=int,default=10,
                    help="Number of queries per plot")
parser.add_argument("-n", "--nn",type=int,default=100,
                    help="Number of queries per plot")
parser.add_argument("-g", "--gpu",type=int,default=0,
                    help="GPU device to use for image generation")
args = parser.parse_args()

############################################
# 0. Functions
############################################
def load_image(v,f,c=None):
    im_path = (cfg.crops_path+v+'/%06d.jpg'%f) 
    if not os.path.exists(im_path):
        return 255*np.ones([128,128,3],'uint8')
    im = Image.open(im_path)
    return np.array(im.resize((128,128)))


############################################
# 0. Prepare magnifier object
############################################
generator = Generator(z_dim=cfg.encode_dim,path_model=cfg.vae_weights_path)

############################################
# 1. Load sequences and features
############################################
detections = load_table(cfg.detection_file,asDict=False)
det_cohort= np.array(detections['cohort']) # Used for classifier and plots
det_time  = np.array(detections['time'])   # Used for classifier and plots
det_frames= np.array(detections['frames'])
det_videos= np.array(detections['videos'])
uni_videos= np.unique(detections['videos'].values)

print('Load features...')
pos_features,pos_frames,pos_coords,pos_videos = load_features('fc6', cfg.features_path,uni_videos.tolist())


dt = datetime.now()
dt = '{}-{}-{}-{}-{}/'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
results_folder = cfg.results_path+'/magnification/average/'+dt
if not os.path.exists(results_folder): os.makedirs(results_folder)

Q = args.queries
R, C = 3, Q
for P in trange(args.pages,desc='Pages'):
    ############################################
    # 2. Compute nearest neighbor
    ############################################
    queries = np.random.permutation(len(pos_frames))[:Q]
    #queries_images = [load_image(pos_videos[q],pos_frames[q]) for q in queries]
    nearest = cdist(pos_features[queries],pos_features,'cosine').argsort(1)[:,:args.nn]
    results = []
    for nn in tqdm(nearest,desc="Computing averages"):
        images = [load_image(pos_videos[n],pos_frames[n]) for n in nn]
        z = [generator.encode(im,pos_features[n]) for im, n in zip(images,nn)]
        z_app = [za.detach().cpu().numpy() for za, zp in z]
        z_pos = [zp.detach().cpu().numpy() for za, zp in z]
        query = images[0]
        rgb_avg = np.mean(images,0).astype('uint8')
        vae_avg = generator.decode(z_app[0],np.mean(z_pos,0))
        results.append([query,rgb_avg,vae_avg])
    
    ############################################
    # 3. Plot
    ############################################
    _=plt.figure(figsize=(C,R+1))
    gs = gridspec.GridSpec(R,C)
    row1 = [plt.subplot(gs[0,i]) for i in range(C)]
    row2 = [plt.subplot(gs[1,i]) for i in range(C)]
    row3 = [plt.subplot(gs[2,i]) for i in range(C)]
    
    for i in range(len(results)):
        query,rgb_avg,vae_avg = results[i]
        _=row1[i].imshow(query);   _=row1[i].axis('Off')
        _=row2[i].imshow(rgb_avg); _=row2[i].axis('Off')
        _=row3[i].imshow(vae_avg); _=row3[i].axis('Off')
    
    _=row1[Q/2].set_title('Queries',color='black')
    _=row2[Q/2].set_title('Pixel Average',color='blue')
    _=row3[Q/2].set_title('Generated',color='red')
    plt.savefig(results_folder+'page_%d.png'%(P))
    plt.savefig(results_folder+'page_%d.eps'%(P))
    #plt.show()
    plt.close('all')

