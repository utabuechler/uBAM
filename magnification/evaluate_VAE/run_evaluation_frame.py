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
parser.add_argument("-q", "--queries",type=int,default=5,
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

############################################
# 2. Evaluate disentanglement of posture and appearance
############################################
dt = datetime.now()
dt = '{}-{}-{}-{}-{}/'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
results_folder = cfg.results_path+'/magnification/evaluation/'+dt
if not os.path.exists(results_folder): os.makedirs(results_folder)

args.queries, args.pages = 2, 50
Q = args.queries
for P in trange(args.pages,desc="Save plots"):
    queries = np.random.permutation(len(pos_frames))[:Q*2]
    images  = [load_image(pos_videos[q],pos_frames[q]) for q in queries]
    z = [generator.encode(im,pos_features[q]) for im, q in zip(images,queries)]
    #x = generator.decode(z[0][0],z[0][1])
    #generated = np.zeros((Q,Q)+x.shape,x.dtype)
    #for i in range(Q):
        #for j in range(Q):
            #generated[i,j] = generator.decode(z[i][0],z[j][1])
    
    ############################################
    # 3. Plot
    ############################################
    _=plt.figure(figsize=(15,15))
    R, C = Q+1, Q+1
    gs = gridspec.GridSpec(R,C)
    first_row = [plt.subplot(gs[0,i+1]) for i in range(C-1)]
    first_column = [plt.subplot(gs[i+1,0]) for i in range(R-1)]
    middle = [[plt.subplot(gs[i+1,j+1]) for j in range(R-1)]  for i in range(R-1)]
    
    for i, im in enumerate(images[Q:]): _=first_row[i].imshow(im); _=first_row[i].axis('Off')
    for i, im in enumerate(images[:Q]): _=first_column[i].imshow(im); _=first_column[i].axis('Off')
    
    for i in range(Q):
        for j in range(Q,2*Q):
            im = generator.decode(z[i][0],z[j][1])
            _=middle[i][j-Q].imshow(im); _=middle[i][j-Q].axis('Off')
    
    plt.savefig(results_folder+'page_%d.png'%(P))
    plt.savefig(results_folder+'page_%d.eps'%(P))


#plt.show()
#_=plt.title('Query %d'%(query),fontsize=20)
#
#plt.close('all')

