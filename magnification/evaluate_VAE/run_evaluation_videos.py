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
import imageio

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

#import config_pytorch as cfg
import config_pytorch_human as cfg

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pages",type=int,default=20,
                    help="Number of videos to plot")
parser.add_argument("-q", "--queries",type=int,default=5,
                    help="Number of queries per plot")
parser.add_argument("-l", "--length",type=int,default=50,
                    help="Number of frames")
parser.add_argument("-g", "--gpu",type=int,default=0,
                    help="GPU device to use for image generation")
args = parser.parse_args()

############################################
# 0. Functions
############################################
def load_image(v,f,c=None):
    im_path = (cfg.crops_path+v+'/%06d.jpg'%f) 
    if not os.path.exists(im_path):
        im_path = (cfg.crops_path+v+'/%d.jpg'%f)
    if not os.path.exists(im_path):
        return 128*np.ones([128,128,3],'uint8')
    im = Image.open(im_path)
    return np.array(im.resize((128,128)))


def fig2data(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

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

#uni_videos = np.array([v for v in uni_videos if u'2kmh' in v])

print('Load features...')
pos_features,pos_frames,pos_coords,pos_videos = load_features('fc6', cfg.features_path,uni_videos.tolist())
#seq_features,seq_frames,seq_coords,seq_videos = load_features('lstm',cfg.features_path,uni_videos.tolist())

############################################
# 2. Evaluate disentanglement of posture and appearance
############################################
dt = datetime.now()
dt = '{}-{}-{}-{}-{}/'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
results_folder = cfg.results_path+'/magnification/evaluation_video/'+dt
if not os.path.exists(results_folder): os.makedirs(results_folder)

for P in trange(args.pages,desc='Videos'):
    Q = args.queries
    queries = np.random.permutation(len(pos_frames))
    app_queries, pos_queries = queries[:Q], queries[Q:Q*2]
    app_images  = [load_image(pos_videos[q],pos_frames[q]) for q in app_queries]
    z_app = [generator.encode(im,pos_features[q])[0] for im, q in zip(app_images,app_queries)]
    ############################################
    # 3. Plot
    ############################################
    _=plt.figure(figsize=(11,11))
    plt.tight_layout()
    R, C = Q+1, Q+1
    gs = gridspec.GridSpec(R,C)
    first_row = [plt.subplot(gs[0,i+1]) for i in range(C-1)]
    first_column = [plt.subplot(gs[i+1,0]) for i in range(R-1)]
    middle = [[plt.subplot(gs[i+1,j+1]) for j in range(C-1)]  for i in range(R-1)]
    first_row[Q/2].set_title('Query Appearance',color='black',fontsize=30)
    for i, im in enumerate(app_images): _=first_row[i].imshow(im); _=first_row[i].axis('Off')
    
    video = []
    for f in trange(args.length):
        pos_images  = [load_image(pos_videos[q+f],pos_frames[q+f]) for q in pos_queries]
        z_pos = [generator.encode(im,pos_features[q+f])[1] for im, q in zip(pos_images,pos_queries)]
        for i, im in enumerate(pos_images): _=first_column[i].imshow(im); _=first_column[i].axis('Off')
        
        for i in range(Q):
            for j in range(Q):
                im = generator.decode(z_app[j],z_pos[i])
                _=middle[i][j].imshow(im); _=middle[i][j].axis('Off')
        
        first_column[Q/2].set_title('Query Posture',color='blue',rotation='vertical',x=-0.3,y=0.8,fontsize=30)
        #middle[0][Q/2].set_title('Generated',color='red',fontsize=30)
        frame = fig2data(plt.gcf())
        video.append(frame)
    
    imageio.mimsave(results_folder+'video%d.gif'%(P),video,fps=5)
    try:
        writer = imageio.get_writer(results_folder+'/video%d.mp4'%(P),fps=6)
        for img in video: 
            writer.append_data(img)
        
        writer.close()
    except:
        continue

