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
import matplotlib.gridspec as gridspec

from scipy.io import loadmat
from skimage.io import imread, imsave
from skimage.transform import resize

from utils import load_table,load_features, draw_border, fig2data, load_image
from sklearn.decomposition import PCA

#import config_pytorch as cfg
import config_pytorch_human as cfg

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--queries",type=int,default=5,
                    help="Number of queries per plot")
parser.add_argument("-g", "--gpu",type=int,default=0,
                    help="GPU device to use for image generation")
args = parser.parse_args()


############################################
# 1. Load sequences and features
############################################
detections = load_table(cfg.detection_file,asDict=False)
det_cohort= np.array(detections['cohort']) # Used for classifier and plots
det_time  = np.array(detections['time'])   # Used for classifier and plots
det_frames= np.array(detections['frames'])
det_videos= np.array(detections['videos'])
uni_videos= np.unique(detections['videos'].values)
uni_videos= uni_videos[::5]

print('Load features...')
pos_features,pos_frames,pos_coords,pos_videos = load_features('fc6', cfg.features_path,uni_videos.tolist())


sel = [v in uni_videos for v in pos_videos]

sel = np.logical_and(pos_frames < 50, sel)
pos_features, pos_frames = pos_features[sel], pos_frames[sel]
pos_coords, pos_videos = pos_coords[sel], pos_videos[sel]

images = [load_image(cfg.crops_path, v, f) 
            for v, f in zip(tqdm(pos_videos, desc='load videos'), pos_frames)]
images = (np.stack([resize(im, (128, 128)) for im in images]) * 255).astype('uint8')


############################################
# TRAIN PCA CLASSIFIER
############################################
X = images.astype('float32') / 255
pca = PCA(n_components=25)
features = pca.fit_transform(X.reshape([images.shape[0], -1]))

############################################
# Evaluate disentanglement of posture and appearance
############################################
dt = datetime.now()
dt = '{}-{}-{}-{}-{}/'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
results_folder = cfg.results_path+'/magnification/evaluate_interpolation/PCA_'+dt
if not os.path.exists(results_folder): os.makedirs(results_folder)

Q = args.queries
#for P in trange(args.pages,desc="Save plots"):
#    queries = np.random.permutation(len(pos_frames))[:Q*2]
#queries = np.linspace(0, len(pos_frames), Q).astype(int)
#references = np.linspace(len(pos_frames)//2, len(pos_frames)-1, Q).astype(int)
queries = [np.where(pos_videos==v)[0][0] for v in uni_videos]
queries = queries[::len(queries)//Q]

############################################
# 3. Plot
############################################
_=plt.figure(figsize=(8,Q*2))
R, C = Q, 4
gs = gridspec.GridSpec(R,C)
grid = [[plt.subplot(gs[i,j]) for j in range(C)] for i in range(R)]

#    first_row[Q//2].set_title('Query Posture',color='blue',fontsize=30)
for i in range(R):
#    app = z_s[i][0]
#    pos = (z_s[i][1] + z_e[i][1]) / 2
#    im = generator.decode(app, pos)
#    im = (im * 255).astype('uint8')
    pos = (features[queries[i]] + features[queries[i]+5]) / 2
    im = pca.inverse_transform(pos).reshape([128, 128, 3])
    im[im<0] = 0
    im[im>1] = 1
#    im = (im * 255).astype('uint8')
    _=grid[i][0].imshow(images[queries[i]])
    _=grid[i][1].imshow(images[queries[i]+3])
    _=grid[i][2].imshow(im)
    _=grid[i][3].imshow(images[queries[i]+5])
    for j in range(C): _=grid[i][j].axis('Off')

grid[0][0].set_title('T=0')#,color='black',rotation='horizontal',x=-0.3,y=0.8,fontsize=30)
grid[0][1].set_title('T=3')
grid[0][2].set_title('Interpolation')
grid[0][3].set_title('T=5')
plt.tight_layout()
plt.savefig(results_folder+'pca_interpolation.png')
plt.savefig(results_folder+'pca_interpolation.eps')


