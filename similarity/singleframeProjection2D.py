#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 2.7
@author: Biagio Brattoli 
biagio.brattoli@iwr.uni-heidelberg.de
Last Update: 23.8.2018

Use the trained features to plot the data on a 2D space
"""

import os, sys, numpy as np, argparse
from time import time
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.misc import imread

from joblib import Parallel, delayed
import multiprocessing
CORES = multiprocessing.cpu_count()

from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from scipy.stats import kde

from utils import load_table, load_features

try:
    import umap
    IS_UMAP=True
except ImportError:
    IS_UMAP=False
    print('!!UMAP not installed, tSNE will be used!!')

import config_pytorch as cfg
#import config_pytorch_human as cfg

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--samples",type=int,default=100,
                    help="max number of plot")
parser.add_argument("-t", "--train_samples",type=int,default=200,
                    help="number of samples for training the projection")
parser.add_argument("-a", "--algorithm",type=str,default='umap',
                    help="algorithm for dimensionality reduction, ['tsne','umap']")
parser.add_argument("-ft", "--feature_type",type=str,default='fc6',
                    help="type of features, 'fc6','fc7' or 'fc6fc7' for postures")
parser.add_argument("-c", "--cores",type=int,default=-1,
                    help="number of cores when loading images. -1: half of available cores, 0: single core")
parser.add_argument("-sc", "--scale",type=int,default=1,
                    help="scale distance between images in the 2D plot")
args = parser.parse_args()

#if __name__ == "__main__":
############################################
# 1. Load Features
############################################
results_fold = cfg.results_path+'/similarity/singleFrame/'
if not os.path.exists(results_fold): os.makedirs(results_fold)

detections = load_table(cfg.detection_file,asDict=False)
uni_videos = np.unique(detections['videos'].values)
print('Load features...')
features,frames,coords,videos = load_features(args.feature_type,cfg.features_path,uni_videos.tolist())

############################################
# 2. Select samples for reducing computation
############################################
sel = np.linspace(0,features.shape[0]-1,
                min(args.train_samples,features.shape[0])).astype(int)
videos, frames, coords = videos[sel], frames[sel], coords[sel].astype(int)
features = normalize(features[sel])

############################################
# 3. Dimensionality Reduction
############################################
if args.algorithm=='tsne' or not IS_UMAP:
    clf = TSNE(n_components=2, perplexity=30.0, init='pca')
elif args.algorithm=='umap':
    clf = umap.UMAP(n_neighbors=15,metric='cosine')

t = time()
points = clf.fit_transform(features)
print('Dimensionality Reduction done in %.1fsec'%(time()-t))
np.savez(results_fold+'/projections2D_%s'%(args.algorithm),
                    points=points,videos=videos,
                    frames=frames,coords=coords)

x1, x2 = points[:,0].min()-0.1, points[:,0].max()+0.1
y1, y2 = points[:,1].min()-0.1, points[:,1].max()+0.1

############################################
# 4. Plot 2D Density
############################################
nbins = 100
k = kde.gaussian_kde(points.T)
xi, yi = np.mgrid[x1:x2:nbins*1j, y1:y2:nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

_=plt.figure(figsize=(14,10))
_=plt.pcolormesh(xi, yi, zi.reshape(xi.shape),cmap=plt.cm.viridis)
_=plt.colorbar()
#plt.contour(xi, yi, zi.reshape(xi.shape) )
_=plt.title('Single Frame Density',fontsize=20)
_=plt.xlim([x1,x2])
_=plt.ylim([y1,y2])
_=plt.savefig(results_fold+'/projection_2D_density_%s_training%d.png'%(
                    args.algorithm,args.train_samples),
                    bbox_inches='tight',dpi=60)
# plt.show()

############################################
# 5. Plot 2D Manifold
############################################
def load_image(v,f,c):
    im_path = (cfg.frames_path+v+'/%06d.jpg'%f) 
    if not os.path.exists(im_path): 
        im_path = cfg.frames_path+v+'/%d.jpg'%f
    
    im = imread(im_path)
    return im[c[1]:c[3],c[0]:c[2]]

sel = np.linspace(0,features.shape[0]-1,
                min(args.samples,features.shape[0])).astype(int)
videos, frames, coords = videos[sel], frames[sel], coords[sel]
points = points[sel]

# LOAD IMAGES
if args.cores!=0:
    cores = CORES/2 if args.cores==-1 else min(args.cores,CORES)
    with Parallel(n_jobs=cores) as parallel:
        images = parallel(delayed(load_image)(videos[i],frames[i],coords[i]) 
                    for i in trange(len(videos),desc='Loading images'))
else:
    images = [load_image(videos[i],frames[i],coords[i]) 
                            for i in trange(len(videos))]

# PLOT
sc = args.scale*points.shape[0]
_=plt.figure(figsize=(25,25))
for i, im in enumerate(tqdm(images,desc='Plotting')):
    pos = sc*points[i]
    l,r = pos[0]-im.shape[0]/2,pos[0]+im.shape[0]/2
    b,t = pos[1]-im.shape[1]/2,pos[1]+im.shape[1]/2
    _=plt.imshow(im,extent=(l,r,b,t))

_=plt.xlim([sc*x1,sc*x2])
_=plt.ylim([sc*y1,sc*y2])
_=plt.savefig(results_fold+'/projection_2D_images_%s_samples%d.png'%(args.algorithm,args.samples),
                bbox_inches='tight', dpi=150)

