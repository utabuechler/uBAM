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
#from scipy.misc import imread
from skimage.io import imread, imsave
from skimage.transform import resize

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

from utils import load_table,load_features, draw_border, fig2data, load_image
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
#parser.add_argument("-g", "--gpu",type=int,default=0,
#                    help="GPU device to use for image generation")
args = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)

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

#print('Load features...')
#pos_features,pos_frames,pos_coords,pos_videos = load_features('fc6', cfg.features_path,uni_videos.tolist())
#seq_features,seq_frames,seq_coords,seq_videos = load_features('lstm',cfg.features_path,uni_videos.tolist())

############################################
# 2. Evaluate disentanglement of posture and appearance
############################################
def get_video(path, video, frames):
    return np.stack([resize(load_image(path,video,f), (128, 128)) for f in frames])
    

dt = datetime.now()
dt = '{}-{}-{}-{}-{}/'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
results_folder = cfg.results_path+'/magnification/evaluation_video_without_vae/'+dt
if not os.path.exists(results_folder): os.makedirs(results_folder)

for P in trange(args.pages,desc='Videos'):
    Q = args.queries
#    queries = np.linspace(0, len(uni_videos)-1, Q*2) #np.random.permutation(len(uni_videos))
#    app_queries, pos_queries = queries[:Q], queries[Q:Q*2]
    queries = np.linspace(0, len(det_frames)-1000, Q*2).astype(int)
    videos = np.stack([get_video(cfg.crops_path, det_videos[q], det_frames[q:q+args.length]) 
            for q in tqdm(queries, desc='load videos')])
    appearance = videos.mean(1)
    postures = [v - np.repeat(a[np.newaxis], v.shape[0], axis=0) for v, a in zip(videos, appearance)]
    appearance, postures = appearance[:Q], np.stack(postures[Q:Q*2])
    pos_images = videos[Q:Q*2]
#    print(postures.shape)
#    print(appearance.shape)
    ############################################
    # 3. Plot
    ############################################
    _=plt.figure(figsize=(11,11))
    plt.tight_layout()
    R, C = Q+2, Q+1
    gs = gridspec.GridSpec(R,C)
    first_row = [plt.subplot(gs[0,i+1]) for i in range(C-1)]
    second_row = [plt.subplot(gs[1,i+1]) for i in range(C-1)]
    first_column = [plt.subplot(gs[i+2,0]) for i in range(R-2)]
    middle = [[plt.subplot(gs[i+2,j+1]) for j in range(C-1)]  for i in range(R-2)]
    first_row[Q//2].set_title('Query Appearance',color='black',fontsize=30)
    for i, im in enumerate(appearance): _=first_row[i].imshow(im); _=first_row[i].axis('Off')
    for i, im in enumerate(videos[:Q, 0]): _=second_row[i].imshow(im); _=second_row[i].axis('Off')
    
    video = []
    for f in trange(args.length):
        for i, im in enumerate(pos_images[:, f]): _=first_column[i].imshow(im); _=first_column[i].axis('Off')
        
        for i in range(Q):
            for j in range(Q):
                im = postures[i, f] + appearance[j]
#                im = (im - im.min())/ (im.max() - im.min())
                im[im < 0.0] = 0.0
                im[im > 1.0] = 1.0
                _=middle[i][j].imshow(im); _=middle[i][j].axis('Off')
        
        first_column[Q//2].set_title('Query Posture',color='blue',rotation='vertical',x=-0.3,y=0.8,fontsize=30)
        if f % 10 == 0:
            plt.savefig(results_folder+'/video%d_%d.png'%(P, f))
        #middle[0][Q/2].set_title('Generated',color='red',fontsize=30)
#        frame = fig2data(plt.gcf())
        fig = plt.gcf()
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer._renderer)
#        print(frame.shape, frame.dtype)
        video.append(frame)
    
    imageio.mimsave(results_folder+'video%d.gif'%(P),video,fps=5)
    try:
        writer = imageio.get_writer(results_folder+'/video%d.mp4'%(P),fps=6)
        for img in video: 
            im = resize(img, (512, 512))
            im = (im * 255).astype('uint8')
            writer.append_data(im)
        
        writer.close()
    except:
        continue

