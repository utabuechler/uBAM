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

#from skimage.transform import resize
from skimage import measure
from scipy.io import loadmat
from scipy.misc import imread
from scipy.spatial.distance import euclidean, cdist

from sklearn.preprocessing import normalize
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
except:
    from sklearn.lda import LDA

from sklearn.svm import LinearSVC

from utils import load_table, load_features, draw_border, fig2data, load_image
sys.path.append('./magnification/')
from Generator2 import Generator, find_differences, find_differences_cc

#import config_pytorch as cfg
#import config_pytorch_human as cfg
import config_pytorch_rats_biagio as cfg

# python ./magnification/run_magnification/random_queries_posture_magnification_2.py -q 30 -nn 5 -g 0 -t 0.15
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--length",type=int,default=8,
                    help="Sequence length")
parser.add_argument("-q", "--queries",type=int,default=10,
                    help="Frame query")
parser.add_argument("-nn", "--nn",type=int,default=30,
                    help="Nearest neighbor for posture average")
parser.add_argument("-l", "--lambdas",type=float,default=2.5,
                    help="Extrapolation factor")
parser.add_argument("-g", "--gpu",type=int,default=0,
                    help="GPU device to use for image generation")
parser.add_argument("-t", "--th",type=float,default=0.12,
                    help="Deviation difference")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
#args.query = 1599

############################################
# 0. Prepare magnifier object
############################################
generator = Generator(z_dim=cfg.vae_encode_dim,path_model=cfg.vae_weights_path)

############################################
# 1. Load sequences and features
############################################
detections = load_table(cfg.detection_file,asDict=False)
det_cohort= np.array(detections['cohort']) # Used for classifier and plots
det_time  = np.array(detections['time'])   # Used for classifier and plots
det_frames= np.array(detections['frames'])
det_videos= np.array(detections['videos'])
uni_videos= np.unique(detections['videos'].values)
uni_videos = np.array([v for v in uni_videos if os.path.isdir(cfg.crops_path+v)])

print('Load features...')
pos_features,pos_frames,pos_coords,pos_videos = load_features('fc6', cfg.features_path,uni_videos.tolist())

############################################
# 2. Posture healthy/impaired assignment
############################################
video_time  =np.array([det_time[det_videos==v][0]   for v in uni_videos]).astype(int)
video_cohort=np.array([det_cohort[det_videos==v][0] for v in uni_videos]).astype(int)
pos_time    =np.concatenate([video_time[uni_videos==v] for v in pos_videos])

healthy, impaired = pos_time==0, pos_time==1
h_pos_feat,  h_pos_videos= pos_features[healthy], pos_videos[healthy]
h_pos_frames,h_pos_coords= pos_frames[healthy], pos_coords[healthy]
i_pos_feat,  i_pos_videos= pos_features[impaired], pos_videos[impaired]
i_pos_frames,i_pos_coords= pos_frames[impaired], pos_coords[impaired]

############################################
# 3. Magnify sequences by averaging over nearest neighbor
############################################
#query = args.query
#print('Producing magnification: %d query'%(query))
queries = np.linspace(200,pos_features.shape[0]-200,args.queries).astype(int)
queries.sort()
for query in tqdm(queries):
    F = args.length
    selection_frames = np.arange(query-F/2,query+F/2).astype('int')
    video, frame, coord = pos_videos[query], pos_frames[selection_frames], pos_coords[selection_frames]
    feat = pos_features[selection_frames]
    
    # Look for NN in the healthy and impaired postures, 
    # then average over NN in the VAE space
    healthy_nn = cdist(feat,h_pos_feat,'cosine').argsort(1)[:,:args.nn]
    impaired_nn= cdist(feat,i_pos_feat,'cosine').argsort(1)[:,:args.nn//2]
    healthy_seq = [[load_image(cfg.crops_path,h_pos_videos[nn],h_pos_frames[nn]) for nn in healthy_nn[f]] for f in range(F)]
    impaired_seq= [[load_image(cfg.crops_path,i_pos_videos[nn],i_pos_frames[nn]) for nn in impaired_nn[f]] for f in range(F)]
    appearances = [load_image(cfg.frames_path,i_pos_videos[f[0]],i_pos_frames[f[0]]) 
                for f in impaired_nn]
    
    ############################################
    # 4. Plot
    ############################################
    #j, f = 0, frame[0]
    dt = datetime.now()
    dt = '{}-{}-{}/'.format(dt.year, dt.month, dt.day)
    results_folder = cfg.results_path+'/magnification/magnified/th%.3f_nn%d_'%(args.th,args.nn)+dt
    if not os.path.exists(results_folder): os.makedirs(results_folder)
    
    R, C = 5, F
    _=plt.figure(figsize=(C*4,R*4+1))
    plt.tight_layout()
    gs = gridspec.GridSpec(R,C)
    subplot = [[plt.subplot(gs[i,j]) for j in range(C)] for i in range(R)]
    for f in range(len(impaired_seq)):
        image_app = appearances[f]
        healthy_res, impaired_res, magnified_res = generator.extrapolate_multiple(
                        healthy_seq[f], h_pos_feat[healthy_nn[f]],
                        impaired_seq[f],i_pos_feat[impaired_nn[f]],
                        image_app, [0.0,1.0,args.lambdas])
        diff_image,flow_filtered,X,Y,_ = find_differences_cc(
                    healthy_res,impaired_res,magnified_res,Th=args.th,scale=20)
        healthy_res  = draw_border(healthy_res, l=2,color=[0,1.0,0])
        impaired_res = draw_border(impaired_res,l=2,color=[0,0,1.0])
        magnified_res= draw_border(magnified_res,l=2,color=[1.0,0,0])
        #diff_image=draw_border(diff_image,l=2,color=[1.0,0,0])
        
        _=subplot[0][f].imshow(healthy_res); _=subplot[0][f].axis('Off')
        if f==0: _=subplot[0][f].set_title('Healthy',fontsize=20,color='green')
        
        _=subplot[1][f].imshow(impaired_res); _=subplot[1][f].axis('Off')
        if f==0: _=subplot[1][f].set_title('Unhealthy',fontsize=20,color='red',x=0.50)
        
        _=subplot[2][f].imshow(magnified_res); _=subplot[2][f].axis('Off')
        if f==0: _=subplot[2][f].set_title('Magnification',fontsize=20,color='red',x=0.70)
        
        _=subplot[3][f].imshow(diff_image); _=subplot[3][f].axis('Off')
        if f==0: _=subplot[3][f].set_title('Difference',fontsize=20,color='red',x=0.55)
        
        _=subplot[4][f].imshow(impaired_res); _=subplot[4][f].axis('Off')
        _=subplot[4][f].quiver(X, Y, flow_filtered[:,0], flow_filtered[:,1], 
                        width=0.1, headwidth=3, color='red', 
                        scale_units='width', scale=10,minlength=0.1)
        if f==0: _=subplot[4][f].set_title('Improvement',fontsize=20,color='red',x=0.70)
    
    plt.savefig(results_folder+'query_%d.png'%(query),bbox_inches='tight',dpi=75)
    plt.savefig(results_folder+'query_%d.eps'%(query),bbox_inches='tight',dpi=75)
    #plt.show()
    plt.close('all')


