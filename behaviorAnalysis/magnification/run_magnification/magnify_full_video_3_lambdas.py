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

from skimage.transform import resize
from skimage import measure
from scipy.io import loadmat
from scipy.misc import imread, imsave
from scipy.spatial.distance import euclidean, cdist

from sklearn.preprocessing import normalize
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
except:
    from sklearn.lda import LDA

from sklearn.svm import LinearSVC

from utils import load_table, load_features, draw_border, fig2data, load_image
sys.path.append('./magnification/')
from Generator import Generator, find_differences, find_differences_cc

#import config_pytorch as cfg
import config_pytorch_human as cfg

parser = argparse.ArgumentParser()
parser.add_argument("-nn", "--nn",type=int,default=10,
                    help="Nearest neighbor for posture average")
parser.add_argument("-l", "--lambdas",type=float,default=2.5,
                    help="Extrapolation factor")
parser.add_argument("-g", "--gpu",type=int,default=0,
                    help="GPU device to use for image generation")
#parser.add_argument("-t", "--th",type=float,default=0.12,
                    #help="Deviation difference")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
#args.query = 1599

os.environ["OMP_NUM_THREADS"] = "10" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "10" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "10" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10" # export NUMEXPR_NUM_THREADS=6

#print(cfg.results_path)
#exit()
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
#uni_videos= [v for v in uni_videos if '2kmh' in v]
#uni_videos= [v for v in uni_videos if 'H' in v]
uni_videos= np.array([v for v in uni_videos if os.path.isdir(cfg.crops_path+v)])

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
for video in tqdm(uni_videos):
    selection_frames = np.where(pos_videos==video)[0][:200]
    frame, coord = pos_frames[selection_frames], pos_coords[selection_frames]
    feat = pos_features[selection_frames]
    F = len(frame)
    # Look for NN in the healthy and impaired postures, 
    # then average over NN in the VAE space
    sel_h = h_pos_videos!=video
    h_pos_feat_sel = h_pos_feat[sel_h]
    h_pos_frames_sel = h_pos_frames[sel_h]
    h_pos_videos_se = h_pos_videos[sel_h]
    
    healthy_nn = cdist(feat,h_pos_feat_sel,'cosine').argsort(1)[:,:args.nn]
    impaired_nn= cdist(feat,i_pos_feat,'cosine').argsort(1)[:,:1]#args.nn]
    ############################################
    perimage_folder= cfg.results_path+'/magnification/magnification_pervideo/%s/impaired_lambda_%.2f/'%(video,args.lambdas)
    if not os.path.exists(perimage_folder): 
        os.makedirs(perimage_folder)
   
    for f, fr in enumerate(frame):
        healthy_seq = [load_image(cfg.crops_path,h_pos_videos_se[nn],h_pos_frames_sel[nn]) 
                       for nn in healthy_nn[f]]
        impaired_seq= [load_image(cfg.crops_path,i_pos_videos[nn],i_pos_frames[nn]) 
                       for nn in impaired_nn[f]]
        healthy_res, impaired_res, magnified_res = generator.extrapolate_multiple(
                        healthy_seq, h_pos_feat_sel[healthy_nn[f]],
                        impaired_seq,i_pos_feat[impaired_nn[f]],
                        [0.0,1.0,args.lambdas])
        _=imsave(perimage_folder+'/%06d.png'%(fr),magnified_res)

