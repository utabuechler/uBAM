#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 2.7
@author: Biagio Brattoli 
biagio.brattoli@iwr.uni-heidelberg.de
Last Update: 23.8.2018

Use Generative Model for posture extrapolation
"""

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

from sklearn.preprocessing import normalize
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
except:
    from sklearn.lda import LDA

from sklearn.svm import LinearSVC
import peakutils

from utils import load_table,load_features, draw_box
sys.path.append('./magnification/run_magnification/')
from Magnifier import Magnifier, find_differences

import config_pytorch as cfg

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--samples",type=int,default=20000,
                    help="max number of samples to show in the 2D plot")
parser.add_argument("-q", "--queries",type=int,default=5,
                    help="Number of query sequences to plot")
parser.add_argument("-nn", "--nn",type=int,default=5,
                    help="Nearest neighbor for posture average")
parser.add_argument("-l", "--lambdas",type=float,default=2.5,
                    help="Extrapolation factor")
parser.add_argument("-g", "--gpu",type=int,default=0,
                    help="GPU device to use for image generation")
args = parser.parse_args()

############################################
# 0. Functions
############################################

def load_image(v,f,c=None):
    im = Image.open('%s%s/%06d.jpg'%(cfg.frames_path,v,f))
    return np.array(im.resize((128,128)))
    #im_path = (cfg.frames_path+v+'/%06d.jpg'%f) 
    #if not os.path.exists(im_path): 
        #im_path = cfg.frames_path+v+'/%d.jpg'%f
    
    #im = imread(im_path)
    #if c is not None:
        #im = im[c[1]:c[3],c[0]:c[2]]
    #return resize(im, (128, 128))


############################################
# 0. Prepare magnifier object
############################################
magnifier = Magnifier(z_dim=cfg.encode_dim,path_model=cfg.vae_weights_path)

############################################
# 1. Load sequences and features
############################################
results_folder = cfg.results_path+'/magnification/magnified/'
if not os.path.exists(results_folder): os.makedirs(results_folder)

detections = load_table(cfg.detection_file,asDict=False)
det_cohort= np.array(detections['cohort']) # Used for classifier and plots
det_time  = np.array(detections['time'])   # Used for classifier and plots
det_frames= np.array(detections['frames'])
det_videos= np.array(detections['videos'])
uni_videos= np.unique(detections['videos'].values)

print('Load features...')
seq_features,seq_frames,seq_coords,seq_videos = load_features('lstm',cfg.features_path,uni_videos.tolist())
pos_features,pos_frames,pos_coords,pos_videos = load_features('fc6', cfg.features_path,uni_videos.tolist())

seq_features = seq_features.reshape([seq_features.shape[0],cfg.seq_len,-1])
seq_features = seq_features[:,[cfg.seq_len/2-1,-1],:].reshape([seq_features.shape[0],-1])

############################################
# 2. Select high density sequences to remove outliers
############################################
print('Removing outliers')
uni_videos= np.unique(detections['videos'].values)
count = np.array([(seq_videos==v).sum() for v in uni_videos])
#pos_count = np.array([(pos_videos==v).sum() for v in uni_videos])
uni_videos = uni_videos[count>0.20*count.mean()]
V = len(uni_videos)

samples_per_video = int(np.ceil(float(args.samples)/V))
selection = []
for v in uni_videos:
    sel = np.where(seq_videos==v)[0]
    sel = sel[np.random.permutation(len(sel))]
    for s in sel[:samples_per_video]:
        selection.append(s)

selection = np.array(selection[:args.samples])

seq_feat_norm = normalize(seq_features[selection])
similarity = seq_feat_norm.dot(seq_feat_norm.T)
#plt.hist(similarity.flatten(),100); plt.show()
sorted_sim = np.sort(similarity,1)[:,::-1]
density = sorted_sim[:,1:int(0.05*similarity.shape[1])].mean(1)
#plt.hist(density,100); plt.show()
sel_dense = density>density.mean()-2*density.std()
selection = selection[sel_dense]

# features,frames,coords,videos = \
#     pos_features[selection],pos_frames[selection],\
#     pos_coords[selection],pos_videos[selection]

features,frames,coords,videos = \
    seq_features[selection],seq_frames[selection],\
    seq_coords[selection],seq_videos[selection]

print('Number of samples used for nearest neighbor search: %d'%(features.shape[0]))

############################################
# 3. Linear classifier for Healthy/Impaired sequence selection
############################################
video_time  =np.array([det_time[detections['videos']==v][0]   for v in uni_videos])
video_cohort=np.array([det_cohort[detections['videos']==v][0] for v in uni_videos])

healthy_videos, impaired_videos = uni_videos[video_time==0], uni_videos[video_time==1]
healthy_cohort, impaired_cohort = uni_videos[video_time==0], uni_videos[video_time==1]

# Select healthy and impaired sequences
healthy = []
for v in healthy_videos[np.random.permutation(len(healthy_videos))]:
    for s in np.where(videos==v)[0]: 
        healthy.append(s)

impaired = []
for v in impaired_videos[np.random.permutation(len(impaired_videos))]:
    for s in np.where(videos==v)[0]: 
        impaired.append(s)

# Select samples for training and validation
# Trainin data have the same number of samples for positives and negatives
n   = int(0.80*min(len(healthy), len(impaired)))
train_pos, train_neg = healthy[:n], impaired[:n]
X_train = features[train_pos+train_neg]
y_train = np.array([1]*len(train_pos)+[-1]*len(train_neg))
val_pos, val_neg = healthy[n:], impaired[n:]
X_val = features[val_pos+val_neg]
y_val = np.array([1]*len(val_pos)+[-1]*len(val_neg))
# Train and evaluate classifier
clf = LinearSVC(C=1.5).fit(X_train,y_train)
# clf = LDA().fit(X_train,y_train)
train_tp = clf.score(X_train[y_train==1], y_train[y_train==1])
train_tn = clf.score(X_train[y_train==-1],y_train[y_train==-1])
val_tp = clf.score(X_val[y_val==1], y_val[y_val==1])
val_tn = clf.score(X_val[y_val==-1],y_val[y_val==-1])
print 'Classifier Accuracy: Train %.2f (%.2f,%.2f), Val %.2f (%.2f,%.2f)'%(
        (train_tp+train_tn)/2,train_tp,train_tn,(val_tp+val_tn)/2,val_tp,val_tn)

# Use the score given by the classifier for healthy/impaired selection
seq_scores = clf.decision_function(features)
# plt.hist(seq_scores,100); plt.show()
m, s = seq_scores.mean(), seq_scores.std()
seq_impaired = np.where(seq_scores<-seq_scores.std())[0]
seq_healthy  = np.where(seq_scores> seq_scores.std())[0]

filtered_results_folder = results_folder+'/sequences_healthy%d_impaired%d/'%(
                          seq_healthy.shape[0],seq_impaired.shape[0])
if not os.path.exists(filtered_results_folder): os.makedirs(filtered_results_folder)

np.savez(filtered_results_folder+'sequences',seq_healthy=seq_healthy,
         seq_impaired=seq_impaired,seq_scores=seq_scores,train_pos=train_pos,
         train_neg=train_neg,
         features=features,frames=frames,coords=coords,videos=videos,
         video_time=video_time,video_cohort=video_cohort,uni_videos=uni_videos)

############################################
# 3. Posture healthy/impaired assignment
############################################
pos_healthy = []
for v in healthy_videos:#[np.random.permutation(len(healthy_videos))]:
    for s in np.where(pos_videos==v)[0]:
        pos_healthy.append(s)

pos_impaired = []
for v in impaired_videos:#[np.random.permutation(len(impaired_videos))]:
    for s in np.where(pos_videos==v)[0]: 
        pos_impaired.append(s)

h_pos_feat,  h_pos_videos= pos_features[pos_healthy], pos_videos[pos_healthy]
h_pos_frames,h_pos_coords= pos_frames[pos_healthy], pos_coords[pos_healthy]
i_pos_feat,  i_pos_videos= pos_features[pos_impaired], pos_videos[pos_impaired]
i_pos_frames,i_pos_coords= pos_frames[pos_impaired], pos_coords[pos_impaired]

############################################
# 3. Magnify sequences by averaging over nearest neighbor
############################################
print('Producing magnification: %d queries'%(args.queries))
# Select queries randomly from impaired sequences
seq_queries = seq_impaired[np.random.permutation(seq_impaired.shape[0])][:args.queries]

query = seq_queries[0]
for query in tqdm(seq_queries,desc='Saving queries'):
    video, frame, coord = videos[query], frames[query], coords[query]
    feat = np.concatenate([pos_features[np.logical_and(pos_videos==video,pos_frames==f)] for f in frame])
    assert len(feat.shape)==2, 'Problems with query features size'
    original_seq = [load_image(v,f) for f, c in zip(frame,coord)]
    F = len(frame)
    # Look for NN in the healthy and impaired postures, 
    # then average over NN in the VAE space
    healthy_nn = np.dot(feat,h_pos_feat.T).argsort(1)[:,::-1][:,:args.nn]
    impaired_nn= np.dot(feat,i_pos_feat.T).argsort(1)[:,::-1][:,:args.nn]
    #j, f = 0, frame[0]
    _=plt.figure(figsize=(F*1,7))
    for j, f in enumerate(frame):
        healthy_seq = [load_image(h_pos_videos[nn],h_pos_frames[nn]) for nn in healthy_nn[j]]
        impaired_seq= [load_image(i_pos_videos[nn],i_pos_frames[nn]) for nn in impaired_nn[j]]
        magnified = magnifier.extrapolate_multiple(
                        healthy_seq, h_pos_feat[healthy_nn[j]],
                        impaired_seq,i_pos_feat[impaired_nn[j]],
                        [0.0,1.0,args.lambdas])
        healthy_res, impaired_res, magnified_res = magnified
        diff_image,flow_filter,X,Y = find_differences(healthy_res,impaired_res,
                                                    magnified_res,Th=0.25)
        healthy_res =  draw_box(healthy_res, l=2,color=[0,1.0,0])
        impaired_res=  draw_box(impaired_res,l=2,color=[0,0,1.0])
        magnified_res=  draw_box(magnified_res,l=2,color=[1.0,0,0])
        diff_image=draw_box(diff_image,l=2,color=[1.0,0,0])
        _=plt.subplot(5,F,j+1)
        _=plt.imshow(healthy_res); _=plt.axis('Off')
        if j==0: _=plt.title('Healthy average',fontsize=20,color='green')
        _=plt.subplot(5,F,F+j+1)
        _=plt.imshow(impaired_res); _=plt.axis('Off')
        if j==0: _=plt.title('Unhealthy',fontsize=20,color='red')
        _=plt.subplot(5,F,F*2+j+1)
        _=plt.imshow(magnified_res); _=plt.axis('Off')
        if j==0: _=plt.title('Magnification',fontsize=20,color='red')
        _=plt.subplot(5,F,F*3+j+1)
        _=plt.imshow(diff_image); _=plt.axis('Off')
        if j==0: _=plt.title('Difference',fontsize=20,color='red')
    
    #_=plt.title('Query %d'%(query),fontsize=20)
    plt.savefig(filtered_results_folder+'query_%d.jpg'%(query))
    #plt.show()
    plt.close('all')


# Save a video
#fig = plt.gcf()
#fig.canvas.draw()
#frame = np.array(fig.canvas.renderer._renderer)
#video.append(frame)

#writer = imageio.get_writer(outfold_videos+'/video%d.mp4'%(vi),fps=6)
#for img in video: 
    #writer.append_data(img)

#writer.close()
#imageio.mimsave(outfold_videos+'video%d.gif'%(vi),video,fps=5)
