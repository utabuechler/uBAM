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

os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)

############################################
# 0. Prepare magnifier object
############################################
magnifier = Generator(z_dim=cfg.encode_dim,path_model=cfg.vae_weights_path)

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


dt = datetime.now()
dt = '{}-{}-{}-{}-{}/'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
filtered_results_folder = results_folder+'/sequences_healthy%d_impaired%d/'%(
                          seq_healthy.shape[0],seq_impaired.shape[0])+dt
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
    original_seq = [load_image(cfg.crops_path,video,f) for f, c in zip(frame,coord)]
    F = len(frame)
    # Look for NN in the healthy and impaired postures, 
    # then average over NN in the VAE space
    healthy_nn = np.dot(feat,h_pos_feat.T).argsort(1)[:,::-1][:,:args.nn]
    impaired_nn= np.dot(feat,i_pos_feat.T).argsort(1)[:,::-1][:,:args.nn]
    #j, f = 0, frame[0]
    R=5
    _=plt.figure(figsize=(F*1,2+R))
    for j, f in enumerate(frame):
        healthy_seq = [load_image(cfg.crops_path,h_pos_videos[nn],h_pos_frames[nn]) for nn in healthy_nn[j]]
        impaired_seq= [load_image(cfg.crops_path,i_pos_videos[nn],i_pos_frames[nn]) for nn in impaired_nn[j]]
        magnified = magnifier.extrapolate_multiple(
                        healthy_seq, h_pos_feat[healthy_nn[j]],
                        impaired_seq,i_pos_feat[impaired_nn[j]],
                        [0.0,1.0,args.lambdas])
        healthy_res, impaired_res, magnified_res = magnified
        diff_image,flow_filtered,X,Y = find_differences(healthy_res,impaired_res,
                                        magnified_res,Th=0.20,scale=20)
        healthy_res  = draw_border(healthy_res, l=2,color=[0,1.0,0])
        impaired_res = draw_border(impaired_res,l=2,color=[0,0,1.0])
        magnified_res= draw_border(magnified_res,l=2,color=[1.0,0,0])
        diff_image=draw_border(diff_image,l=2,color=[1.0,0,0])
        #_=plt.subplot(R,F,j+1)
        #_=plt.imshow(original_seq[j]); _=plt.axis('Off')
        #if j==0: _=plt.title('Original',fontsize=20,color='black')
        C = 0
        _=plt.subplot(R,F,F*C+j+1); C+=1
        _=plt.imshow(healthy_res); _=plt.axis('Off')
        if j==0: _=plt.title('Healthy',fontsize=20,color='green')
        _=plt.subplot(R,F,F*C+j+1); C+=1
        _=plt.imshow(impaired_res); _=plt.axis('Off')
        if j==0: _=plt.title('Unhealthy',fontsize=20,color='red')
        _=plt.subplot(R,F,F*C+j+1); C+=1
        _=plt.imshow(magnified_res); _=plt.axis('Off')
        if j==0: _=plt.title('Magnification',fontsize=20,color='red')
        _=plt.subplot(R,F,F*C+j+1); C+=1
        _=plt.imshow(diff_image); _=plt.axis('Off')
        if j==0: _=plt.title('Difference',fontsize=20,color='red')
        _=plt.subplot(R,F,F*C+j+1); C+=1
        _=plt.imshow(impaired_res); _=plt.axis('Off')
        _=plt.quiver(X, Y, flow_filtered[X,Y,0], flow_filtered[X,Y,1], 
                     width=0.1, headwidth=3, color='red', 
                     scale_units='width', scale=5,minlength=0.1)
        if j==0: _=plt.title('Improvement',fontsize=20,color='red')
    
    #_=plt.title('Query %d'%(query),fontsize=20)
    plt.savefig(filtered_results_folder+'query_%d.png'%(query))
    #plt.show()
    plt.close('all')


#DEBUGGING CODE
#difference = compute_difference(impaired_res,magnified_res,kernel=7)
#diff_image = difference2color(difference,impaired_res,0.10)
#diff_rescale = np.linspace(0,difference.shape[0]-1,int(0.5*difference.shape[0])).astype(int)
#difference_small = difference[diff_rescale,diff_rescale]#resize(difference,scale1)
#mask = difference_small>=0.2

##diff_image,flow_filtered,X,Y = find_differences(
    ##healthy_res,impaired_res,magnified_res,Th=0.05)

#healthy, impaired, enhanced = healthy_res.copy(), impaired_res.copy(), magnified_res.copy()
#scale1, scale2, Th = 0.5, 10, 0.10
#optical_flow = cv2.createOptFlow_DualTVL1()
## RGB DIFFERENCE
#difference = compute_difference(impaired,enhanced,kernel=7)
#diff_image = difference2color(difference,impaired,Th)
## FLOW
#impaired_small   = (resize(impaired, scale1)*255).astype('uint8') # resize for reducing flow resolution
#enhanced_small   = (resize(enhanced,scale1)*255).astype('uint8')
#diff_rescale = np.linspace(0,difference.shape[0]-1,int(scale1*difference.shape[0])).astype(int)
#difference_small = difference[diff_rescale,:][:,diff_rescale]#resize(difference,scale1)
#impaired_gray=cv2.cvtColor(impaired_small,cv2.COLOR_RGB2GRAY)
#enhanced_gray=cv2.cvtColor(enhanced_small,cv2.COLOR_RGB2GRAY)
#flow = optical_flow.calc(enhanced_gray, impaired_gray, None) # compute flow
## FLOW VECTOR TO IMAGE
#mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#flow = flow/np.repeat(mag[:,:,np.newaxis],2,axis=2)
#mask = np.logical_and(difference_small>=Th,mag>5)
#flow_filter = np.stack([flow[:,:,0]*mask,flow[:,:,1]*mask],axis=2)
#flow_filter_sparse = np.zeros([int(flow_filter.shape[0]/scale1),
                               #int(flow_filter.shape[1]/scale1),
                               #flow_filter.shape[2]],flow_filter.dtype)
#for x in range(0,flow_filter.shape[0]-scale2/2,scale2):
    #for y in range(0,flow_filter.shape[1]-scale2/2,scale2):
        #X, Y = int((x+scale2/2)/scale1), int((y+scale2/2)/scale1)
        #m = flow_filter[x:x+scale2,y:y+scale2].reshape([-1,2])
        #mx, my = m[:,0], m[:,1]
        #m = np.stack([mx[mx!=0].mean(0),my[my!=0].mean(0)])
        #flow_filter_sparse[X,Y] = m if not np.isnan(m).any() else 0

#mag, ang = cv2.cartToPolar(flow_filter_sparse[...,0], flow_filter_sparse[...,1])
#flow_filter_sparse = flow_filter_sparse/(np.repeat(mag[:,:,np.newaxis],2,axis=2)+0.01)
##flow_filter_sparse = np.stack([resize(flow_filter[:,:,0],1/scale1),resize(flow_filter[:,:,1],1/scale1)],2)
#X, Y = np.where(np.logical_or(flow_filter_sparse[:,:,0]!=0,flow_filter_sparse[:,:,1]!=0))

#_=plt.subplot(1,2,1)
#_=plt.imshow(diff_image); _=plt.axis('Off')
#_=plt.subplot(1,2,2)
#_=plt.imshow(original_seq[j]); _=plt.axis('Off')
#_=plt.quiver(X, Y, flow_filter_sparse[X,Y,0], flow_filter_sparse[X,Y,1], 
                #width=0.1, headwidth=3, color='red', 
                #scale_units='width', scale=10,minlength=0.1)
#plt.show()



#_=plt.imshow((mask).astype('float32'));plt.show()

#DEBUGGING CODE
#import cv2
#scale1, scale2 = 0.50, 10**2
#optical_flow = cv2.createOptFlow_DualTVL1()

#healthy, impaired, enhanced = healthy_res.copy(), impaired_res.copy(), magnified_res.copy()

## RGB DIFFERENCE
#kernel=10
##difference = compute_difference(healthy,enhanced,kernel=5)
#healthy_gray  = rgb2gray(healthy)
#impaired_gray = rgb2gray(impaired)#*255).astype('uint8')
#enhanced_gray = rgb2gray(enhanced)#*255).astype('uint8')
#healthy_blur  = cv2.blur(healthy_gray,(kernel,kernel))
#impaired_blur = cv2.blur(impaired_gray,(kernel,kernel))
#enhanced_blur = cv2.blur(enhanced_gray,(kernel,kernel))
#difference_he = np.abs(healthy_blur  - enhanced_blur)
#difference_hi = np.abs(healthy_blur  - impaired_blur)
#difference    = np.abs(impaired_blur - enhanced_blur)
##difference[0:difference.shape[0]/2] = 0
#diff_image_he= difference2color(difference_he,impaired,0.25)
#diff_image_hi= difference2color(difference_hi,impaired,0.25)
#diff_image = difference2color(difference,impaired,0.25)

## FLOW
##healthy_small    = (resize(healthy, scale1)*255).astype('uint8') # resize for reducing flow resolution
##impaired_small   = (resize(impaired, scale1)*255).astype('uint8')
##enhanced_small   = (resize(enhanced,scale1)*255).astype('uint8')
##healthy_small_gray =cv2.cvtColor(healthy_small,cv2.COLOR_RGB2GRAY)
##impaired_small_gray=cv2.cvtColor(impaired_small,cv2.COLOR_RGB2GRAY)
##enhanced_small_gray=cv2.cvtColor(enhanced_small,cv2.COLOR_RGB2GRAY)
#healthy_small_gray =(healthy_blur*255).astype('uint8')
#impaired_small_gray =(impaired_blur*255).astype('uint8')
#enhanced_small_gray =(enhanced_blur*255).astype('uint8')
#flow_he = optical_flow.calc(enhanced_small_gray, healthy_small_gray, None) # compute flow
#flow_hi = optical_flow.calc(impaired_small_gray, healthy_small_gray, None)
#flow    = optical_flow.calc(enhanced_small_gray,impaired_small_gray, None)
#flow_he_image = flow2image(flow_he)
#flow_hi_image = flow2image(flow_hi)
#flow_image    = flow2image(flow)

#mag_he, _ = cv2.cartToPolar(flow_he[...,0], flow_he[...,1])
#mag_hi, _ = cv2.cartToPolar(flow_hi[...,0], flow_hi[...,1])
#mag,    _ = cv2.cartToPolar(flow[...,0],    flow[...,1])

#R = 6
#_=plt.subplot(R,3,1)
#_=plt.imshow(healthy); _=plt.axis('Off')
#_=plt.subplot(R,3,2)
#_=plt.imshow(healthy_gray); _=plt.axis('Off')
#_=plt.subplot(R,3,3)
#_=plt.imshow(healthy_blur); _=plt.axis('Off')

#_=plt.subplot(R,3,4)
#_=plt.imshow(impaired); _=plt.axis('Off')
#_=plt.subplot(R,3,5)
#_=plt.imshow(impaired_gray); _=plt.axis('Off')
#_=plt.subplot(R,3,6)
#_=plt.imshow(impaired_blur); _=plt.axis('Off')

#_=plt.subplot(R,3,7)
#_=plt.imshow(enhanced); _=plt.axis('Off')
#_=plt.subplot(R,3,8)
#_=plt.imshow(enhanced_gray); _=plt.axis('Off')
#_=plt.subplot(R,3,9)
#_=plt.imshow(enhanced_blur); _=plt.axis('Off')

#_=plt.subplot(R,3,10)
#_=plt.imshow(difference_he); _=plt.axis('Off')
#_=plt.subplot(R,3,11)
#_=plt.imshow(difference_hi); _=plt.axis('Off')
#_=plt.subplot(R,3,12)
#_=plt.imshow(difference); _=plt.axis('Off')

#_=plt.subplot(R,3,13)
#_=plt.imshow(flow_he_image); _=plt.axis('Off')
#_=plt.subplot(R,3,14)
#_=plt.imshow(flow_hi_image); _=plt.axis('Off')
#_=plt.subplot(R,3,15)
#_=plt.imshow(flow_image); _=plt.axis('Off')

#_=plt.subplot(R,3,16)
#_=plt.imshow(mag_he); _=plt.axis('Off')
#_=plt.subplot(R,3,17)
#_=plt.imshow(mag_hi); _=plt.axis('Off')
#_=plt.subplot(R,3,18)
#_=plt.imshow(mag); _=plt.axis('Off')

#plt.show()


# FLOW VECTOR TO IMAGE
#mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#x = np.linspace(0,impaired.shape[0]-1,flow.shape[0]).astype(int)
#X, Y = np.meshgrid(x, x)
#difference_small = resize(difference,scale1)
#mask = np.logical_and(difference_small>=Th,mag>0.5)
#mask = cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8),iterations = 1)==1
#X, Y = X[mask], Y[mask]
#flow = flow/np.repeat(mag[:,:,np.newaxis],2,axis=2)
#flow_filter = np.stack([flow[:,:,0][mask],flow[:,:,1][mask]],axis=1)
#flow_filter = flow_filter[::scale2]
#X, Y = X[::scale2], Y[::scale2]

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
