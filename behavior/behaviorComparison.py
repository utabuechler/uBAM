#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 2.7
@author: Biagio Brattoli 
biagio.brattoli@iwr.uni-heidelberg.de
Last Update: 23.8.2018

Train the LDA to compare each time point to pre (time 0) and post (time 1) desease
Time 0 is Positive. Time 1 is negative.
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

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
except:
    from sklearn.lda import LDA

from sklearn.svm import LinearSVC

from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz
from scipy.spatial.distance import squareform, pdist
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

import config as cfg
# import config_pytorch_human as cfg


def linear_smooth(x):
    for i in range(1,len(x)-1):
        x[i] = (x[i+1]+x[i]+x[i-1])/3
    x[-1] = (2*x[-1]+x[-2])/3
    return x

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    
    output:
        the smoothed signal
        
    example:
    
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    
    if window_len<3:
        return x
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    s=np.r_[x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def intersection_area(dist1, dist2,x):
    points1, points2 = dist1(x), dist2(x)
    points = np.stack([points1,points2],1).min(1)
    overlap = cumtrapz(points,x)
    return overlap[-1]

#def intersection_area(dist1, dist2,x):
    #return dist1.integrate_kde(dist2)

parser = argparse.ArgumentParser()
parser.add_argument("-tp", "--train_ratio_pos",type=float,default=0.05,
                    help="Percentage of positives data to use for training the classifier")
parser.add_argument("-tn", "--train_ratio_neg",type=float,default=0.05,
                    help="Percentage of negatives data to use for training the classifier")
parser.add_argument("-i", "--iters",type=int,default=1,
                    help="Iterations of classifier training using different data")
args = parser.parse_args()

if args.train_ratio_pos>1.0:
    args.train_ratio_pos = args.train_ratio_pos/100.0
    args.train_ratio_neg = args.train_ratio_neg/100.0

# if __name__ == "__main__":
############################################
# 1. Load Features and samples information
############################################
results_fold = cfg.results_path+'/behavior/'
if not os.path.exists(results_fold): os.makedirs(results_fold)

detections = load_table(cfg.detection_file,asDict=False)
det_cohort= np.array(detections['cohort']) # Used for labels and plots
det_time  = np.array(detections['time'])   # Used for labels and plots
det_frames= np.array(detections['frames'])
det_videos= np.array(detections['videos'])
uni_videos= np.unique(detections['videos'].values)

print('Load features...')
features,frames,coords,videos = load_features('lstm',cfg.features_path,uni_videos.tolist())

# Get time and cohort for each feature sample. 
dic_time_cohort = {} # Dictionary used for efficiency.
time, cohort = [], []
for v in videos:
    if v not in dic_time_cohort: 
        I = np.where(det_videos==v)[0][0]
        dic_time_cohort[v] = [det_time[I],det_cohort[I]]
    
    time.append(dic_time_cohort[v][0])
    cohort.append(dic_time_cohort[v][1])

time, cohort = np.array(time), np.array(cohort)

uni_cohort, uni_time = np.unique(cohort), np.unique(time)
C, T = len(uni_cohort), len(uni_time)

# select frames features to use
features = features.reshape([features.shape[0],cfg.seq_len,-1])
features = normalize(features[:,-1,:].reshape([features.shape[0],-1]))

# Show the number of samples for cohort/time
count = np.zeros([C,T],'int')
for i, c in enumerate(uni_cohort):
    for j, t in enumerate(uni_time):
        I = np.logical_and(cohort==c,time==t)
        count[i,j] = I.sum()

print 'Number of samples per cohort and time:\n',count


###########################################
# 1. Train one classifier for all cohorts
###########################################
NtrainPos = (2*args.train_ratio_pos*count[:,0]).astype(int)
NtrainNeg = (2*args.train_ratio_neg*count[:,1]).astype(int)


positives = [np.where(np.logical_and(cohort==c,time==0))[0] for c in uni_cohort]
negatives = [np.where(np.logical_and(cohort==c,time==1))[0] for c in uni_cohort]
train_pos = [positives[j][:NtrainPos[j]] for j in range(C)]
train_neg = [negatives[j][:NtrainNeg[j]] for j in range(C)]
train_pos = [pp for p in train_pos for pp in p]
train_neg = [nn for n in train_neg for nn in n]

X_train = features[train_pos+train_neg]
y_train = np.array([1]*len(train_pos)+[-1]*len(train_neg))
# Collect the samples which have not been used for training, needed for the plot
val_pos = [positives[j][NtrainPos[j]:] for j in range(C)]
val_neg = [negatives[j][NtrainNeg[j]:] for j in range(C)]
val_pos = [pp for p in val_pos for pp in p]
val_neg = [nn for n in val_neg for nn in n]
X_val = features[val_pos+val_neg]
y_val = np.array([1]*len(val_pos)+[-1]*len(val_neg))

clf = LinearSVC(C=20.0).fit(X_train,y_train)
#clf = LDA().fit(X_train,y_train)
train_tp = clf.score(X_train[y_train==1], y_train[y_train==1])
train_tn = clf.score(X_train[y_train==-1],y_train[y_train==-1])
val_tp = clf.score(X_val[y_val==1], y_val[y_val==1])
val_tn = clf.score(X_val[y_val==-1],y_val[y_val==-1])
print 'SVM Classifier Accuracy: Train %.2f (%.2f,%.2f), Val %.2f (%.2f,%.2f)'%(
        (train_tp+train_tn)/2,train_tp,train_tn,(val_tp+val_tn)/2,val_tp,val_tn)

test  = np.where(np.logical_and(time!=0,time!=1))[0].tolist()
test += val_pos+val_neg
predictions = clf.predict(features[test])
pred_labels, pred_cohort = time[test], cohort[test]

###########################################
# 2. Barplot: show the ratio of positive sequences
###########################################
i=0; j=2
[np.logical_and(pred_cohort==c,pred_labels==t).sum() for c in uni_cohort for i in uni_time]

fitness = np.zeros([C,T])
for i in range(C):
    for j in range(T):
        sel = np.logical_and(pred_cohort==i,pred_labels==j)
        x = predictions[sel]
        x = (x+1)/2
        fitness[i,j] = x.mean()


colors = plt.cm.jet(np.linspace(0,1,C))
plt.style.use('ggplot')
_=plt.figure(figsize=(15,13))
cnt = 0
x = np.arange(T)
for i, c in enumerate(uni_cohort):
    y = fitness[i].copy()
    y[1:] = linear_smooth(y[1:])
    y = (y - y[1]) / (y[0] - y[1])
    # y[1:]=smooth(y[1:],3,'flat')[:-2]
    _=plt.subplot(int(np.sqrt(C)),int(C/np.sqrt(C)),i+1)
    _=plt.bar(x,y,width=0.75,color=colors[i],
                label='Cohort %d'%(c))
    _=plt.xticks(range(T),uni_time,fontsize=20)
    _=plt.xlim([-0.5,T-0.5])
    _=plt.ylim([0.0,1.1])
    _=plt.yticks(np.arange(0,1.1,0.1),fontsize=20)
    _=plt.xlabel('Group %d'%(c),fontsize=20)
    _=plt.ylabel('Average score',fontsize=20)

_=plt.savefig(results_fold+'fitness_score_bar_plot_single_svm_tp%.2f_tn%.2f_iters%d.png'%(
              args.train_ratio_pos,args.train_ratio_neg,args.iters),
              bbox_inches='tight',dpi=60)
plt.close('all')

#######################################################
# 3. Train classifier per Cohorts and score each sample
# Time 0 is Positive. Time 1 is negative.
# Train/Val split between time 0 and 1.
# Other times are used i testing for the plot
#######################################################
# Collect some samples from all datasets as shared baseline
NtrainPos = (args.train_ratio_pos*count[:,0]).astype(int)
NtrainNeg = (args.train_ratio_neg*count[:,1]).astype(int)

# Train the classifier per cohort using a same samples in common
positives = [np.where(np.logical_and(cohort==c,time==0))[0] for c in uni_cohort]
negatives = [np.where(np.logical_and(cohort==c,time==1))[0] for c in uni_cohort]

i, c = 1,1
ITERS = args.iters
overlap = np.zeros([ITERS,C,T,2])
scores_train= [[[] for _ in range(C)]]*ITERS
scores_test = [[[] for _ in range(C)]]*ITERS
predictions = [[[] for _ in range(C)]]*ITERS
accuracies = np.zeros([ITERS,C,2])
accuracies_svm=np.zeros([ITERS,C,2])
for IT in trange(ITERS,desc='Train classifier'):
    for i, c in enumerate(uni_cohort):
        # Select samples from all cohorts
        #shared_pos = [positives[j][np.random.permutation(len(positives[j]))][:NtrainPos[j]] for j in range(C)]
        #shared_neg = [negatives[j][np.random.permutation(len(negatives[j]))][:NtrainNeg[j]] for j in range(C)]
        shared_pos = [positives[j][:NtrainPos[j]] for j in range(C)]
        shared_neg = [negatives[j][:NtrainNeg[j]] for j in range(C)]
        shared_pos = [pp for p in shared_pos for pp in p]
        shared_neg = [nn for n in shared_neg for nn in n]
        # Select positives and negatives from this cohort
        pos, neg = positives[i], negatives[i]
        pos = np.array([p for p in pos if p not in shared_pos])
        neg = np.array([n for n in neg if n not in shared_neg])
        pos = pos[np.random.permutation(len(pos))][:cfg.min_train]
        neg = neg[np.random.permutation(len(neg))][:cfg.min_train]
        #pos, neg = pos[:cfg.min_train], neg[:cfg.min_train]
        # Combine both train samples
        train_pos = shared_pos+pos.tolist()
        train_neg = shared_neg+neg.tolist()
        # Make (X,y) pairs for the classifier
        X_train = features[train_pos+train_neg]
        y_train = np.array([1]*len(train_pos)+[-1]*len(train_neg))
        # Collect the samples which have not been used for training, needed for the plot
        val_pos, val_neg = positives[i], negatives[i]
        val_pos = [p for p in val_pos if p not in shared_pos]
        val_neg = [n for n in val_neg if n not in shared_neg]
        X_val = features[val_pos+val_neg]
        y_val = np.array([1]*len(val_pos)+[-1]*len(val_neg))
        test  = np.where(np.logical_and(cohort==c,np.logical_and(time!=0,time!=1)))[0].tolist()
        test += val_pos+val_neg
        # Train the classifier
        clf = LDA().fit(X_train,y_train)
        # Evaluate the classifier on Val
        train_tp = clf.score(X_train[y_train==1], y_train[y_train==1])
        train_tn = clf.score(X_train[y_train==-1],y_train[y_train==-1])
        val_tp = clf.score(X_val[y_val==1], y_val[y_val==1])
        val_tn = clf.score(X_val[y_val==-1],y_val[y_val==-1])
        accuracies[IT,i] = [(train_tp+train_tn)/2,(val_tp+val_tn)/2]
        # svm = LinearSVC(C=10.0).fit(X_train,y_train)
        # train_tp = svm.score(X_train[y_train==1], y_train[y_train==1])
        # train_tn = svm.score(X_train[y_train==-1],y_train[y_train==-1])
        # val_tp = svm.score(X_val[y_val==1], y_val[y_val==1])
        # val_tn = svm.score(X_val[y_val==-1],y_val[y_val==-1])
        # accuracies_svm[IT,i] = [(train_tp+train_tn)/2,(val_tp+val_tn)/2]
        #print 'Cohort %d) Classifier Accuracy: Train %.2f (%.2f,%.2f), Val %.2f (%.2f,%.2f)'%(c,
                #(train_tp+train_tn)/2,train_tp,train_tn,
                #(val_tp+val_tn)/2,val_tp,val_tn)
        # Produce the scores using the classifier
        predictions[IT][i] = [clf.decision_function(features[test]), time[test]]
        pos_scores  =clf.decision_function(features[val_pos])#features[positives[train_pos]])
        neg_scores  =clf.decision_function(features[val_neg])#features[negatives[train_neg]])
        other_scores=clf.decision_function(features[test])
        scores_train[IT][i] = [pos_scores,neg_scores]
        scores_test[IT][i]  = [other_scores,time[test]]
        # Compare behavior using overlapping density
        kernel_pos = gaussian_kde(pos_scores)
        kernel_neg = gaussian_kde(neg_scores)
        X, y = other_scores, time[test]
        x = np.linspace(X.min(), X.max(),2000)
        for j, t in enumerate(uni_time):
            I = y==t
            if I.sum()==0: continue
            kernel = gaussian_kde(X[I])
            overlap[IT,i,j,0] = intersection_area(kernel,kernel_pos,x)
            overlap[IT,i,j,1] = intersection_area(kernel,kernel_neg,x)

overlap_std = overlap.std(0)
overlap = overlap.mean(0)
accuracies = accuracies.mean(0)
# accuracies_svm = accuracies_svm.mean(0)

print 'Classifier Train accuracy: ', ['%.2f'%(accuracies[i][0]) for i in range(C)]
print 'Classifier Validation accuracy: ', ['%.2f'%(accuracies[i][1]) for i in range(C)]
# print 'SVM Classifier Train accuracy: ', ['%.2f'%(accuracies_svm[i][0]) for i in range(C)]
# print 'SVM Classifier Validation accuracy: ', ['%.2f'%(accuracies_svm[i][1]) for i in range(C)]

#plt.figure()
#for i in range(C):
    #x = scores_test[i][0]
    #y = scores_test[i][1]
    #plt.subplot(2,2,i+1)
    #plt.hist(scores_train[i]+[x[y==2]],50)

#plt.show()

#########################################################
# 4. Plot behavior comparison on a 2D plot: Pre X Post
# First normalize beginning and end to 1 and 0
# Then smooth over time
# Finally plot curves
#########################################################
similarity = np.zeros([C,T,2])
for i in range(C):
    x = overlap[i].T.copy()
    # normalize by baseline and 2days
    x[0,:] = (x[0,:]-x[0,1])/(x[0,0]-x[0,1])
    x[1,:] = (x[1,:]-x[1,0])/(x[1,1]-x[1,0])
    #all the lines between baseline and 2days are on top of each other, to
    #see all, move it a bit
    x[0,0] = x[0,0]+i*0.005
    x[0,1] = x[0,1]+i*0.005
    # smooth
    x[0,1:], x[1,1:] = linear_smooth(x[0,1:]), linear_smooth(x[1,1:])
    similarity[i] = x.T

#print '\nSimilarity:\n', similarity
colors = plt.cm.jet(np.linspace(0,1,C))

_=plt.figure(figsize=(12,10))
for i in range(C):
    for j in range(T-1):
        #if similarity[i,j+1,1]>1.0: continue
        if j==0: _=plt.plot(similarity[i,[j,j+1],1],similarity[i,[j,j+1],0],label=i,color=colors[i],linewidth=8)
        else: _=plt.plot(similarity[i,[j,j+1],1],similarity[i,[j,j+1],0],color=colors[i],linewidth=8)

#_=plt.ylim([0,1.0]); _=plt.xlim([0,1.0])
_=plt.ylabel('Similarity to Positive')
_=plt.xlabel('Similarity to Negative')

_=plt.legend()
plt.savefig(results_fold+'behavior_comparison_in_time_tp%.2f_tn%.2f_iters%d.png'%(
            args.train_ratio_pos,args.train_ratio_neg,args.iters),
            bbox_inches='tight')
plt.savefig(results_fold+'behavior_comparison_in_time_tp%.2f_tn%.2f_iters%d.eps'%(
            args.train_ratio_pos,args.train_ratio_neg,args.iters),
            bbox_inches='tight')
#plt.show()

