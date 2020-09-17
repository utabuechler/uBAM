import os, sys
from tqdm import tqdm, trange
from skimage.io import imread
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import gaussian

matplotlib.use('Agg')

def load_keypoints_per_video(fold, video, parts, path):
    keypoints = pd.read_csv(path+video+'/%s/keypoints_train0_400x400_ModelForGenerated/keypoints.csv'%(fold))
    fd = np.zeros([len(keypoints[parts[0]][1:].values),len(parts),3])
    for j, part in enumerate(parts):
        x = np.array(keypoints[part][1:].values).astype(float)
        y = np.array(keypoints[part+'.1'][1:].values).astype(float)
        prob = np.array(keypoints[part+'.2'][1:].values).astype(float)
        xy = np.stack([x,y,prob],1)
        if len(xy)==0: continue
        fd[:,j] = xy#.reshape([len(x),1,3])
    return fd

### Information
parts=['rhip', 'lhip', 'rknee', 'lknee', 'rFootBack',
       'lFootBack', 'rFootFront', 'lFootFront']

flow_source = '/export/scratch/bbrattol/behavior_analysis_demo/humans/magnification/magnification_pervideo_flow/'
flow_source = '/export/scratch/bbrattol/behavior_analysis_demo/humans/magnification/magnification_pervideo_flow_LearningBasedVideoMagnification/ampl_factor2.000000/'
keypoints_source = '/export/scratch/bbrattol/behavior_analysis_demo/humans/magnification/magnification_pervideo/'
image_source = '/export/home/bbrattol/HumanGaitDataset/data/frames_crop/'

speed = '1kmh'
speed = '2kmh'
speed = '3kmh'

dest_path = ''
dest_path = 'res_'+speed
dest_path = 'res_'+speed+'_our'
dest_path = 'res_videomotion'
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

### Initializetion
#videos = sorted([v for v in os.listdir(flow_source) if '.npz' in v])
##with open('videos.txt', 'r') as f:
##    videos = f.readlines()
#
#videos = [v for v in videos if speed in v]
#print(len(videos))
##videos = videos[:20] + videos[-10:]
#videos = np.array(videos)
#
#V = len(videos)
#P = len(parts)
#T = 200
#

#### Load magnitude and keypoints
#magnitude = np.zeros([V, T, 128, 128])
#keypoints = np.zeros([V, T, P, 3])
#for i, v in enumerate(tqdm(videos, 'plotting')):
#    flow = np.load(flow_source+v)['flow']
#    magnitude[i] = np.sqrt(np.power(flow,2).sum(-1))#.flatten()
#    #motion[i] = magnitude.mean()#.reshape([magnitude.shape[0], -1]).mean(-1)
#    keypoints[i] = load_keypoints_per_video('impaired', v[:-4], parts, keypoints_source)
#
#np.savez('magnitude_keypoints', magnitude=magnitude, keypoints=keypoints, videos=videos)

info = np.load('magnitude_keypoints_videomotion.npz')
magnitude, keypoints = info['magnitude'], info['keypoints']
videos = info['videos']
V, T, P, _ = keypoints.shape

H = np.array(['H' in v for v in videos])
I = np.array(['H' not in v for v in videos])

### Get Motion per keypoint
dxy = 5
motion = np.zeros([V, T, P])
for v in trange(V, desc='Compute keypoint magnitude'):
    for p in range(P):
        txy = keypoints[v, :, p, :2]
        txy = ( 128 * (txy.astype(float) / 400) ).astype(int)
        tx, ty = txy[:, 0], txy[:, 1]
        for t, (x, y) in enumerate(zip(tx, ty)):
            m = magnitude[v, t, x-dxy:x+dxy, y-dxy:y+dxy].mean()
            if np.isnan(m):
                motion[v, t, p] = motion[v, t-1, p] 
            else:
                motion[v, t, p] = m

### Plot signals for few subjects
R = 20

sel_I = np.where(I)[0][np.linspace(1, I.sum()-2, R//2).astype(int)]
sel_H = np.where(H)[0][np.linspace(0, H.sum()-2, R//2).astype(int)]
sel = sel_I.tolist() + sel_H.tolist()

X = motion[sel, :, 2:].mean(-1)
names = videos[sel]

def filter_signal(x, filter):
    if filter == 'Gaussian filter':
        f = gaussian(30, 2)
        f = f / f.sum()
    elif filter == 'Box filter':
        box = 10
        f = np.ones(box) / box
    elif filter == 'Peak filter':
        f = np.concatenate([-np.ones(5), np.ones(10), -np.ones(5)]) / 20
    else:
        f = np.ones(1)
    return np.convolve(x, f, mode='same')


filter_name = ['Original', 'Gaussian filter', 'Box filter', 'Peak filter']
C = len(filter_name)
fig, axs = plt.subplots(R, C)
for i in range(R):
    for j in range(C):
        x = np.copy(X[i])
        x = x - x.min()
        x = filter_signal(x, filter_name[j])
        ax = axs[i, j]
        ax.plot(x)
        if filter_name[j] != 'Peak filter':
            ax.set_ylim([0, 3])
        else:
            ax.set_ylim([-1, 1])
        if j == 0:
            v = videos[sel[i]][:-1]
            c = 'g' if 'H' in v else 'r'
            ax.set_title(v, color=c)
        else:
            ax.set_title(filter_name[j])

fig.set_size_inches(12, 25)
fig.tight_layout()
fig.savefig(dest_path+'/filtered_signal_subjects%d.png' % R)#, dpi=150, bbox_inches='tight')
fig.savefig(dest_path+'/filtered_signal_subjects%d.eps' % R)#, dpi=150, bbox_inches='tight')
plt.close()
print('Example saved')


