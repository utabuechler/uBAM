import os, sys
from tqdm import tqdm, trange
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
keypoints_source = '/export/scratch/bbrattol/behavior_analysis_demo/humans/magnification/magnification_pervideo/'
image_source = '/export/home/bbrattol/HumanGaitDataset/data/frames_crop/'

speed = '1kmh'
speed = '2kmh'
speed = '3kmh'

dest_path = 'res_'+speed
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

### Initializetion
videos = sorted([v for v in os.listdir(flow_source) if '.npz' in v])
videos = [v for v in videos if speed in v]
print(len(videos))
#videos = videos[:20] + videos[-10:]
videos = np.array(videos)

V = len(videos)
P = len(parts)
T = 200

H = np.array(['H' in v for v in videos])
I = np.array(['H' not in v for v in videos])

### Load magnitude and keypoints
#magnitude = np.zeros([V, T, 128, 128])
#keypoints = np.zeros([V, T, P, 3])
#for i, v in enumerate(tqdm(videos, 'plotting')):
#    flow = np.load(flow_source+v)['flow']
#    magnitude[i] = np.sqrt(np.power(flow,2).sum(-1))#.flatten()
#    #motion[i] = magnitude.mean()#.reshape([magnitude.shape[0], -1]).mean(-1)
#    keypoints[i] = load_keypoints_per_video('impaired', v[:-4], parts, keypoints_source)
#
#np.savez('magnitude_keypoints', magnitude=magnitude, keypoints=keypoints)

info = np.load('magnitude_keypoints.npz')
magnitude, keypoints = info['magnitude'], info['keypoints']

### Get Motion per keypoint
#dxy = 3
#motion = np.zeros([V, T, P])
#for v in trange(V, desc='Compute keypoint magnitude'):
#    for p in range(P):
#        txy = keypoints[v, :, p, :2]
#        txy = ( 128 * (txy.astype(float) / 400) ).astype(int)
#        tx, ty = txy[:, 0], txy[:, 1]
#        for t, (x, y) in enumerate(zip(tx, ty)):
#            m = magnitude[v, t, x-dxy:x+dxy, y-dxy:y+dxy].mean()
#            if np.isnan(m):
#                motion[v, t, p] = motion[v, t-1, p] 
#            else:
#                motion[v, t, p] = m
#
#print('Motion: ', motion.shape)
#motion = motion.mean(1)
#Hm = motion[H].mean(0)
#Im = motion[I].mean(0)
#print('Hm ',Hm,' Im ',Im)
#
#plt.figure(figsize=(4,4))
#plt.bar(range(P), Im, label='Impaired')
#plt.bar(range(P), Hm, label='Healthy')
#plt.xticks(range(P),parts, rotation=30)
#plt.legend()
#plt.savefig(dest_path+'/keypoints_motion.png', bbox_inches='tight',dpi=90)
#plt.savefig(dest_path+'/keypoints_motion.eps', bbox_inches='tight',dpi=90)
#plt.clf()
#plt.close()
#print('Motion per group saved')



## Plot Motion
#motion = magnitude.reshape([V,-1]).mean(-1)
#Hm = motion[H]
#Im = motion[I]
#print(Im.shape,Hm.shape)

#plt.figure(figsize=(4,4))
#plt.bar(range(2), [Im.mean(), Hm.mean()])
##plt.ylim([0.0, 3.0])
#plt.xticks([0,1],['Impaired', 'Healthy'], rotation=30)
#plt.savefig(dest_path+'/impaired_healthy_motion_allsubj.png', bbox_inches='tight',dpi=90)
#plt.savefig(dest_path+'/impaired_healthy_motion_allsubj.eps', bbox_inches='tight',dpi=90)
#plt.clf()
#plt.close()
#print('Motion per group saved')

### Plot example of keypoint and motion
v, f = 58, 20
ps= range(P) #[4, 6]
frames = range(1, 8, 3)

#plt.figure(figsize=(10,35))
fig, ax = plt.subplots(len(frames), 2)
for i, f in enumerate(frames):
    frame = imread(image_source+videos[v][:-4]+'/%06d.jpg'%(f))
    ax[i, 0].imshow(frame)
    for p in ps:
        ax[i, 0].plot(keypoints[v,f,p,0],keypoints[v,f,p,1], '.', markersize=15, label=parts[p])

    if i==0: ax[i, 0].legend()
    ax[i, 1].imshow(magnitude[v,f])

fig.set_size_inches(10, 15)
fig.tight_layout()
fig.savefig(dest_path+'/example_magnitude_keypoints.png')#, dpi=150, bbox_inches='tight')
fig.savefig(dest_path+'/example_magnitude_keypoints.eps')#, dpi=150, bbox_inches='tight')
plt.close()
print('Example saved')

      

#plt.figure(figsize=(10,3))
#for v in tqdm(videos, 'plotting'):
#    flow = np.load(flow_source+v)['flow']
#    #print(flow.shape) # shape: T, H, W, XY
#    magnitude = np.sqrt(np.power(flow,2).sum(-1))
#    motion = magnitude.reshape([magnitude.shape[0], -1]).mean(-1)
#    plt.plot(motion)
#    plt.ylim([0.0, 3.0])
#    plt.savefig(dest_path+'/'+v[:-4]+'.png')
#    plt.clf()
#
#plt.close()
#
