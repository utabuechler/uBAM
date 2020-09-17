import os, sys
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt

source_path = '/export/scratch/bbrattol/behavior_analysis_demo/humans/magnification/magnification_pervideo_flow/'

speed = '1kmh'
speed = '2kmh'
speed = '3kmh'

dest_path = 'res_'+speed
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

videos = sorted([v for v in os.listdir(source_path) if '.npz' in v])
videos = [v for v in videos if speed in v]
print(len(videos))
#videos = videos[:20] + videos[-10:]
videos = np.array(videos)
H = np.array(['H' in v for v in videos])
I = np.array(['H' not in v for v in videos])


motion = np.zeros(len(videos))
for i, v in enumerate(tqdm(videos, 'plotting')):
    flow = np.load(source_path+v)['flow']
    #print(flow.shape) # shape: T, H, W, XY
    magnitude = np.sqrt(np.power(flow,2).sum(-1)).flatten()
    #magnitude = magnitude.flatten().sort()[::-1]
    #magnitude.sort()
    #magnitude = magnitude[::-1][:int(128*128*0.05)]
    motion[i] = magnitude.mean()#.reshape([magnitude.shape[0], -1]).mean(-1)

#plt.figure(figsize=(10,3))
#plt.bar(range(motion.shape[0]), motion)
##plt.ylim([0.0, 3.0])
#plt.savefig(dest_path+'/motion.png')
#plt.clf()
#plt.close()
#

Hm = motion[H]
#Hm.sort()
#Hm = Hm[:int(Hm.shape[0]*0.80)]

Im = motion[I]
#Im.sort()
#Im = Im[:int(Im.shape[0]*0.80)]

print(Im.shape,Hm.shape)

plt.figure(figsize=(3,3))
plt.bar(range(2), [Im.mean(), Hm.mean()])
#plt.ylim([0.0, 3.0])
plt.xticks([0,1],['Impaired', 'Healthy'], rotation=30)
plt.savefig(dest_path+'/impaired_healthy_motion_allsubj.png')
plt.clf()
plt.close()



plt.figure(figsize=(10,3))
for v in tqdm(videos, 'plotting'):
    flow = np.load(source_path+v)['flow']
    #print(flow.shape) # shape: T, H, W, XY
    magnitude = np.sqrt(np.power(flow,2).sum(-1))
    motion = magnitude.reshape([magnitude.shape[0], -1]).mean(-1)
    plt.plot(motion)
    plt.ylim([0.0, 3.0])
    plt.savefig(dest_path+'/'+v[:-4]+'.png')
    plt.clf()

plt.close()

