import os, sys
from tqdm import tqdm, trange
from skimage.io import imread
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import datetime
import imageio

# matplotlib.use('Agg')

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
parts=['rhip', 'lhip', 'rknee', 'lknee', 'rFootBack', 'lFootBack', 'rFootFront', 'lFootFront']

flow_source = '/export/scratch/bbrattol/behavior_analysis_demo/humans/magnification/frames_difference/trial1/'
keypoints_source = '/export/scratch/bbrattol/behavior_analysis_demo/humans/magnification/magnification_pervideo/'
image_source = '/export/home/bbrattol/HumanGaitDataset/data/frames_crop/'

speed = '1kmh'
speed = '2kmh'
speed = '3kmh'

today = datetime.date.today()
dest_path = '../results/'+today.strftime('%d-%b-%Y')+'/res_' + speed + '/'#''_remove_FP_LP/'
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

### Initializetion
# videos = sorted([v for v in os.listdir(flow_source) if '.npz' in v])
# videos = [v for v in videos if speed in v]
# print(len(videos))
#videos = videos[:20] + videos[-10:]
# videos = np.array(videos)
#
# V = len(videos)
# P = len(parts)
# T = 200

### Load magnitude and keypoints
# magnitude = np.zeros([V, T, 128, 128], 'uint8')
# keypoints = np.zeros([V, T, P, 3])
# for i, v in enumerate(tqdm(videos, 'plotting')):
#     flow = np.load(flow_source+v)['difference']
#     diff = np.sqrt(np.power(flow,2))
#     magnitude[i] = (255*diff).astype('uint8')
#     keypoints[i] = load_keypoints_per_video('impaired', v[:-4], parts, keypoints_source)
#
# np.savez('difference_magnitude_keypoints_'+speed, magnitude=magnitude, keypoints=keypoints, videos=videos)

info = np.load('difference_magnitude_keypoints_'+speed+'.npz')
magnitude, keypoints, videos = info['magnitude'], info['keypoints'], info['videos']
# magnitude = magnitude[:, :, :64]
sel = ['Famp' not in v and 'LP' not in v for v in videos]

videos = videos[sel]
magnitude = magnitude[sel]
keypoints = keypoints[sel]

V, P, T = len(videos), len(parts), 200

H = np.where(['H' in v for v in videos])[0].tolist()
I = np.where(['H' not in v for v in videos])[0].tolist()

motion = magnitude.reshape([V, T, -1]).mean(-1)
Hm = motion[H]
Im = motion[I]
print(Im.shape, Hm.shape)

# Plot Motion per subject in separate gif files
# distance_path = dest_path + 'differences_2D/'
# if not os.path.exists(distance_path):
#     os.makedirs(distance_path)
#
# for m, v in tqdm(zip(magnitude, videos)):
#     imageio.mimsave(distance_path+v[:-4]+'.gif', m)


magnitude = magnitude.astype('float32') / magnitude.max()

# Plot Motion for Impaired and Healthy
# plt.figure(figsize=(4, 4))
# plt.bar(range(2), [Im.mean(), Hm.mean()], yerr=[Im.std(), Hm.std()])
# #plt.ylim([0.0, 3.0])
# plt.xticks([0, 1], ['Impaired', 'Healthy'], rotation=30)
# plt.savefig(dest_path+'/impaired_healthy_motion_allsubj_withErr.png', bbox_inches='tight',dpi=90)
# plt.savefig(dest_path+'/impaired_healthy_motion_allsubj_withErr.eps', bbox_inches='tight',dpi=90)
# plt.clf()
# plt.close()
# print('Motion per group saved')

# Plot Motion for 10 and 10 subjects in one file
# x = [motion[I[:10]], motion[H]]
# vs= [videos[I[:10]], videos[H]]
# f, axes = plt.subplots(10, 2)
# for i in range(2):
#     for j in range(10):
#         a, m, v = axes[j, i], x[i][j], vs[i][j]
#         a.plot(m)
#         a.set_ylim([0, motion.max()])
#         a.set_title(v, color=['r', 'g'][i], fontsize=15)
#         # plt.ylim([0.0, 3.0])
#
# f.set_size_inches((15, 40))
# plt.savefig(dest_path+'difference_per_subject.png', bbox_inches='tight')
# plt.close()


# # Plot Motion per subject in separate files
# motion_dest = dest_path + 'difference_per_subject/'
# if not os.path.exists(motion_dest):
#     os.makedirs(motion_dest)
#
# plt.figure(figsize=(10, 3))
# for m, v in zip(motion, videos):
#     plt.plot(m)
#     # plt.ylim([0.0, 3.0])
#     plt.savefig(motion_dest+v[:-4]+'.png')
#     plt.clf()
#
# plt.close()


#### Get Motion per keypoint
dxy = 2
motion = np.zeros([V, T, P])
for v in trange(V, desc='Compute keypoint magnitude'):
   for p in range(P):
       txy = keypoints[v, :, p, :2]
       txy = (128 * (txy.astype(float) / 400)).astype(int)
       tx, ty = txy[:, 0], txy[:, 1]
       for t, (x, y) in enumerate(zip(tx, ty)):
           m = magnitude[v, t, max(x-dxy, 0):x+dxy, max(y-dxy, 0):y+dxy].mean()
           motion[v, t, p] = m

# Plot difference magnitude per keypoint for healthy and impaired
x = motion[I[:10]+H]
vs= videos[I[:10]+H]
f, axes = plt.subplots(20, P)
for p in range(P):
    for j in range(20):
        a, m, v = axes[j, p], x[j, :, p], vs[j]
        # assert m.shape[0] == T, 'Vector size wrong'
        m = m - m.mean() / m.std()
        a.plot(m)
        # a.set_ylim([0, motion.max()])
        a.set_ylim([-2, 0])
        if p == 0:
            a.set_title(v, color='g' if 'H' in v else 'r', fontsize=15)
        else:
            a.set_title(parts[p], color='black', fontsize=15)
        # plt.ylim([0.0, 3.0])

f.set_size_inches((30, 40))
plt.savefig(dest_path+'difference_per_keypoint4x4_per_subject_normalized.png', bbox_inches='tight')
plt.close()


# x = [motion[I[:10]].mean(1), motion[H].mean(1)]
# f, axes = plt.subplots(P, 2)
# plt.figure()
# for j in range(2):
#     plt.bar(np.arange(P) + 0.5*j, x[j].mean(0), yerr=x[j].std(0), width=0.4, label=['Impaired', 'Healthy'][j])
#
# plt.xticks(np.arange(P) + 0.25, parts, rotation=30)
# plt.legend()
# f.set_size_inches((15, 5))
# plt.savefig(dest_path+'keypoints_difference_healthy_impaired_SbjSTD.png', bbox_inches='tight')
# plt.close()



##### OLD ######
#### Plot peaks per keypoint
#std, dist = 3, 5
#print(motion.mean(), motion.max())
#minH = motion.reshape([-1, P]).mean(0) + std*motion.reshape([-1, P]).std(0)
#print(minH)
#
#motion_H = motion[H].reshape([-1,40,P])
#motion_I = motion[I].reshape([-1,40,P])
#print(motion_H.shape[0], motion_I.shape[0])
#
#def get_peaks(motion, minH):
#    motion = motion[np.random.permutation(motion.shape[0])][:30]
#    V, T = motion.shape
#    p_qnt, pks = [], []
#    for v in range(V):
#        x = motion[v,:]
#        px, _ = find_peaks(x, height=minH[p], distance=dist)#, width=(5,50))
#        pks += x[px].tolist()
#        p_qnt.append(len(px))
#    return pks, p_qnt
#
#IT = 100
#bins = [20,5]
#edges = [np.linspace(0, motion.max(), bins[0]+1), np.linspace(1,6,bins[1]+1)-0.5]
#histograms = [np.zeros([P,2,bins[0],IT]), np.zeros([P,2,bins[1],IT])]
#for p in trange(P):
#    for t in range(IT):
#        peaks_H, peaks_qnt_H = get_peaks(np.copy(motion_H[:,:,p]), minH)
#        peaks_I, peaks_qnt_I = get_peaks(np.copy(motion_I[:,:,p]), minH)
#        peak_type = [[peaks_I, peaks_H], [peaks_qnt_I, peaks_qnt_H]]
#        for i, (pI, pH) in enumerate(peak_type):
#            for j, pk in enumerate([pI, pH]):
#                y, b = np.histogram(pk, bins=edges[i], density=False)
#                histograms[i][p,j,:,t] = y
#
#
##f, axs = plt.subplots(P,2)
##for p in range(P):
##    for i, htype in enumerate(['intensity', 'quantity']):
##        for j, name in enumerate(['Impaired', 'Healthy']):
##            y, b = histograms[i][p,j], edges[i]
##            w = (b[1]-b[0])
##            x = b[:-1] + w/2
##            #x, y = x[y>0], y[y>0]
##            axs[p,i].errorbar(x,y.mean(-1),y.std(-1),label=name)
##            #axs[p,i].hist(pI, bins=bins[i], density=i==0, label='Impaired')
##            axs[p,i].axhline(0.0, color='red')
##        axs[p,i].set_ylim([0,15])
##        axs[p,i].set_title('('+parts[p]+') Peaks '+htype+' histogram')
##        axs[p,i].legend()
##
##f.set_size_inches(15,30)
##plt.savefig(dest_path+'/histogram_peaks_keypoints_motion_std.png', bbox_inches='tight',dpi=90)
##plt.savefig(dest_path+'/histogram_peaks_keypoints_motion_std.eps', bbox_inches='tight',dpi=90)
##plt.clf()
##plt.close()
##print('Motion per group saved')
##
#
#
#f, axs = plt.subplots(P)
#for p in range(P):
#    for j, name in enumerate(['Impaired', 'Healthy']):
#        y, b = histograms[1][p,j], edges[1]
#        w = (b[1]-b[0])
#        x = b[:-1] + w/2
#        axs[p].errorbar(x,y.mean(-1),y.std(-1),label=name)
#        axs[p].axhline(0.0, color='red')
#    axs[p].set_ylim([0,10])
#    axs[p].set_title('('+parts[p]+') Peaks quantity histogram')
#    axs[p].legend()
#
#f.set_size_inches(10,30)
#plt.savefig(dest_path+'/peaks_count_histogram_keypoints_dist5_std3.png', bbox_inches='tight',dpi=90)
#plt.savefig(dest_path+'/peaks_count_histogram_keypoints_dist5_std3.eps', bbox_inches='tight',dpi=90)
#plt.clf()
#plt.close()
#print('Motion per group saved')
#
#
#f, axs = plt.subplots(P)
#for p in range(P):
#    for j, name in enumerate(['Impaired', 'Healthy']):
#        y, b = histograms[1][p,j], edges[1]
#        w = (b[1]-b[0])
#        x = b[:-1] + w/2
#        x, y = x[1:], y[1:]
#        axs[p].errorbar(x,y.mean(-1),y.std(-1),label=name)
#        axs[p].axhline(0.0, color='red')
#    axs[p].set_ylim([0,10])
#    axs[p].set_title('('+parts[p]+') Peaks quantity histogram')
#    axs[p].legend()
#
#f.set_size_inches(10,30)
#plt.savefig(dest_path+'/peaks_count_histogram_keypoints_s2_dist5_std3.png', bbox_inches='tight',dpi=90)
#plt.savefig(dest_path+'/peaks_count_histogram_keypoints_s2_dist5_std3.eps', bbox_inches='tight',dpi=90)
#plt.clf()
#plt.close()
#print('Motion per group saved')
#
#
#
## Average histogram over keypoints
#y = histograms[1].mean(0)
#plt.figure(figsize=(5,4))
#plt.bar(np.arange(1,y.shape[-2]+1), y[0].mean(-1),0.4, yerr=y[0].std(-1), label='Impaired')
#plt.bar(np.arange(1,y.shape[-2]+1)+0.4, y[1].mean(-1),0.4, yerr=y[1].std(-1), label='Healthy')
#plt.title('Peaks count histogram - average over keypoints')
#plt.legend()
#plt.savefig(dest_path+'/peaks_histogram_dist5_std3.png', bbox_inches='tight',dpi=90)
#plt.savefig(dest_path+'/peaks_histogram_dist5_std3.eps', bbox_inches='tight',dpi=90)
#plt.clf()
#plt.close()
#print('Plot count per keypoint saved')
#
#
## Plot peaks count averaged over keypoints
##def count_peaks(motion):
##    V, T, P = motion.shape
##    count = np.zeros([P,V]) 
##    for p in range(P):
##        for v in range(V):
##            x = motion[v,:,p]
##            px, _ = find_peaks(x, height=minH[p], distance=8)#, width=(5,50))
##            count[p,v] = len(px)
##    return count
##
##count_H = count_peaks(motion_H)
##count_I = count_peaks(motion_I)
##
### Plot total number of peaks
##plt.figure(figsize=(5,4))
##plt.bar(np.arange(P), count_I.mean(1),0.4, yerr=count_I.std(1), label='Impaired')
##plt.bar(np.arange(P)+0.4, count_H.mean(1),0.4, yerr=count_H.std(1), label='Healthy')
##plt.savefig(dest_path+'/peaks_keypoints_count.png', bbox_inches='tight',dpi=90)
##plt.savefig(dest_path+'/peaks_keypoints_count.eps', bbox_inches='tight',dpi=90)
##plt.clf()
##plt.close()
##print('Plot count per keypoint saved')
##
##plt.figure(figsize=(4,4))
##plt.bar([0], count_I.mean(), yerr=count_I.std(), label='Impaired')
##plt.bar([1], count_H.mean(), yerr=count_H.std(), label='Healthy')
##plt.savefig(dest_path+'/peaks_count.png', bbox_inches='tight',dpi=90)
##plt.savefig(dest_path+'/peaks_count.eps', bbox_inches='tight',dpi=90)
##plt.clf()
##plt.close()
##print('Plot count saved')
##
#
#### BACKUP Plot peaks per keypoint
##print(motion.mean(), motion.max())
##minH = motion.reshape([-1, P]).mean(0) + 2*motion.reshape([-1, P]).std(0)
##print(minH)
##
##motion_H = motion[H].reshape([-1,50,P])
##motion_I = motion[I].reshape([-1,50,P])
##print(motion_H.shape[0], motion_I.shape[0])
##
##def get_peaks(motion, minH):
##    motion = motion[np.random.permutation(motion.shape[0])][:30]
##    V, T, P = motion.shape
##    peaks_qnt_parts = []
##    peaks_intensity_parts = []
##    for p in range(P):
##        p_qnt, pks = [], []
##        for v in range(V):
##            #for t in range(0,T,50):
##            #    x = motion[v,t:t+50,p]
##            x = motion[v,:,p]
##            px, _ = find_peaks(x, height=minH[p], distance=5)#, width=(5,50))
##            pks += x[px].tolist()
##            p_qnt.append(len(px))
##        peaks_qnt_parts.append(p_qnt) 
##        peaks_intensity_parts.append(pks) 
##    return peaks_intensity_parts, peaks_qnt_parts
##
##peaks_H, peaks_qnt_H = get_peaks(np.copy(motion_H), minH)
##peaks_I, peaks_qnt_I = get_peaks(np.copy(motion_I), minH)
##
##bins = [20, 7]
##f, axs = plt.subplots(P,2)
##for p in range(P):
##    peaks = [[peaks_I[p], peaks_H[p]], [peaks_qnt_I[p], peaks_qnt_H[p]]]
##    for i, (pI, pH) in enumerate(peaks):
##        for pk, name in zip([pI, pH], ['Impaired', 'Healthy']):
##            if len(pI)==0: continue
##            b = np.linspace(0,max(pI),bins[i]+1)
##            y, b = np.histogram(pk, bins=b, density=True)
##            w = (b[1]-b[0])
##            x = b[:-1] + w/2
##            axs[p,i].errorbar(x,y,label=name)
##            #axs[p,i].hist(pI, bins=bins[i], density=i==0, label='Impaired')
##        axs[p,i].set_title('('+parts[p]+') Peaks intensity histogram')
##        axs[p,i].legend()
##
##f.set_size_inches(15,30)
##plt.savefig(dest_path+'/peaks_keypoints_motion_numpy.png', bbox_inches='tight',dpi=90)
##plt.savefig(dest_path+'/peaks_keypoints_motion_numpy.eps', bbox_inches='tight',dpi=90)
##plt.clf()
##plt.close()
##print('Motion per group saved')
##
#
#
#### Plot example of keypoint and motion
##v, f = 58, 20
##ps= range(P) #[4, 6]
##frames = range(1, 8, 3)
##
###plt.figure(figsize=(10,35))
##fig, ax = plt.subplots(len(frames), 2)
##for i, f in enumerate(frames):
##    frame = imread(image_source+videos[v][:-4]+'/%06d.jpg'%(f))
##    ax[i, 0].imshow(frame)
##    for p in ps:
##        ax[i, 0].plot(keypoints[v,f,p,0],keypoints[v,f,p,1], '.', markersize=15, label=parts[p])
##
##    if i==0: ax[i, 0].legend()
##    ax[i, 1].imshow(magnitude[v,f])
##
##fig.set_size_inches(10, 15)
##fig.tight_layout()
##fig.savefig(dest_path+'/example_magnitude_keypoints.png')#, dpi=150, bbox_inches='tight')
##fig.savefig(dest_path+'/example_magnitude_keypoints.eps')#, dpi=150, bbox_inches='tight')
##plt.close()
##print('Example saved')
##
#      
#
