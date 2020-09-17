# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:50:39 2017
@author: Biagio Brattoli
"""
import os, sys, numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from skimage.morphology import binary_dilation
from skimage.transform import resize as skresize
from skimage.color import rgb2gray
from skimage.io import imread
import skimage.measure
import cv2

################ FUNCTIONS ################
def resize(image,scale):
    image = skresize(image, (int(scale*float(image.shape[0])), 
                            int(scale*float(image.shape[1]))))
    return image

def flow2image(flow,saturation=0.20):
    hsv = np.zeros([flow.shape[0],flow.shape[1],3],'uint8')
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    #plt.hist(mag.flatten(),50); plt.show()
    # mag[mag>saturation*mag.max()] = saturation*mag.max()
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    flow_image = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return flow_image


def compute_difference(impaired,enhanced,kernel=3):
    impaired_gray = rgb2gray(impaired)#*255).astype('uint8')
    enhanced_gray = rgb2gray(enhanced)#*255).astype('uint8')
    impaired_blur = cv2.blur(impaired_gray,(kernel,kernel))
    enhanced_blur = cv2.blur(enhanced_gray,(kernel,kernel))
    difference    = np.abs(impaired_blur - enhanced_blur)
    return difference

def difference2color(difference,impaired,threshold,cm=plt.cm.jet):
    difference[difference<threshold]=0
    difference_color = cm(difference/difference.max())[:,:,:3]
    # difference_color = (255*difference_color).astype('uint8')
    for i in range(difference_color.shape[0]):
        for j in range(difference_color.shape[1]):
            if difference_color[i,j,2]==0.5:
                for k in range(difference_color.shape[2]):
                    difference_color[i,j,k] = impaired[i,j,k]
    return difference_color

################ MAGNIFICATION CLASS ################
import torch, torch.nn as nn
from torchvision import transforms
import torchvision, glob
from PIL import Image
from skimage import io, transform

sys.path.append('../magnification/model/')
import network as net

class Generator:
    def __init__(self,z_dim=20,path_model='../extrapolation/'):
        self.data_transforms = transforms.Compose([
             transforms.Resize((128,128)),transforms.ToTensor()])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = net.VAE_FC6({'lstm_features':4096,'z_dim':z_dim})
        self.model.train(False).to(self.device)
        self.model.load_state_dict(torch.load(path_model + '/checkpoint_lowest_loss.pth.tar')['state_dict'])
        
    def encode(self,img,fc):
        pos = torch.from_numpy(fc).type(torch.FloatTensor).to(self.device)
        x = self.data_transforms(img).to(self.device)
        za, zp = self.model.get_latent_var(x, pos)
        return za, zp
    
    def decode(self,z_app,z_pos):
        z = torch.cat((z_app.to(self.device), z_pos.to(self.device)), 
                    dim=1).to(self.device)
        recon = self.model.decode(z).detach().cpu()
        return recon.numpy().reshape([3,128,128]).transpose([1,2,0])
    
    def extrapolate(self,z_app,z_pos,z_origin,l):
        z_origin = z_origin.to(self.device)
        z = (z_pos.to(self.device) - z_origin) * l + z_origin
        return self.decode(z_app,z)


################ MAGNIFICATION FUNCTIONS ################
def healthy_latent_space(generator, z_dim, indeces, videos, save=False):
    '''
    This function computes the mean latent representation (appearance and
    posture) for all healthy patients
    '''
    images, feat = [], []
    for i in range(indeces.shape[0]):
        video = videos[i]
        for j in range(indeces.shape[1]):
            img = Image.open(path_img + video + '/'+str(int(indeces[i,j])+1)  + '.jpg')
            fc6 = np.load(path_fc6 + video + '.npz')['fc6'][int(indeces[i,j])].reshape((1,4096))
            images.append(img); feat.append(fc6)
    
    z_app = torch.zeros([len(images), z_dim])
    z_pos = torch.zeros([len(images), z_dim])
    for idx in range(len(images)):
        za, zp = generator.encode(images[idx],feat[idx])
        z_app[idx], z_pos[idx] = za.detach(), zp.detach()
    
    z_mean_app   = torch.mean(z_app,dim=0).view((1,-1))
    z_mean_shape = torch.mean(z_pos,dim=0).view((1,-1))
    return (z_mean_app, z_mean_shape)

def encode(generator, z_dim, index, video):
    if not np.isscalar(index):
        z_pos_list = torch.zeros([len(index), z_dim])
        z_app_list = torch.zeros([len(index), z_dim])
        for i, idx in enumerate(index):
            img = Image.open(path_img+video+'/%d.jpg'%(int(idx)+1))
            fc6 = np.load(path_fc6 + video + '.npz')['fc6'][int(idx)].reshape((1,4096))
            z_app, z_pos = generator.encode(img,fc6)
            z_pos_list[i], z_app_list[i] = z_pos.detach(), z_app.detach()
        
        z_pos = torch.mean(z_pos_list,dim=0).view((1,-1))
        z_app = torch.mean(z_app_list,dim=0).view((1,-1))
    else:
        img = Image.open(path_img+video+'/%d.jpg'%(int(index)+1))
        fc6 = np.load(path_fc6 + video + '.npz')['fc6'][int(index)].reshape((1,4096))
        z_app, z_pos = generator.encode(img,fc6)
    
    return z_app, z_pos


################ INITIALIZE NETWORK ################
path_model = path_features = '../extrapolation/'
path_img      = '/export/home/mdorkenw/frames_keypoint/'
path_fc6      = '/export/home/mdorkenw/FC6/fc6_human/'

z_dim = 20
generator = Generator(z_dim,path_model)

### Load order of the sequences after alignment and video names (have the same order)
ranges     = np.load(path_features + 'ranges_552.npy').astype(int).reshape([-1,12,27])
video_list = np.load(path_features + 'video_list.npy')[:-1]

z_healthy = [healthy_latent_space(generator, z_dim, ranges[-9:,:,k], video_list[-9:]) 
                           for k in trange(27,desc='Healthy frame')]

# img = generator.decode(z_healthy[0][0],z_healthy[0][1])
# plt.imshow(img); plt.show()

################ LOAD DATA ################
out_fold = "./results/healthy_magnif_difference_class_online/"
if not os.path.exists(out_fold):  os.makedirs(out_fold)

names  = ['healthy','impaired','enhanced','Difference','Flow']
names_color=['g','b','r','r','r']

scale1, scale2 = 0.50, 25
F, K, L, Th, AVG = 26, 5, 2.5, 0.22, True
selratio = 0.06
fs = range(27)#np.linspace(0,26,F).astype(int).tolist()
fs = fs[6:]+fs[:6]

out_fold2 = out_fold+'lambda%.1f_kernel%d_threshold%.2f_OFscale1_%.2f_OFscale2_%d_selratio%.2f_useAVG%d/'%(L,K,Th,scale1,scale2,selratio,AVG)
if not os.path.exists(out_fold2): os.mkdir(out_fold2)

#videos = [f for f in os.listdir('../extrapolation/results/lambda_3.0/') if '.jpg' not in f]
#with open('videos.txt','w') as f:
    #_ = [f.write(v+'\n') for v in videos]

with open('videos.txt','r') as f:
    videos = f.readlines()

videos = [v[:-1] for v in videos]

v = f = 0
optical_flow = cv2.createOptFlow_DualTVL1()
for v in trange(len(videos)):
    video = videos[v]
    indexes = ranges[v]
    max_diff = 0
    images, flows = [], []
    distances = np.zeros(F)
    for f in range(F):
        z_app, z_pos = encode(generator, z_dim, indexes[:,f] if AVG else indexes[0,f], video)
        healthy  = generator.extrapolate(z_app,z_pos,z_healthy[f][1],0.0)
        impaired = generator.extrapolate(z_app,z_pos,z_healthy[f][1],1.0)
        enhanced = generator.extrapolate(z_app,z_pos,z_healthy[f][1],L)
        # DIFFERENCE
        difference = compute_difference(healthy,enhanced,kernel=K)
        # difference = compute_difference(impaired,enhanced,kernel=K)
        difference[0:difference.shape[0]/2] = 0
        diff_image = difference2color(difference,impaired,Th)
        distance = np.sort(difference.copy().flatten())
        distance = distance[-int(selratio*distance.shape[0]):].mean()
        distance = min(distance/0.25-0.0, 0.95)
        distances[f] = distance
        # RESIZE
        impaired_small   = (resize(healthy, scale1)*255).astype('uint8')
        # impaired_small   = (resize(impaired, scale1)*255).astype('uint8')
        enhanced_small   = (resize(impaired,scale1)*255).astype('uint8')
        difference_small = resize(difference,scale1)
        impaired_gray=cv2.cvtColor(impaired_small,cv2.COLOR_RGB2GRAY)
        enhanced_gray=cv2.cvtColor(enhanced_small,cv2.COLOR_RGB2GRAY)
        flow = optical_flow.calc(enhanced_gray, impaired_gray, None)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        x = np.linspace(0,impaired.shape[0]-1,flow.shape[0]).astype(int)
        X, Y = np.meshgrid(x, x)
        mask = np.logical_and(difference_small>=Th,mag>0.5)
        mask = cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8),iterations = 1)==1
        X, Y = X[mask], Y[mask]
        flow = flow/np.repeat(mag[:,:,np.newaxis],2,axis=2)
        flow_filter = np.stack([flow[:,:,0][mask],flow[:,:,1][mask]],axis=1)
        flow_filter = flow_filter[::scale2]
        X, Y = X[::scale2], Y[::scale2]
        images.append([healthy,impaired,enhanced,diff_image,impaired])
        flows.append([flow_filter,X,Y])
    
    _=plt.figure(figsize=(F*4,len(images[0])*5))
    for f in range(F):
        for i, im in enumerate(images[f]):
            _=plt.subplot(len(names)+1,F,i*F+f+1)
            _=plt.imshow(im)
            _=plt.axis('Off')
            if i==4:
                _=plt.quiver(flows[f][1], flows[f][2], 
                             flows[f][0][:,0], flows[f][0][:,1], 
                              width=0.1, headwidth=5, color='red', 
                              scale_units='width', 
                              scale=scale1*25,minlength=0.1)
            if f==0: _=plt.title(names[i],fontsize=20,color=names_color[i])
            else:    _=plt.title(str(fs[f]),fontsize=20)
    
    _=plt.subplot(len(names)+1,1,len(names)+1)
    _=plt.plot(range(F),distances,linewidth=10)
    _=plt.yticks(np.linspace(0,1.0,5))#np.arange(0.5,1.2))
    _=plt.xticks(range(F),fs)
    _=plt.ylim([0,1.0])
    _=plt.xlim([-0.5,float(F)-0.5])
    _=plt.title('pixel distance, sequence %d, %s'%(v,video.decode('utf-8')),color='b',fontsize=20)
    _=plt.savefig(out_fold2+video.decode('utf-8')+'.png',bbox_inches='tight',dpi=75)
    _=plt.savefig(out_fold2+video.decode('utf-8')+'.eps',bbox_inches='tight',dpi=75)
    _=plt.close('all')

