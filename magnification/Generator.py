# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:50:39 2017
@author: Biagio Brattoli
"""
import os, sys, numpy as np
import matplotlib.pyplot as plt

################ MAGNIFICATION CLASS ################
import torch, torch.nn as nn
from torchvision import transforms
import torchvision, glob
from PIL import Image
#from skimage import io, transform

sys.path.append('./magnification/')
import network as VAE

class Generator:
    def __init__(self,z_dim=20,device='gpu',path_model='../extrapolation/'):
        self.data_transforms = transforms.Compose([
             transforms.Resize((128,128)),transforms.ToTensor()])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and device!='cpu' else "cpu")
        self.vae = VAE.VAE_FC6({'feature_size':4096,'z_dim':z_dim})
        self.vae.train(False).to(self.device)
        if device=='cpu':
            init_weights = torch.load(path_model + '/checkpoint.pth.tar',map_location=lambda storage, loc: storage)
        else:
            init_weights = torch.load(path_model + '/checkpoint.pth.tar')
        
        self.vae.load_state_dict(init_weights['state_dict'])
        self.z_dim = z_dim
        
    def encode(self,img,fc):
        if isinstance(img, np.ndarray): img = Image.fromarray(img)
        pos = torch.from_numpy(fc.reshape([1,-1])).type(torch.FloatTensor).to(self.device)
        x = self.data_transforms(img).to(self.device)
        za, zp = self.vae.get_latent_var(x, pos)
        return za, zp
    
    def decode(self,z_app,z_pos):
        if isinstance(z_app, np.ndarray): z_app = torch.from_numpy(z_app)
        if isinstance(z_pos, np.ndarray): z_pos = torch.from_numpy(z_pos)
        z = torch.cat((z_app.to(self.device), z_pos.to(self.device)), 
                    dim=1).to(self.device)
        recon = self.vae.decode(z).detach().cpu()
        return recon.numpy().reshape([3,128,128]).transpose([1,2,0])
    
    def extrapolate(self,z_app,z_pos,z_origin,l):
        z_origin = z_origin.to(self.device)
        z = (z_pos.to(self.device) - z_origin) * l + z_origin
        return self.decode(z_app,z)
    
    def average(self,images,fc6):
        pos_list = torch.zeros([len(images), self.z_dim])
        app_list = torch.zeros([len(images), self.z_dim])
        apperance= None
        for i, (img, fc) in enumerate(zip(images,fc6)):
            app, pos = self.encode(img,fc)
            pos_list[i], app_list[i] = pos, app
        
        posture  = torch.mean(pos_list,dim=0).view((1,-1)).to(self.device)
        apperance= torch.mean(app_list,dim=0).view((1,-1)).to(self.device)
        return apperance, posture
    
    def extrapolate_multiple(self,healthy_images,healthy_fc6,
                                  impaired_images,impaired_fc6,
                                  lambdas):
        _,  h_pos  = self.average(healthy_images, healthy_fc6 )
        im_app, im_pos = self.average(impaired_images,impaired_fc6)
        results = []
        for l in lambdas:
            z_pos = (im_pos - h_pos) * l + h_pos
            img = self.decode(im_app, z_pos)
            results.append(img)
        
        return results

################ FUNCTIONS ################
#from skimage.morphology import binary_dilation
#from skimage.transform import resize as skresize
#from skimage.color import rgb2gray
from time import time
from scipy.misc import imresize
from skimage.morphology import erosion, dilation, disk
from skimage import measure

try:
    import cv2
except:
    print('OpenCV not available for difference and direction!!')

def find_differences_cc(healthy,impaired,enhanced,Th=0.25,scale=20):
    optical_flow = cv2.createOptFlow_DualTVL1()
    scale1 = 0.5
    # RGB DIFFERENCE
    t = time()
    difference = compute_difference(impaired,enhanced,kernel=5)
    #print('Difference in %.2f'%(time()-t))
    difference[0:difference.shape[0]//6] = 0
    difference[-difference.shape[0]//8:]  = 0
    difference[:,0:difference.shape[1]//6] = 0
    difference[:,-difference.shape[1]//6:]  = 0
    t = time()
    diff_image = difference2color(difference,impaired,Th).astype('float32')/255
    #print('DiffImage in %.2f'%(time()-t))
    # FLOW
    t = time()
    impaired, enhanced = resize(impaired,scale1), resize(enhanced,scale1)
    impaired_gray=cv2.cvtColor(impaired,cv2.COLOR_RGB2GRAY)
    enhanced_gray=cv2.cvtColor(enhanced,cv2.COLOR_RGB2GRAY)
    flow = optical_flow.calc(impaired_gray, enhanced_gray, None) # compute flow
    #print('Flow in %.2f'%(time()-t))
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    flow = flow/np.repeat(mag[:,:,np.newaxis],2,axis=2)
    # MASK
    t = time()
    mask = difference>=Th
    #mask = np.logical_and(mask,mag>1.0)
    mask = dilation(mask, disk(6))
    mask = erosion(mask, disk(3))
    #print('Mask in %.2f'%(time()-t))
    t = time()
    mask_labels = measure.label(mask, background=0)
    labels = np.unique(mask_labels)
    flow_sparse, areas = [], []
    for l in range(1,labels.max()+1):
        m = np.where(mask_labels==l)
        #print(len(m[0]))
        if len(m[0])<150: continue
        
        m = [(m[0]*scale1).astype(int), (m[1]*scale1).astype(int)]
        fx, fy = flow[m[0],m[1],0].max(), flow[m[0],m[1],1].max()
        x, y = int(np.median(m[0])/scale1), int(np.median(m[1])/scale1)
        flow_sparse.append([x,y,fx,fy])
        areas.append(len(m[0]))
    
    #print('CC in %.2f'%(time()-t))
    #diff_image = mask_labels
    diff_image = resize(diff_image, 1/scale1)
    #print diff_image.dtype
    if len(flow_sparse)==0:
        return diff_image,np.zeros([1,2]),[0],[0],np.zeros(1)
    
    flow_sparse = np.array(flow_sparse)
    return diff_image,flow_sparse[:,2:],flow_sparse[:,1].astype(int),flow_sparse[:,0].astype(int), np.array(areas)


def find_differences(healthy,impaired,enhanced,Th=0.25,scale=20):
    scale1, scale2 = 1.0, scale
    optical_flow = cv2.createOptFlow_DualTVL1()
    #optical_flow = cv2.DualTVL1OpticalFlow_create()
    # RGB DIFFERENCE
    difference = compute_difference(impaired,enhanced,kernel=5)
    #difference[0:difference.shape[0]/5] = 0
    #difference[difference.shape[0]/8:]  = 0
    #difference[:,0:difference.shape[1]/5] = 0
    #difference[:,difference.shape[1]/5:]  = 0
    diff_image = difference2color(difference,impaired,Th)
    # FLOW
    impaired_small   = (resize(impaired, scale1)*255).astype('uint8') # resize for reducing flow resolution
    enhanced_small   = (resize(enhanced,scale1)*255).astype('uint8')
    diff_rescale = np.linspace(0,difference.shape[0]-1,int(scale1*difference.shape[0])).astype(int)
    difference_small = difference[diff_rescale,diff_rescale]#resize(difference,scale1)
    impaired_gray=cv2.cvtColor(impaired_small,cv2.COLOR_RGB2GRAY)
    enhanced_gray=cv2.cvtColor(enhanced_small,cv2.COLOR_RGB2GRAY)
    flow = optical_flow.calc(enhanced_gray, impaired_gray, None) # compute flow
    # FLOW VECTOR TO IMAGE
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    flow = flow/np.repeat(mag[:,:,np.newaxis],2,axis=2)
    mask = np.logical_and(difference_small>=Th,mag>2)
    flow_filter = np.stack([flow[:,:,0]*mask,flow[:,:,1]*mask],axis=2)
    flow_filter_sparse = np.zeros([int(flow_filter.shape[0]/scale1),
                                int(flow_filter.shape[1]/scale1),
                                flow_filter.shape[2]],flow_filter.dtype)
    for x in range(0,flow_filter.shape[0]-scale2//2,scale2):
        for y in range(0,flow_filter.shape[1]-scale2//2,scale2):
            X, Y = int((x+scale2/2)/scale1), int((y+scale2/2)/scale1)
            m = flow_filter[x:x+scale2,y:y+scale2].reshape([-1,2])
            mx, my = m[:,0], m[:,1]
            #m = np.stack([mx[mx!=0].mean(0),my[my!=0].mean(0)])
            m = np.stack([mx.max(),my.max()])
            flow_filter_sparse[X,Y] = m if not np.isnan(m).any() else 0
    
    mag, ang = cv2.cartToPolar(flow_filter_sparse[...,0], flow_filter_sparse[...,1])
    flow_filter_sparse = flow_filter_sparse/(np.repeat(mag[:,:,np.newaxis],2,axis=2)+0.01)
    #flow_filter_sparse = np.stack([resize(flow_filter[:,:,0],1/scale1),resize(flow_filter[:,:,1],1/scale1)],2)
    X, Y = np.where(np.logical_or(flow_filter_sparse[:,:,0]!=0,flow_filter_sparse[:,:,1]!=0))
    return diff_image,flow_filter_sparse,X,Y


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def resize(image,scale):
    result = imresize(image, (int(scale*float(image.shape[0])), 
                            int(scale*float(image.shape[1]))))
    if image.dtype=='float32':
        result = result.astype(image.dtype)/255
    
    return result


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
    difference = difference.copy()
    difference[difference<threshold]=0
    difference_color = cm(difference/difference.max())[:,:,:3]
    # difference_color = (255*difference_color).astype('uint8')
    for i in range(difference_color.shape[0]):
        for j in range(difference_color.shape[1]):
            if difference_color[i,j,2]==0.5:
                for k in range(difference_color.shape[2]):
                    difference_color[i,j,k] = impaired[i,j,k]
    return difference_color


################ MAGNIFICATION FUNCTIONS ################
#def healthy_latent_space(generator, z_dim, indeces, videos, save=False):
    #'''
    #This function computes the mean latent representation (appearance and
    #posture) for all healthy patients
    #'''
    #images, feat = [], []
    #for i in range(indeces.shape[0]):
        #video = videos[i]
        #for j in range(indeces.shape[1]):
            #img = Image.open(path_img + video + '/'+str(int(indeces[i,j])+1)  + '.jpg')
            #fc6 = np.load(path_fc6 + video + '.npz')['fc6'][int(indeces[i,j])].reshape((1,4096))
            #images.append(img); feat.append(fc6)
    
    #z_app = torch.zeros([len(images), z_dim])
    #z_pos = torch.zeros([len(images), z_dim])
    #for idx in range(len(images)):
        #za, zp = generator.encode(images[idx],feat[idx])
        #z_app[idx], z_pos[idx] = za.detach(), zp.detach()
    
    #z_mean_app   = torch.mean(z_app,dim=0).view((1,-1))
    #z_mean_shape = torch.mean(z_pos,dim=0).view((1,-1))
    #return (z_mean_app, z_mean_shape)

#def encode(generator, z_dim, index, video):
    #if not np.isscalar(index):
        #z_pos_list = torch.zeros([len(index), z_dim])
        #z_app_list = torch.zeros([len(index), z_dim])
        #for i, idx in enumerate(index):
            #img = Image.open(path_img+video+'/%d.jpg'%(int(idx)+1))
            #fc6 = np.load(path_fc6 + video + '.npz')['fc6'][int(idx)].reshape((1,4096))
            #z_app, z_pos = generator.encode(img,fc6)
            #z_pos_list[i], z_app_list[i] = z_pos.detach(), z_app.detach()
        
        #z_pos = torch.mean(z_pos_list,dim=0).view((1,-1))
        #z_app = torch.mean(z_app_list,dim=0).view((1,-1))
    #else:
        #img = Image.open(path_img+video+'/%d.jpg'%(int(index)+1))
        #fc6 = np.load(path_fc6 + video + '.npz')['fc6'][int(index)].reshape((1,4096))
        #z_app, z_pos = generator.encode(img,fc6)
    
    #return z_app, z_pos

