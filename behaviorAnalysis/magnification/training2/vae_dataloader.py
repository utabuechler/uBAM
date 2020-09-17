import numpy as np, time, random, csv, os
from tqdm import tqdm
from utils import load_table
from skimage import io, transform

import torch

"""============================================"""
#dataset for dataloader
class dataset(torch.utils.data.Dataset):

    def __init__(self, opt):

        self.data_path  = opt.Paths['img']
        self.frame_path = opt.Paths['frame_path']
        table = load_table(opt.Paths['detections'],asDict=False)
        self.info = {'videos':table['videos'].values,
                     'frames':table['frames'].values}
        ### Determine length of our training set
        uni_videos = np.unique(self.info['videos'])
        if opt.Training['size'] is None:
            self.length = len(self.info['frames'])
        else:
            per_video = int(opt.Training['size']/len(uni_videos))
            #self.length = min(opt.Training['size'],len(self.info['frames']))
            selection = [np.where(self.info['videos']==v)[0][:per_video] 
                            for v in uni_videos]#np.random.permutation(len(self.info['frames']))
            selection = np.concatenate(selection)
            self.info['videos'] = self.info['videos'][selection]
            self.info['frames'] = self.info['frames'][selection]
            self.length = len(self.info['frames'])
        
        ### Load fc6 features into the RAM
        data_iter = tqdm(uni_videos,position=2)
        data_iter.set_description('Load Posture Representation')
        
        self.fc6 = []
        for i, v in enumerate(data_iter):
            frames = self.info['frames'][self.info['videos']==v]
            features_file = np.load(opt.Paths['fc6'] + v + '.npz')
            selection = [np.where(features_file['frames']==frame)[0][0] 
                         for frame in frames 
                         if frame in features_file['frames']]
            self.fc6.append(features_file['fc6'][selection])
            # self.fc6.append(features_file['fc6'])
        
        self.fc6 = np.concatenate(self.fc6,0)
        assert len(self.fc6.shape)==2, 'Features not properly concatenated'
        assert self.fc6.shape[0]==self.info['videos'].shape[0],'Wrong number of features loaded: %d - %d'%(self.info['videos'].shape[0],self.fc6.shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video, frame = self.info['videos'][idx], self.info['frames'][idx]
        frames_sel = self.info['videos']==video
        frames = self.info['frames'][frames_sel]
        iframe = np.where(frames==frame)[0][0]
        F = len(frames)
        ### Load original image
        image_orig          = load_image(self.data_path,video,frame)

        rand_inter          = np.random.randint(2,min(4,F),1)[0]
        if (np.random.rand()<0.5 and iframe>=rand_inter) or iframe>F-rand_inter-1:
            rand_inter *= -1
        # if int(idx%200) > 100:
        #     rand_inter      *= -1
        image_inter         = load_image(self.data_path,video,frames[iframe+rand_inter])
        
        rand                = int(rand_inter/2)
        image_inter_truth   = load_image(self.data_path,video,frames[iframe+rand])
                    
        ### Load random images
        rand            = np.random.randint(0,F,1)[0]
        image_rand1     = load_image(self.frame_path,video,frames[rand])

        rand            = np.random.randint(0,F,1)[0]
        image_rand2     = load_image(self.frame_path,video,frames[rand])

        ### Downsample images
        image_orig          = transform.resize(image_orig, (128, 128), mode='constant').transpose((2, 0, 1))
        image_inter         = transform.resize(image_inter, (128, 128), mode='constant').transpose((2, 0, 1))
        image_inter_truth   = transform.resize(image_inter_truth, (128, 128), mode='constant').transpose((2, 0, 1))
        image_rand1         = transform.resize(image_rand1, (128, 128), mode='constant').transpose((2, 0, 1))
        image_rand2         = transform.resize(image_rand2, (128, 128), mode='constant').transpose((2, 0, 1))

        sample = {'image_orig': image_orig,'image_inter': image_inter, 
                  'image_inter_truth': image_inter_truth,'image_rand1':image_rand1, 
                  'image_rand2':image_rand2, 'fc6': self.fc6[idx],
                  'fc6_inter': self.fc6[frames_sel][iframe+rand_inter]}

        return sample

def load_image(path,v,f):
    name = '%s%s/%06d.jpg'%(path,v,f)
    if not os.path.exists(name): name = '%s%s/%d.jpg'%(path,v,f)
    im = io.imread(name)
    return im
