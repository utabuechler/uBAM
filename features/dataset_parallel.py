# --------------------------------------------------------
# Code for Python 2.7
# Class for creating the training and testing data
# --------------------------------------------------------

from Utils import load_table
import random
import numpy as np
from scipy.misc import imread
from PIL import Image #consider with PIL the first axis is x and the second is y!!!
from joblib import Parallel, delayed#for loading the images per sequence in parallel


def load_image(img_file):
    return Image.open(img_file)

class Dataset:
    
    def __init__(self,
                 det_file,
                 frames_path,
                 crops_saved = True,#in this case: frames_path should be =crops_path
                 train=True,
                 seq_len=8,
                 crop_size=227,
                 split_train = 0.95,
                 split_video = False,
                 batchsize=48,
                 idx_test=[]):
        '''
        EXPLANATION
        
        Input:
        Output:
        '''
        
        if np.mod(batchsize,2)!=0:
            raise ValueError("Please choose an even batchsize!")
        
        self.split_train = split_train
        self.batchsize=batchsize
        self.frames_path = frames_path
        self.F = seq_len
        self.crop_size = crop_size
        self.crops_saved = crops_saved
        self.train = train
        self.parallelizer = Parallel(n_jobs=seq_len)
        
        
        self._compute_denseSeq(det_file)
        if not train:
            self.idx_test = idx_test
            if len(idx_test)==0 and not split_video:
                raise ValueError("Please provide the video indizes for the testing videos given by the 'Dataset' object of the training data")
        
        self._split_trainTest(split_video)
        
        if self.train: random.shuffle(self.seqs)#shuffle the training data (but not the test data)
        
        self._construct_batches()
        
        
    def _compute_denseSeq(self,det_file):
        '''
        EXPLANATION
        
        Input:
        Output:
        '''
        detections = load_table(det_file,asDict=False)#the detections are given in an excel or csv file

        self.videos = np.unique(detections['videos'].values)
        self.N_vids = len(self.videos)

        #print('Computing dense sequences...')
        self.seq_dense = []#initialize dictionary
        #for iter in trange(0,self.N_vids):
        for iter in range(self.N_vids):
            idx = [i for i,vid in enumerate(detections['videos'].values) if vid==self.videos[iter]]#get all entries for the current video
            frames = detections['frames'][idx].values#get all frames where we have detections
            coords = np.array([detections['x1'][idx].values,detections['y1'][idx].values,#get all detections
                    detections['x2'][idx].values,detections['y2'][idx].values]).T
            #if self.crops_saved: orig_size = np.array([detections['original_size_x'][idx].values,detections['original_size_y'][idx].values]).T
            orig_size = [detections['original_size_x'][idx[0]],detections['original_size_y'][idx[0]]] if self.crops_saved else [0,0]
            
            #it might be that some detections inbetween are missing, so find out where we have a gap
            gap = np.where(((frames[1:]-frames[:-1])>1)*1)[0]
            if len(gap)==0:#if all frames are consecutive
                connectSeq = [range(len(frames))]
            else:#get all connected sequences
                connectSeq = [range(gap[0]+1)]#first sequence
                connectSeq.extend([range(gap[i]+1,gap[i+1]+1) for i in range(len(gap)-1)])#all sequences inbetween
                connectSeq.extend([range(gap[-1]+1,len(frames))])#last sequence
            
            #create the dense sequences
            idx_dense = []
            for i in range(len(connectSeq)):
                seq = connectSeq[i]
                idx_dense.extend([range(seq[i],seq[i]+self.F) for i in range(len(seq)-self.F+1)])
            
            #save all information for the dense sequences
            idx_dense = np.array(idx_dense)
            frames_dense = frames[idx_dense]
            coords_dense = coords[idx_dense]
            seq_dense_iter = {'video': self.videos[iter],'frames':frames_dense,'coords':coords_dense,'orig_size':orig_size}
            self.seq_dense.append(seq_dense_iter)
        
    
    def _split_trainTest(self,split_video):
        '''
        EXPLANATION
        
        Input:
        Output:
        '''
        
        if split_video:#split the videos itself (per video use the first self.split_train sequences as training and the rest of the video as testing)
            #useful in case the behavior changes within one video significantly and/or not enough separate videos are available
            
            self.seqs = []
            #self.seq_test = []
            for iter in range(self.N_vids):
                n = len(self.seq_dense[iter]['frames'])#number of sequences
                if self.train:
                    idx = range(int(np.round(self.split_train*n)))#the first self.split_train percentage of sequences are for training
                else:
                    idx = range(int(np.round(self.split_train*n)),n)#the last ones for testing
                
                #vid = [self.seq_dense[iter]['video']]*n
                vid = self.seq_dense[iter]['video']
                orig_size = self.seq_dense[iter]['orig_size']
                frames = self.seq_dense[iter]['frames'].tolist()
                coords = self.seq_dense[iter]['coords'].tolist()
                
                _=[self.seqs.append([vid,frames[i],coords[i],orig_size]) for i in idx]
                
        
        else:#split between the videos (take self.split_train of the videos (randomly) for training and the rest for testing)
            if self.train:
                idx_all = np.arange(self.N_vids)
                random.shuffle(idx_all)
                idx = idx_all[:int(np.round(self.split_train*self.N_vids))]
                self.idx_test = idx_all[int(np.round(self.split_train*self.N_vids)):]#we need to save it somehow for later when creating an object for the testing data
            else:
                idx = self.idx_test
            
            ## save, so that one entry in seqs corresponds to one sequence
            
            self.seqs = []
            for iter in idx:#for loop over the videos we want for training
                n = len(self.seq_dense[iter]['frames'])#number of sequences
                #vid = [self.seq_dense[iter]['video']]*n
                vid = self.seq_dense[iter]['video']
                orig_size = self.seq_dense[iter]['orig_size']
                frames = self.seq_dense[iter]['frames'].tolist()
                coords = self.seq_dense[iter]['coords'].tolist()
                _=[self.seqs.append([vid,frames[i],coords[i],orig_size]) for i in range(n)]
        
        self.N_seqs = len(self.seqs)#total number of sequences for training
        
    def _construct_batches(self):
        '''
        EXPLANATION
        
        Input:
        Output:
        '''
        self.N = int(np.floor(self.N_seqs/float(self.batchsize/2)))#'self.batchsize/2' because the batch is filled half with the real sequences and half with its shuffled version
        self.batches = [self.seqs[i*(self.batchsize/2):(i+1)*(self.batchsize/2)] for i in range(self.N)]
    
    def shuffle(self):
        '''
        EXPLANATION
        
        Input:
        Output:
        '''
        random.shuffle(self.seqs)
        self._construct_batches()
    
    def __len__(self):
        return self.N
    
    def augment(self,image,c):
        '''
        EXPLANATION
        
        Input:
        Output:
        '''
        
        #1. crop randomly around the given bounding box
        w = c[2]-c[0]
        h = c[3]-c[1]
        s = image.size
        
        #we need a translation which is valid, i.e. it cannot go beyond the image (in case the bounding box (c) is at the border of the image)
        transl_w = random.randint(max(-c[0],int(-0.05*w)),min(s[0]-c[2],int(0.05*w)))
        transl_h = random.randint(max(-c[1],int(-0.05*h)),min(s[1]-c[3],int(0.05*h)))
        
        #2. rgb jittering
        image = np.asarray(image.crop((c[0]+transl_w,c[1]+transl_h,c[2]+transl_w,c[3]+transl_h)),np.float32)
        for ch in range(3):
            thisRand = random.uniform(0.8, 1.2)
            image[:,:,ch] *= thisRand
        
        shiftVal = random.randint(0,6)
        if random.randint(0,1) == 1:
            shiftVal = -shiftVal
        image += shiftVal
        image = Image.fromarray(image.astype(np.uint8))
        
        
        return image
    
    def __getitem__(self,index):
        '''
        Load a new batch
        
        Input:
        Output:
        '''
        
        
        #get all the information for the current batch (videos, frames and coords)
        batch_info = self.batches[index]
        
        #initialize
        batch = np.empty((self.F*self.batchsize,self.crop_size,self.crop_size,3))
        labels = np.empty((self.F*self.batchsize,))
        
        #build the batch
        for s in range(self.batchsize/2):#'/2' because half of the sequences are the real ones and half the shuffled ones
            imgSeq_pos = np.empty((self.F,self.crop_size,self.crop_size,3))#initialize
            
            #get all information for current sequence
            vid = batch_info[s][0]
            frames = batch_info[s][1]
            coords = batch_info[s][2]
            orig_size = batch_info[s][3]
            
            #load the images in parallel
            #parallelizer = Parallel(n_jobs=self.F)
            tasks_iterator = (delayed(load_image)('%s%s/%06d.jpg'%(self.frames_path,vid,frames[i])) for i in range(self.F))
            images = []
            images.extend(self.parallelizer(tasks_iterator))
                
            
            #for loop over the frames per sequence
            for iter,ci in enumerate(coords):
                c = [int(i) for i in ci]#make sure that it is int
                image = images[iter]#self.frames_path is either the path to the crops or the path to the original frames
                #image = Image.open('%s%s/%06d.jpg'%(self.frames_path,vid,f))#self.frames_path is either the path to the crops or the path to the original frames
                sz = orig_size if self.crops_saved else image.size #original size of the frames
                c[0],c[1],c[2],c[3] = max(0,c[0]),max(0,c[1]),min(sz[0],c[2]),min(sz[1],c[3])#make sure that the bounding box is not outside of the original image
                if self.crops_saved:
                    #we cropped 5% more on every side (when saving the croppings in preprocessing), so define where we need to crop now based on the original coordinates
                    w,h = c[2]-c[0],c[3]-c[1]
                    c_new = [0]*len(c)
                    c_new[0] = int(0.05*w) if (c[0]-int(0.05*w))>0 else c[0] #the original cropping starts either at the 5% or c[0]
                    c_new[1] = int(0.05*h) if (c[1]-int(0.05*h))>0 else c[1] #the original cropping starts either at the 5% or c[1]
                    c_new[2] = image.size[0]-int(0.05*w) if (c[2]+int(0.05*w))<sz[0] else image.size[0]#the original cropping ends either at 5% more, or at the end of the image
                    c_new[3] = image.size[1]-int(0.05*h) if (c[3]+int(0.05*h))<sz[1] else image.size[1]#the original cropping ends either at 5% more, or at the end of the image
                    
                    c = c_new
                    
                
                #do augmentation
                if self.train:
                    image = self.augment(image,c)
                else:
                    #crop the given area without using augmentation
                    image = image.crop(c)
                    
                #resize the image to the final size
                image = np.array(image.resize((self.crop_size,self.crop_size),Image.BILINEAR))
                
                #change from RGB to BGR
                image = image[:, :, ::-1]
                
                #add image to sequence
                imgSeq_pos[iter,:,:,:] = np.reshape(image,(1,self.crop_size,self.crop_size,3))
            
            #randomly permut for the negative
            imgSeq_neg = np.copy(imgSeq_pos)
            random.shuffle(imgSeq_neg)
        
            #concatenate the positives and negatives
            batch[s*2*self.F:(s+1)*2*self.F,:,:,:] = np.concatenate((imgSeq_pos,imgSeq_neg),axis=0)
            
            #save the labels
            labels[s*2*self.F:(s+1)*2*self.F] = np.concatenate((np.ones(self.F,),np.zeros(self.F,)))
            
        #change order (as input in CNN (with Caffe) we need at first all first frames of the sequences, then every second and so on)
        newOrder = np.zeros((labels.shape),np.int)
        for iter in range(self.F):
            newOrder[iter*self.batchsize:(iter+1)*self.batchsize] = np.arange(iter,self.F*self.batchsize,self.F).astype(np.int)
            
        batch2 = batch[newOrder,:,:,:]
        labels2 = labels[newOrder]
        
        batch2 = np.array(batch2, dtype=np.float32)
        
        #change from [N,y,x,ch] to [N,ch,y,x]
        batch2 = batch2.transpose((0,3,1,2))
        
        return batch2,labels2
    
    
#batch_info = data_train.batches[index]
        
##initialize
#batch = np.empty((F*batchsize,crop_size,crop_size,3))
#labels = np.empty((F*batchsize,))

##build the batch
#for s in range(batchsize/2):#'/2' because half of the sequences are the real ones and half the shuffled ones
#imgSeq_pos = np.empty((F,crop_size,crop_size,3))#initialize

##get all information for current sequence
#vid = batch_info[s][0]
#frames = batch_info[s][1]
#coords = batch_info[s][2]
#orig_size = batch_info[s][3]

##load the images in parallel
##parallelizer = Parallel(n_jobs=self.F)
#tasks_iterator = (delayed(load_image)('%s%s/%06d.jpg'%(frames_path,vid,frames[i])) for i in range(F))
#images = []
#images.extend(data_train.parallelizer(tasks_iterator))
        
    
    ##for loop over the frames per sequence
    #for iter,ci in enumerate(coords):
        #c = [int(i) for i in ci]#make sure that it is int
        #image = images[iter]#self.frames_path is either the path to the crops or the path to the original frames
        ##image = Image.open('%s%s/%06d.jpg'%(self.frames_path,vid,f))#self.frames_path is either the path to the crops or the path to the original frames
        #sz = orig_size if self.crops_saved else image.size #original size of the frames
        #c[0],c[1],c[2],c[3] = max(0,c[0]),max(0,c[1]),min(sz[0],c[2]),min(sz[1],c[3])#make sure that the bounding box is not outside of the original image
        #if self.crops_saved:
            ##we cropped 5% more on every side (when saving the croppings in preprocessing), so define where we need to crop now based on the original coordinates
            #w,h = c[2]-c[0],c[3]-c[1]
            #c_new = [0]*len(c)
            #c_new[0] = int(0.05*w) if (c[0]-int(0.05*w))>0 else c[0] #the original cropping starts either at the 5% or c[0]
            #c_new[1] = int(0.05*h) if (c[1]-int(0.05*h))>0 else c[1] #the original cropping starts either at the 5% or c[1]
            #c_new[2] = image.size[0]-int(0.05*w) if (c[2]+int(0.05*w))<sz[0] else image.size[0]#the original cropping ends either at 5% more, or at the end of the image
            #c_new[3] = image.size[1]-int(0.05*h) if (c[3]+int(0.05*h))<sz[1] else image.size[1]#the original cropping ends either at 5% more, or at the end of the image
            
            #c = c_new
            
        
        ##do augmentation
        #if self.train:
            #image = self.augment(image,c)
        #else:
            ##crop the given area without using augmentation
            #image = image.crop(c)
            
        ##resize the image to the final size
        #image = np.array(image.resize((self.crop_size,self.crop_size),Image.BILINEAR))
        
        ##change from RGB to BGR
        #image = image[:, :, ::-1]
        
        ##add image to sequence
        #imgSeq_pos[iter,:,:,:] = np.reshape(image,(1,self.crop_size,self.crop_size,3))
    
    ##randomly permut for the negative
    #imgSeq_neg = np.copy(imgSeq_pos)
    #random.shuffle(imgSeq_neg)

    ##concatenate the positives and negatives
    #batch[s*2*self.F:(s+1)*2*self.F,:,:,:] = np.concatenate((imgSeq_pos,imgSeq_neg),axis=0)
    
    ##save the labels
    #labels[s*2*self.F:(s+1)*2*self.F] = np.concatenate((np.ones(self.F,),np.zeros(self.F,)))
    
##change order (as input in CNN (with Caffe) we need at first all first frames of the sequences, then every second and so on)
#newOrder = np.zeros((labels.shape),np.int)
#for iter in range(self.F):
    #newOrder[iter*self.batchsize:(iter+1)*self.batchsize] = np.arange(iter,self.F*self.batchsize,self.F).astype(np.int)
    
#batch2 = batch[newOrder,:,:,:]
#labels2 = labels[newOrder]

#batch2 = np.array(batch2, dtype=np.float32)

##change from [N,y,x,ch] to [N,ch,y,x]
#batch2 = batch2.transpose((0,3,1,2))

#return batch2,labels2