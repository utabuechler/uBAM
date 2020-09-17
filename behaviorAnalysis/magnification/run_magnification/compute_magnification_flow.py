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

from skimage.transform import resize
from scipy.misc import imread, imsave

#import config_pytorch as cfg
import config_pytorch_human as cfg
import cv2

optical_flow = cv2.createOptFlow_DualTVL1()

source_path = cfg.results_path+'/magnification/magnification_pervideo/'
dest_path = cfg.results_path+'/magnification/magnification_pervideo_flow/'
if not os.path.exists(dest_path):
    os.mkdir(dest_path)

videos = sorted(os.listdir(source_path))
for video in tqdm(videos, desc="Compute Flow - videos"):
    files = sorted(os.listdir(source_path+'/'+video+'/impaired/'))
    files = [f for f in files if '.png' in f]
    flows = []
    for frame in tqdm(files, desc="Frames"):
        original = imread(source_path+'/'+video+'/impaired/'+frame)
        magnified= imread(source_path+'/'+video+'/magnified/'+frame)
        original = cv2.cvtColor(original,cv2.COLOR_RGB2GRAY)
        magnified= cv2.cvtColor(magnified,cv2.COLOR_RGB2GRAY)
        flow = optical_flow.calc(original, magnified, None)
        flows.append(flow)
        #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    flows = np.stack(flows, axis=0)
    np.savez(dest_path+'/'+video, flow=flows)
