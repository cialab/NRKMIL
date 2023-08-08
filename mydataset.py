import numpy as np
import glob
import os
from os.path import join
import random
import h5py
from torch.utils.data import Dataset
import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

class cam16(Dataset):
    def __init__(self, train='train', transform=None, r=None, keys='', split=42):

        self.img_dir = './CM16/Ziyu/feats/'#'./data/feats/cam16res'

        self.split = split
        
        postrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'tumor', '*.npy'))
        
        #print(postrainlist)
        negtrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'normal', '*.npy'))

        postrainlist, posvallist = train_test_split(postrainlist, test_size=0.1, random_state=self.split)
        negtrainlist, negvallist = train_test_split(negtrainlist, test_size=0.1, random_state=self.split)
        testnamelist = glob.glob(os.path.join(self.img_dir, 'test', '*', '*.npy'))

        if train == 'train':
            self.img_names = postrainlist + negtrainlist
        elif train == 'test':
            self.img_names = testnamelist
        elif train == 'val':
            self.img_names = posvallist + negvallist
            
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        image = np.load(img_path)

        label = img_path.split('/')[-2]
        if label == 'tumor':
            label = 1
        elif label == 'normal':
            label = 0

        if self.transform:
            image = self.transform(image)

        return torch.Tensor(image), label


class cam16_curcos(Dataset):
    def __init__(self, train='train', transform=None, r=None, keys='', k=-1, split=42):

        #self.img_dir = './data/feats/cam16res'
        self.img_dir = './CM16/Ziyu/feats/'
        #self.img_dir = './CM16/Ziyu/feats2/'
        
        self.split = split
        self.r = r
        self.keys = keys
        self.k = k
		
        postrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'tumor', '*.npy'))
        negtrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'normal', '*.npy'))

        #print(postrainlist)
        postrainlist, posvallist = train_test_split(postrainlist, test_size=0.1, random_state=self.split)
        negtrainlist, negvallist = train_test_split(negtrainlist, test_size=0.1, random_state=self.split)
        testnamelist = glob.glob(os.path.join(self.img_dir, 'test', '*', '*.npy'))

        if train == 'train':
            self.img_names = postrainlist + negtrainlist
        elif train == 'test':
            self.img_names = testnamelist
        elif train == 'val':
            self.img_names = posvallist + negvallist
            
        self.transform = transform
        self.sortdict = np.load("./CM16/ssim_cos_sorted_indexes/{}".format(self.keys), allow_pickle=True).item()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        image = np.load(img_path)
		
        sortidx = self.sortdict[img_path.split('/')[-1].split('.')[0]+'.npy']
		
        label = img_path.split('/')[-2]
        if label == 'tumor':
            label = 1
        elif label == 'normal':
            label = 0
            
        n = int(self.r * sortidx.shape[0])
        ids = sortidx[:n]
		
        return torch.Tensor(image[ids]), label

class cam16_curcos_att_maps(Dataset):
    def __init__(self, train='train', transform=None, r=None, keys='', split=42, slidePath = ""):

        #self.img_dir = './data/feats/cam16res'
        self.img_dir = './CM16/Ziyu/feats/'
        #self.img_dir = './CM16/Ziyu/feats2/'
        self.slidePath = slidePath
		
        self.split = split
        self.r = r
        self.keys = keys
		
        postrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'tumor', '*.npy'))
        negtrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'normal', '*.npy'))

        #print(postrainlist)
        postrainlist, posvallist = train_test_split(postrainlist, test_size=0.1, random_state=self.split)
        negtrainlist, negvallist = train_test_split(negtrainlist, test_size=0.1, random_state=self.split)
        testnamelist = glob.glob(os.path.join(self.img_dir, 'test', '*', '*.npy'))

        if train == 'train':
            self.img_names = postrainlist + negtrainlist
        elif train == 'test':
            self.img_names = testnamelist
        elif train == 'val':
            self.img_names = posvallist + negvallist
            
        self.transform = transform
        self.sortdict = np.load("./CM16/ssim_cos_sorted_indexes/{}".format(self.keys), allow_pickle=True).item()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = np.load(self.slidePath)
		
        sortidx = self.sortdict[self.slidePath.split('/')[-1].split('.')[0]+'.npy']
		
        label = 1
            
        n = int(self.r * sortidx.shape[0])
        ids = sortidx[:n]
		
        return torch.Tensor(image[ids]), label, ids

class cam16_curcos_ctranspath(Dataset):
    def __init__(self, train='train', transform=None, r=None, sortedDct='', k=-1, split=42):

        #self.img_dir = './data/feats/cam16res'
        self.img_dir = './CM16/CTranspath/cam16CTP/'
        #self.img_dir = './CM16/Ziyu/feats2/'
        
        self.split = split
        self.r = r
        self.k = k
		
        postrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'tumor', '*.npy'))
        negtrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'normal', '*.npy'))

        #print(postrainlist)
        postrainlist, posvallist = train_test_split(postrainlist, test_size=0.1, random_state=self.split)
        negtrainlist, negvallist = train_test_split(negtrainlist, test_size=0.1, random_state=self.split)
        testnamelist = glob.glob(os.path.join(self.img_dir, 'test', '*', '*.npy'))

        if train == 'train':
            self.img_names = postrainlist + negtrainlist
        elif train == 'test':
            self.img_names = testnamelist
        elif train == 'val':
            self.img_names = posvallist + negvallist
            
        self.transform = transform
        self.sortdict = np.load(sortedDct, allow_pickle=True).item()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        image = np.load(img_path)
		
        sortidx = self.sortdict[img_path.split('/')[-1].split('.')[0]+'.npy']
		
        label = img_path.split('/')[-2]
        if label == 'tumor':
            label = 1
        elif label == 'normal':
            label = 0
            
        n = int(self.r * sortidx.shape[0])
        ids = sortidx[:n]
		
        return torch.Tensor(image[ids]), label
        
class cam16_17_curcos_ctranspath(Dataset):
    def __init__(self, r=None, sortedDctPath = '', img_names = [], labelsFile = []):
        self.img_dir = './CM1617/cam1617CTP/'
        
        self.r = r
        self.img_names = img_names
        
        self.sortdict = np.load(sortedDctPath, allow_pickle=True).item()
        self.labelsList = np.load(labelsFile, allow_pickle = True).item()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        image = np.load(self.img_dir + img_path + ".npy")
		
        sortidx = self.sortdict[img_path.split('/')[-1].split('.')[0]+'.npy']
		
        #name = img_path.split('/')[-2]
        label = self.labelsList[img_path + ".npy"]
        
        n = int(self.r * sortidx.shape[0])
        ids = sortidx[:n]
		
        return torch.Tensor(image[ids]), label
        
        
class cam16_17_curcos_ctranspath_2(Dataset):
    def __init__(self, r=None, sortedDctPath = '', img_names = [], labelsFile = []):
        self.img_dir = './CM1617/cam1617CTP/'
        
        self.r = r
        self.img_names = img_names
        
        self.sortdict = np.load(sortedDctPath, allow_pickle=True).item()
        self.labelsList = np.load(labelsFile, allow_pickle = True).item()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        image = np.load(self.img_dir + img_path)
		
        sortidx = self.sortdict[img_path.split('/')[-1].split('.')[0]+'.npy']
		
        #name = img_path.split('/')[-2]
        label = self.labelsList[img_path]
        
        n = int(self.r * sortidx.shape[0])
        ids = sortidx[:n]
		
        return torch.Tensor(image[ids]), label