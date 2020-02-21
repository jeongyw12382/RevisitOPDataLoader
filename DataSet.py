import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from PIL import Image

DATASETS = ['roxford5k', 'rparis6k', 'revisitop1m']

data_path = '/home/yoonwoo/Data/revisitop/data/datasets'


def configdataset(dataset, dir_main):

    dataset = dataset.lower()

    if dataset not in DATASETS:    
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    if dataset == 'roxford5k' or dataset == 'rparis6k':
        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'

    elif dataset == 'revisitop1m':
        # loading imlist from a .txt file
        cfg = {}
        cfg['imlist_fname'] = os.path.join(dir_main, dataset, '{}.txt'.format(dataset))
        cfg['imlist'] = read_imlist(cfg['imlist_fname'])
        cfg['qimlist'] = []
        cfg['ext'] = ''
        cfg['qext'] = ''

    cfg['dir_data'] = os.path.join(dir_main, dataset)
    cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    # length of db
    cfg['n'] = len(cfg['imlist'])
    
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset

    return cfg

def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])

def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])

def read_imlist(imlist_fn):
    with open(imlist_fn, 'r') as file:
        imlist = file.read().splitlines()
    return imlist

class DBSet(Dataset):

    def __init__(self, dataset, transforms):
        self.cfg = configdataset(dataset, data_path)
        self.imlist = self.cfg['imlist']
        self.dir_images = self.cfg['dir_images']
        self.nDB = self.cfg['n']
        self.image_link_list = [self.dir_images + '/' + self.imlist[i] + self.cfg['ext'] for i in range(self.nDB)]
        self.transforms = transforms

    def get_image(self, idx):
        return Image.open(self.image_link_list[idx])

    def __len__(self):
        return self.nDB

    def __getitem__(self, idx):
        data = {}
        data['filename'] = self.imlist[idx]
        data['original_image'] = Image.open(self.image_link_list[idx])
        data['transformed_image'] = self.transforms(data['original_image'])
        return data


class QuerySet(Dataset):
    
    def __init__(self, dataset, transforms):
        self.DBSet = DBSet(dataset, transforms)
        self.cfg = configdataset(dataset, data_path)
        self.imlist = self.cfg['qimlist']
        self.dir_images = self.cfg['dir_images']
        self.nQ = self.cfg['nq']
        self.nDB = self.cfg['n']
        self.image_link_list = [self.dir_images + '/' + self.imlist[i] + self.cfg['qext'] for i in range(self.nQ)]
        self.transforms = transforms
        gt = self.cfg['gnd']
        self.bbx = [x['bbx'] for x in gt]
        self.easy = [x['easy'] for x in gt]
        self.hard = [x['hard'] for x in gt]
        self.junk = [x['junk'] for x in gt]

        self.pos = np.array([np.concatenate([self.easy[i], self.hard[i]]) for i in range(self.nQ)])
        self.pair = np.array([(x, y) for x in range(self.nQ) for y in self.pos[x]], dtype=(int,int))
        
        outlier = np.array([np.setdiff1d(np.arange(self.nDB), self.pos[i]) for i in range(self.nQ)])
        outlier = np.array([np.setdiff1d(outlier[i], self.junk[i]) for i in range(self.nQ)])
        self.neg = np.array([np.random.choice(mat, 5) for mat in outlier], dtype=int)
        # for i in range(len(self.junk)):
        #     if len(self.junk[i]) < 5:
        #         l = len(self.junk[i])
        #         self.junk[i] = np.concatenate([self.junk[i], np.random.choice(outlier[i], 5-l)])      
        # self.neg = np.array([np.random.choice(mat, 5) for mat in self.junk], dtype=int)

    def get_image(self, idx):
        return Image.open(self.image_link_list[idx])

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        data = {}
        query = self.pair[idx][0]
        data['q_filename'] = self.imlist[query]
        data['q_idx'] = query
        data['q_tf_image'] = self.transforms(Image.open(self.image_link_list[query]))
        data['pos_idx'] = self.pair[idx][1]
        data['pos_filename'] = self.DBSet.imlist[self.pair[idx][1]]
        data['pos_tf_image'] = self.transforms(Image.open(self.DBSet.image_link_list[self.pair[idx][1]]))
        data['neg_idx'] = self.neg[query]
        data['neg_filename'] = np.array([self.DBSet.imlist[self.neg[query][i]] for i in range(5)])
        data['neg_tf_image'] = [
            self.transforms(Image.open(self.DBSet.image_link_list[self.neg[query][i]]))
            for i in range(5)
        ]
        return data

if __name__ == '__main__':
    QuerySet('roxford5k', transforms=None)