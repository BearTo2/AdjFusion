from random import sample, shuffle

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class Dataset(Dataset):
    def __init__(self, ir_path, vi_path, input_shape, epoch_length):
        super(Dataset, self).__init__()
        self.ir_path            = ir_path
        self.vi_path            = vi_path
        self.input_shape        = input_shape
        self.epoch_length       = epoch_length
        self.train              = False

        self.epoch_now          = -1
        self.length             = len(self.ir_path)

    def __len__(self):
        return self.length
    
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def __getitem__(self, index):
        index = index % self.length
        p = self.rand(0,1)
        ir_image, x, y = self.get_random_data(self.ir_path[index], self.input_shape, p)
        ir_image = np.transpose(preprocess_input(np.array(ir_image, dtype=np.float32)), (2, 0, 1))

        vi_image, x, y = self.get_random_data(self.vi_path[index], self.input_shape, p, x, y)
        vi_image = np.transpose(preprocess_input(np.array(vi_image, dtype=np.float32)), (2, 0, 1))
        
        en_image, x, y = self.get_random_data(self.vi_path[index], self.input_shape, p, x, y)
        en_image = np.transpose(preprocess_input(np.array(en_image, dtype=np.float32)), (2, 0, 1))
        
        vi_mark, x, y = self.get_random_data(self.vi_path[index].replace('visible','Vis_annotation'), self.input_shape, p, x, y)
        vi_mark = np.transpose(preprocess_input(np.array(vi_mark, dtype=np.float32)), (2, 0, 1))
        
        ir_mark, x, y = self.get_random_data(self.vi_path[index].replace('visible','Ir_annotation'), self.input_shape, p, x, y)
        ir_mark = np.transpose(preprocess_input(np.array(ir_mark, dtype=np.float32)), (2, 0, 1))
        #print(ir_mark.shape)
        return ir_image, vi_image, en_image, vi_mark, ir_mark

    def get_random_data(self, annotation_line, input_shape, p, x=None, y=None):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #if self.epoch_now>=30:
        #    return image, x, y
        h, w = input_shape
        PS = 0.8
        if p > 0.8:
            image = image.resize((w, h), Image.BICUBIC)
        else:
            if x == None:
                W, H = image.size
                x, y = self.rand(0,W-w), self.rand(0,H-h)
            image = image.crop([x,y,x+w,y+h])
        return image, x, y

    
# DataLoader中collate_fn使用
def dataset_collate(batch):
    ir_images  = []
    vi_images = []
    en_images = []
    vi_marks = []
    ir_marks = []
    for ir_img, vi_img, en_img, vi_mark, ir_mark in batch:
        ir_images.append(ir_img)
        vi_images.append(vi_img)
        en_images.append(en_img)
        vi_marks.append(vi_mark)
        ir_marks.append(ir_mark)
            
    ir_images  = torch.from_numpy(np.array(ir_images)).type(torch.FloatTensor)
    vi_images  = torch.from_numpy(np.array(vi_images)).type(torch.FloatTensor)
    en_images  = torch.from_numpy(np.array(en_images)).type(torch.FloatTensor)
    vi_marks  = torch.from_numpy(np.array(vi_marks)).type(torch.FloatTensor)
    ir_marks  = torch.from_numpy(np.array(ir_marks)).type(torch.FloatTensor)
    return ir_images, vi_images, en_images, vi_marks, ir_marks
