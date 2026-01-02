import json
import cv2
from PIL import Image
from longclip_model import longclip
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import os
import numpy as np
import random
data4v_root = 'sharegpt4v/data/'
json_name = 'share-captioner_coco_lcs_sam_1246k_1107.json'
image_root = 'sharegpt4v/data/'
device='cuda'
class share4v_val_dataset(data.Dataset):
    def __init__(self):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[:self.total_len]
        _ , self.preprocess = longclip.load("ViT-L/14")
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption


class share4v_train_dataset(data.Dataset):
    def __init__(self):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[self.total_len:]
        _ , self.preprocess = longclip.load("ViT-L/14")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        

        caption_short = caption.split(". ")[0]
        
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption, caption_short


class DE_t_Dataset(Dataset):
    def __init__(self, cfg):
        _ , self.preprocess = longclip.load("/mnt/10T/zjy/D_OIQA/longclip_model/longclip-B.pt",device=device)
        self.root_dir = cfg.root_dir
        # self.transform = transform
        # self.transform1 = transform1
        self.csv_path = cfg.csv_path
        self.data = pd.read_csv(cfg.csv_path)
        column_names =   ['name','re', 'score','mos','mean','text']
        # column_names =   ['ref','name', 'mos','text']
        # column_names =   ['name','mos', 'text']
        self.df = pd.read_csv(cfg.csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        # name = split_dataset_name(cfg.csv_path)
        self.image_name = self.df['name']
        self.mos=self.df['mos']
        self.text=self.df['text']
        # self.short_text=self.df['short_text']
        
    def __len__(self):
        return len(self.image_name)

    def __getitem__(self,index):
        dis_image_basename = os.path.splitext(self.image_name[index])[0]
        image_path = os.path.join(self.root_dir, self.image_name[index])
        # patch_path = f'/mnt/10T/lzy/test/patch/data896/dis/{dis_image_basename}_dis_patches.npy'
        # position_path = f'/mnt/10T/lzy/test/patch/data896/position/{dis_image_basename}_positions.npy'
        patch_path = f'/mnt/10T/zjy/database/OIQ_10k/data896/num10/{dis_image_basename}_dis_patches.npy'
        position_path = f'/mnt/10T/zjy/database/OIQ_10k/data896/position10/{dis_image_basename}_positions.npy'
        
        # patch_path = f'/mnt/10T/lzy/Test-FR/equatorial_5/dis/{dis_image_basename}_dis_patches.npy'
        # patch_path = f'/mnt/10T/wkc/Database/OIQ_10K_patch/{dis_image_basename}_dis_patches.npy'
        # position_path = f'/mnt/10T/lzy/Test-FR/equatorial_5/position/{dis_image_basename}_positions.npy'
        

        dis_patchs = np.load(patch_path)
        
        # image_path = os.path.join(self.root_dir, self.image_name[index])
        
        # image = Image.open(image_path).convert('RGB')

        # for idx,row in self.data.iterrows():
        #     if row[0] == self.image_name[index]:
        #         label = row[3]
        #         text= row[5]
        label = torch.FloatTensor(np.array(self.mos[index]))
        text=self.text[index]
        # short_text=self.short_text[index]
        # image_tensor = self.preprocess(image)
        # text = self.df[index]
        # if self.transform is not None:
        # image1 = self.transform(image)
        # image2 = self.transform1(image)
        # 假设标签信息存储在一个csv文件中，根据图像名称获取对应的标签


       # label = self.get_label_from_csv(self.image_files[0])
        return dis_patchs,text

    def get_label_from_csv(self, image_file):
        # 从csv文件中获取图像对应的标签信息
        # 根据自己的需求进行实现
        pass