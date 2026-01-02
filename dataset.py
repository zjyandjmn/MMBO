import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import csv
Image.MAX_IMAGE_PIXELS=None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def split_dataset_name(csv_pach):
    name=[]
    with open('/home/d310/10t/zjy/D_OIQA/database_csv/GPT_4o_OIQ_10k_train.csv', 'r') as file:  # Replace 'file.csv' with the actual file path
         data = csv.reader(file)
        #  next(data)
         for row in data:
            name.append(row[0])
    return name


class DEDataset(Dataset):
    def __init__(self, cfg ,transform):
        self.root_dir = cfg.root_dir
        self.transform = transform
        self.csv_path = cfg.csv_path
        self.data = pd.read_csv(cfg.csv_path,skiprows=1)

        name = split_dataset_name(cfg.csv_path)
        self.image_name = name
        
    def __len__(self):
        return len(self.image_name)

    def __getitem__(self,index):
        image_path = os.path.join(self.root_dir, self.image_name[index])
        
        image = Image.open(image_path).convert('RGB')

        for idx,row in self.data.iterrows():
            if row[1] == self.image_name[index]:
                label = row[2]

        if self.transform is not None:
            image = self.transform(image)

        # 假设标签信息存储在一个csv文件中，根据图像名称获取对应的标签


       # label = self.get_label_from_csv(self.image_files[0])
        return image,label

    def get_label_from_csv(self, image_file):
        # 从csv文件中获取图像对应的标签信息
        # 根据自己的需求进行实现
        pass
# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     from config import DE_360IQA_config
#     dataset=DEDataset# 创建数据集实例
#     train_dataset = dataset(
#         cfg=DE_360IQA_config,
#         transform = transforms.Compose([
#             transforms.Resize((512, 512)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]))
#     test_dataset = dataset(
#         cfg=DE_360IQA_config,
#         transform = transforms.Compose([
#             transforms.Resize((512, 512)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]))
#     logging.info('number of train scenes: {}'.format(len(train_dataset)))
#     logging.info('number of val scenes: {}'.format(len(test_dataset)))
    
#     train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,drop_last=True, num_workers=cfg.num_workers,shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, drop_last=True,num_workers=cfg.num_workers,shuffle=False)
        

class DE_t_Dataset(Dataset):
    def __init__(self, cfg ,transform,transform1,csv_path):
        self.root_dir = cfg.root_dir
        self.transform = transform
        self.transform1 = transform1
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        column_names =   ['name','re', 'score','mos','mean','text']
        # column_names =   ['ref','name', 'mos','text']
        self.df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        # name = split_dataset_name(cfg.csv_path)
        self.image_name = self.df['name']
        self.mos=self.df['mos']
        self.text=self.df['text']
        
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
        position = np.load(position_path)
        image = Image.open(image_path).convert('RGB')

        label = torch.FloatTensor(np.array(self.mos[index]))
        text=self.text[index]
        # text = self.df[index]
        # if self.transform is not None:
        image1 = self.transform(image)
        image2 = self.transform1(image)
        # 假设标签信息存储在一个csv文件中，根据图像名称获取对应的标签


       # label = self.get_label_from_csv(self.image_files[0])
        return image2,dis_patchs,position,label,text

    def get_label_from_csv(self, image_file):
        # 从csv文件中获取图像对应的标签信息
        # 根据自己的需求进行实现
        pass
    

class DE_t_vp_Dataset(Dataset):
    def __init__(self, cfg ,transform,transform1,csv_path):
        self.root_dir = cfg.root_dir
        self.cfg=cfg
        self.transform = transform
        self.transform1 = transform1
        self.csv_path = csv_path
        idx_list = [str(i) for i in range(cfg.num_vps)]
        column_names = idx_list + ['d','mos','text']
        # column_names = idx_list + ['mos','text']
        self.df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.X = self.df[idx_list]
        # self.d = self.df['d']
        self.text=self.df['text']
        self.mos = self.df['mos']
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self,index): 
        img_list = []
        for i in range(self.cfg.num_vps):
            p1, p2 = self.X.iloc[index, i].split("/")
            path = os.path.join(self.cfg.vp_path, p1, p2)
            img = Image.open(path)
        
            if self.transform:
                image1 = self.transform(img)
                image2 = self.transform1(img)
            image2 = image2.float().unsqueeze(0)
            
            img_list.append(image2)
        text=self.text[index]
        imgs1 = torch.cat(img_list)
        # d = torch.asarray(self.d[index], dtype=float) - 1
        mos = torch.FloatTensor(np.array(self.mos[index]))
        
        return image2,imgs1,image1, mos,text

    def get_label_from_csv(self, image_file):
        # 从csv文件中获取图像对应的标签信息
        # 根据自己的需求进行实现
        pass
    
    
class DE_t_re_Dataset(Dataset):
    def __init__(self, cfg ,transform,transform1,csv_path):
        self.root_dir = cfg.root_dir
        self.transform = transform
        self.transform1 = transform1
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        column_names =   ['name','re', 'score','mos','mean','text']
        # column_names =   ['ref','name', 'mos','text']
        self.df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        # name = split_dataset_name(cfg.csv_path)
        self.image_name = self.df['name']
        self.mos=self.df['mos']
        self.text=self.df['text']
        
    def __len__(self):
        return len(self.image_name)

    def __getitem__(self,index):
        dis_image_basename = os.path.splitext(self.image_name[index])[0]
        image_path = os.path.join(self.root_dir, self.image_name[index])
        # patch_path = f'/mnt/10T/lzy/test/patch/data896/dis/{dis_image_basename}_dis_patches.npy'
        # patch_path = f'/mnt/10T/zjy/database/JUFE_10k/data512/num10/{dis_image_basename}_dis_patches.npy'
        # position_path = f'/mnt/10T/lzy/test/patch/data896/position/{dis_image_basename}_positions.npy'
        # position_path = f'/mnt/10T/zjy/database/JUFE_10k/data512/position10/{dis_image_basename}_positions.npy'
        
        # patch_path = f'/mnt/10T/lzy/Test-FR/equatorial_5/dis/{dis_image_basename}_dis_patches.npy'
        # patch_path = f'/mnt/10T/wkc/Database/OIQ_10K_patch/{dis_image_basename}_dis_patches.npy'
        # position_path = f'/mnt/10T/lzy/Test-FR/equatorial_5/position/{dis_image_basename}_positions.npy'
        

        # dis_patchs = np.load(patch_path)
        # position = np.load(position_path)
        image = Image.open(image_path).convert('RGB')

        label = torch.FloatTensor(np.array(self.mos[index]))
        text=self.text[index]
        # text = self.df[index]
        # if self.transform is not None:
        image1 = self.transform(image)
        image2 = self.transform1(image)
        # 假设标签信息存储在一个csv文件中，根据图像名称获取对应的标签


       # label = self.get_label_from_csv(self.image_files[0])
        return image1,image2,label,text

    def get_label_from_csv(self, image_file):
        # 从csv文件中获取图像对应的标签信息
        # 根据自己的需求进行实现
        pass
    
class DE_t_new_ERP_Dataset(Dataset):
    def __init__(self, cfg ,transform,transform1,csv_path):
        self.root_dir = cfg.root_dir
        self.transform1 = transform
        self.transform2 = transform1
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        # column_names =   ['name','re', 'score','mos','mean','text']
        # column_names =   ['ref','name', 'mos','text1','text2','text3','text4','text5']
        column_names =   ['name','stage', 'score','mos','mean','text1','text2','text3','text4','text5']
        self.df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        # name = split_dataset_name(cfg.csv_path)
        self.image_name = self.df['name']
        self.mos=self.df['mos']
        self.text1=self.df['text1']
        self.text2=self.df['text2']
        self.text3=self.df['text3']
        self.text4=self.df['text4']       
        self.text5=self.df['text5']
        
    def __len__(self):
        return len(self.image_name)

    def __getitem__(self,index):
        dis_image_basename = os.path.splitext(self.image_name[index])[0]
        image_path = os.path.join(self.root_dir, self.image_name[index])
        # patch_path = f'/mnt/10T/lzy/test/patch/data896/dis/{dis_image_basename}_dis_patches.npy'
        # position_path = f'/mnt/10T/lzy/test/patch/data896/position/{dis_image_basename}_positions.npy'
        patch_path = f'/mnt/10T/zjy/database/OIQ_10k/data896/num10/{dis_image_basename}_dis_patches.npy'
        position_path = f'/mnt/10T/zjy/database/OIQ_10k/data896/position10/{dis_image_basename}_positions.npy'

        dis_patchs = np.load(patch_path)
        position = np.load(position_path)
        image = Image.open(image_path).convert('RGB')

        # for idx,row in self.data.iterrows():
        #     if row[0] == self.image_name[index]:
        #         label = row[3]
        #         text= row[5]
        label = torch.FloatTensor(np.array(self.mos[index]))
        text1=self.text1[index]
        text2=self.text2[index]
        text3=self.text3[index]
        text4=self.text4[index]
        text5=self.text5[index]
        
        # text = self.df[index]
        # if self.transform is not None:
        image1 = self.transform1(image)
        image2 = self.transform2(image)
        # 假设标签信息存储在一个csv文件中，根据图像名称获取对应的标签


       # label = self.get_label_from_csv(self.image_files[0])
        return dis_patchs,position,image2,label,text1,text2,text3,text4,text5

    def get_label_from_csv(self, image_file):
        # 从csv文件中获取图像对应的标签信息
        # 根据自己的需求进行实现
        pass