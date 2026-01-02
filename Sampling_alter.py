import torch
import numpy as np
from torchvision import transforms as transforms
import math
import time
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
# 假设这是在某处全局定义或作为参数传递
Image.MAX_IMAGE_PIXELS = None
resize = transforms.Resize((512,512))
def normal_distribution(x, mean, sigma):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

def generate_lat_prob():
    x = np.linspace(-90, 90, 181) / 90
    y = {}    
    val = []   

    for i in range(x.shape[0]):
        val.append(normal_distribution(x[i], 0, 0.2))

    val = np.array(val)
    val = softmax(val)

    idx = -90
    for i in range(val.shape[0]):
        y[idx] = val[i]
        idx += 1

    return y

def prior_guided_patch_sampling(img, patch_num=10, p_h=0.2, p_m=0.6, p_l=0.2, seed=3):
    assert p_h + p_m + p_l == 1, "概率之和必须为1。"
    c, h, w = img.shape
    y = generate_lat_prob()  
    lat_prob = list(y.values())   
    cumsum_lat_prob = np.cumsum(lat_prob)   
    # patch_size_v = int(h*0.2)
    # patch_size_u = int(w*0.1)
    patch_size_v = int(896)
    patch_size_u = int(896)

    x_1 = np.searchsorted(cumsum_lat_prob, p_h)
    x_2 = np.searchsorted(cumsum_lat_prob, p_h + p_m)

    h_1 = round(x_1 / 181 * h) 
    h_2 = round(x_2 / 181 * h)   

    patches = torch.empty((patch_num, 3, 512, 512))  
    patch_position_index = np.empty((patch_num, 2))
    
    idx = 0
    np.random.seed(seed)
    # regions = [
    #     (0, h_1, int(patch_num * p_h)),
    #     (h_1, h_2, int(patch_num * p_m)),
    #     (h_2, h, int(patch_num * p_l))
    # ]
    regions = [
        (0, h_1, 2),
        (h_1, h_2, 6),
        (h_2, h, 2)
    ]
    for start_h, end_h, num_patches in regions:
        width_per_patch = w // num_patches
        for i in range(num_patches):
            if end_h - patch_size_v <= start_h:
                raise ValueError("patch_size_v is too large for the given height range.")
            if (i+1) * width_per_patch - patch_size_u <= i * width_per_patch:
                raise ValueError("patch_size_u is too large for the given width range.")
        for i in range(num_patches):
            v = np.random.randint(start_h, end_h - patch_size_v)
            u = np.random.randint(i * width_per_patch, (i+1) * width_per_patch - patch_size_u)
            patch = img[:, v:v+patch_size_v, u:u+patch_size_u]
            patches[idx] = resize(patch)
            patch_position_index[idx] = (v/h, u/w)
            idx += 1
            
    return patches, patch_position_index

if __name__ == '__main__':
   # 设置环境变量以避免库冲突
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        eval_df = pd.read_csv('/mnt/10T/zjy/database/rjl.csv')
        for index, row in eval_df.iterrows():
            image_name = row['name']
            dis_image_basename = os.path.splitext(image_name)[0]
            dis_patch_path = f'/mnt/10T/zjy/database/JUFE_10k/data896resize512/num10/{dis_image_basename}_dis_patches.npy'
            position_path = f'/mnt/10T/zjy/database/JUFE_10k/data896resize512/position10/{dis_image_basename}_positions.npy'
            image_path = os.path.join('/mnt/10T/wkc/Database/JUFE_10k/final_dis_10320', image_name)
            img = Image.open(image_path).convert('RGB')
            dis_img=transform(img)
            dis_patchs, position = prior_guided_patch_sampling(dis_img, patch_num=10, p_h=0.2, p_m=0.6, p_l=0.2, seed=100)
            np.save(dis_patch_path, dis_patchs)
            np.save(position_path, position)