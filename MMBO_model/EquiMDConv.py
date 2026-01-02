import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.ops.deform_conv import DeformConv2d
import math
from torchvision import transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from EquiDMmodel.EquiModel_2DIQA.MyDataset_SPAQ import MyDataset
from thop import profile
from einops import rearrange
import time
import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor.*")
torch.cuda.empty_cache()
#EquiMDConv卷积
# layer_dic = torch.load('/mnt/10T/wkc/fix_offset/layerdic_8k.pt')
# offset_dic = torch.load('/mnt/10T/wkc/fix_offset/offsetdic_8k.pt')
class EquiMDConv(nn.Module):
    def __init__(self,input_channel,out_put_channel,layer):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = out_put_channel
        self.conv = DeformConv2d(in_channels=self.input_channel, out_channels=self.output_channel, kernel_size=3, stride=1, padding=1)
        # 用于生成 offset 的卷积层
        self.conv_offset = nn.Conv2d(self.input_channel, 18, kernel_size=3, stride=1, padding=1)
        init_offset = torch.Tensor(np.zeros([18, self.input_channel, 3, 3]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset)  #初始化为0
        # 用于生成 mask 的卷积层
        self.conv_mask = nn.Conv2d(self.input_channel, 9, kernel_size=3, stride=1, padding=1)
        init_mask = torch.Tensor(np.zeros([9, self.input_channel, 3, 3])+np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask)  #初始化为0.5
        self.layer = layer
 
    def forward(self, x,position):
        #offset_fix = creat_off_set(self.layer,position,x.shape[2],x.shape[3],layer_dic,offset_dic,x.device)
        #learnable offset
        offset = self.conv_offset(x)
        offset = offset #+ offset_fix
        mask = torch.sigmoid(self.conv_mask(x))
        out = self.conv(input=x, offset=offset, mask=mask)
        return out
  
    
def creat_off_set(layer,position=None,patch_u=None,patch_v=None,layer_dic=layer_dic,offset_dic=offset_dic,device=None):
    offset_global =offset_dic[layer_dic[layer]]
    u,v = layer_dic[layer][0]
    if position[0] >= 0.94 or position[1] >= 0.89:
        position[0] = 0.92
        position[1] = 0.83
        
    u_star = int(math.ceil(position[0]*u))
    v_star = int(math.ceil(position[1]*v))
    offset_local=offset_global[:,:,u_star:u_star+patch_u,v_star:v_star+patch_v]
    offset_local = torch.tensor(offset_local).clone().detach().to(device=device)
    return offset_local