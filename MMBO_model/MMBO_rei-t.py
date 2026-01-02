import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from EquiMDConv import EquiMDConv
from transformers import GPT2Tokenizer, GPT2Model
import torch.nn.functional as F
from longclip_model import longclip
from transformers import BertTokenizer, BertModel
from torchvision import models
# torch.backends.cudnn.enabled = False
device='cuda'

class longclip_text_encoder(nn.Module):  
    def __init__(self,device,output_channel=2048):
        super().__init__()
        self.model, preprocess = longclip.load("/mnt/10T/zjy/D_OIQA/longclip_model/longclip-B.pt", device=device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.conv1 = nn.Conv2d(in_channels=5120, out_channels=2048, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=4096, out_channels=2048, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu=nn.ReLU()
        self.textlinear=nn.Linear(512,2048)
    
    def forward_vector(self, x):
        B,N, C, H, W = x.shape
        feature_l_x = torch.tensor([]).to(x.device)
        for i in range(N):
            input_x = x[:,i,:,:,:]
            l_x=self.model.encode_image(input_x)
            feature_l_x=torch.cat((l_x,feature_l_x),dim=1)
        return feature_l_x


    def compute_attention_weights(self,query:any,key:any):

        key=key.transpose(0,1)
        sim = torch.matmul(query,key)  # (batch_size, query_len, key_len)
        attention_weights = F.softmax(sim, dim=-1)  # (batch_size, query_len, key_len)
        return attention_weights

    def apply_attention_weights(self,values, attention_weights):
        # values: (batch_size, value_len, feature_dim)
        # attention_weights: (batch_size, query_len, key_len)

        weighted_values = torch.matmul(attention_weights, values)  # (batch_size, query_len, feature_dim)
        return weighted_values

    def forward(self,patch,text):
        texts=longclip.tokenize(text)
        texts=texts.to(device)
        text_features=self.model1.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_feature=text_features.to(dtype=torch.float)
        text_feature=self.textlinear(text_feature)
        return text_feature

class EDCBlock(nn.Module):
    def __init__(self,input_channel,output_channel,layer,stride=2):
        super(EDCBlock,self).__init__()
        self.conv1 = EquiMDConv(input_channel,output_channel,layer)
        self.use_1x1 = nn.Conv2d(input_channel,output_channel,kernel_size=1,stride = stride)
        self.use_1x1_2=nn.Conv2d(2*output_channel,output_channel,kernel_size=1,stride=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3,padding=1,stride=stride)
        self.sigmoid  = nn.Sigmoid()
    def forward(self,x,position):
        out_line = self.use_1x1(x)
        x = self.conv1(x,position)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        out=torch.cat((out_line,x),dim=1)
        out=self.use_1x1_2(out)
        out = self.sigmoid(out)
        return out


class DAM(nn.Module):
    def __init__(self):
        super(DAM,self).__init__()
        self.layer1_block1 = EDCBlock(3,16,0)
        self.layer1_block2 = EDCBlock(16,32,0,1)
        #layer2
        self.layer2_block1 = EDCBlock(32,64,1)
        self.layer2_block2 = EDCBlock(64,128,1,1)
        #layer3
        self.layer3_block1 = EDCBlock(128,256,2)
        self.layer3_block2 = EDCBlock(256,512,2,1)
 
    def forward(self,x,position):
        out = self.layer1_block1(x,position)
        out = self.layer1_block2(out,position)
        out = self.layer2_block1(out,position)
        out = self.layer2_block2(out,position)
        out = self.layer3_block1(out,position)
        out = self.layer3_block2(out,position)
        return out

class MMBO(nn.Module):
    def __init__(self,num_classes=1):
        super(MMBO, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.QAM = nn.Sequential(*list(resnet50.children())[:-1])
        self.DAM = DAM()
        self.Avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_channels=5120, out_channels=2048, kernel_size=1, stride=1, padding=0)
        self.textlinear = nn.Linear(512,2048)
        self.textencode = longclip_text_encoder(device=device)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2048,512)
        self.fc1 = nn.Linear(4096,2048)
        self.classifier = self.regressor = nn.Sequential(
            nn.Linear(4096,2048), 
            nn.ReLU(),
            nn.Linear(2048, num_classes) 
        )#img:2048+text:512=2560
    def compute_attention_weights(self,query:any,key:any):
        # query: (batch_size, query_len, feature_dim)
        # key: (batch_size, key_len, feature_dim)
        query=query.transpose(0,1)
        key=key.float()
        sim = torch.matmul(query,key)  # (batch_size, query_len, key_len)
        attention_weights = F.softmax(sim, dim=-1)  # (batch_size, query_len, key_len)
        return attention_weights
    
    def apply_attention_weights(self,values, attention_weights):
        # values: (batch_size, value_len, feature_dim)
        # attention_weights: (batch_size, query_len, key_len)
        weighted_values = torch.matmul(values,attention_weights)  # (batch_size, query_len, feature_dim)
        return weighted_values
    def forward_vector(self, x,position):
        B,N, C, H, W = x.shape
        feature_dis = torch.tensor([]).to(x.device)
        feature_dis1 = torch.tensor([]).to(x.device)
        for i in range(N):
            input_x = x[:,i,:,:,:]
            input_position = position[:,i,:]
            input_position = input_position[0]
            DAM_feature = self.DAM(input_x,input_position) 
            QAM_feature = self.QAM(input_x) 
            QAM_feature=QAM_feature.view(QAM_feature.size(0),-1)
            QAM_feature=self.relu(self.fc(QAM_feature))
            
            feature_dis = torch.cat((DAM_feature, feature_dis),dim=1)
            feature_dis1 = torch.cat((QAM_feature, feature_dis1),dim=1)
        feature_dis1 = feature_dis1.unsqueeze(-1).unsqueeze(-1)


        return feature_dis,feature_dis1
    
    def forward(self, patch,position,text):

        text_feature = self.textencode(patch,text)
        DAM_feature,QAM_feature = self.forward_vector(patch,position)
        DAM_feature= self.conv1(DAM_feature)
        DAM_feature = self.Avgpool2d(DAM_feature)
        QAM_feature = self.conv1(QAM_feature)
        QAM_feature = self.Avgpool2d(QAM_feature)
        DAM_feature = DAM_feature.view(DAM_feature.size(0), -1)
        QAM_feature = QAM_feature.view(QAM_feature.size(0), -1)
        QAM_DAM_features = torch.cat((QAM_feature,DAM_feature),dim=1)
        QAM_DAM_features = self.fc1(QAM_DAM_features)
        att_image_to_text_features=self.compute_attention_weights(QAM_DAM_features,text_feature)
        weight_image_features=self.apply_attention_weights(text_feature,att_image_to_text_features)+text_feature
        QAM_DAM_text_features=torch.cat((QAM_DAM_features,weight_image_features),dim=1)
        score = self.classifier(QAM_DAM_text_features)
        return score
