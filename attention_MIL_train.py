
import matplotlib.pyplot as plt
import numpy as np
import helper
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn as nn
import torchvision.models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torchvision.utils
import torch
import pandas as pd
from torchinfo import summary
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split
from copy import copy
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from sklearn.metrics import classification_report
from tqdm import tqdm
import math
from torcheval.metrics import BinaryAccuracy
import os
import torchmetrics
import timm
import time
from sklearn.metrics import f1_score
import datetime
import random
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=4
image_count=50
img_size=256
tf = ToTensor()
class CustomDataset(Dataset):
    def __init__(self, id,image_list, label_list):
        self.img_path = image_list

        self.label = label_list
        self.id=id

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        id_tensor=self.id[idx]
        image_tensor = self.img_path[idx]
    
        label_tensor =  self.label[idx]
        return image_tensor, label_tensor


train_data=pd.read_csv('../../data/mteg_data/internal/1-fold_train.csv') 
file_path='../../data/mteg_data/internal/all/'
train_image_list=[]
for i in range(len(train_data)):
    file_name=train_data.loc[i]['File Name']
    id=file_name[:file_name.find('_')]
    train_image_list.append(file_path+id)
label_data=pd.read_csv('../../data/mteg_data/internal/label.csv')  
train_label_list=[]
train_id_list=[]
train_image_tensor = torch.empty((len(train_image_list),image_count,3, img_size, img_size))
for i in tqdm(range(len(train_image_list))):
    folder_name=os.path.basename(train_image_list[i])
    dst_label=label_data.loc[label_data['일련번호']==int(folder_name[:-1])]
    dst_label=dst_label.loc[dst_label['구분값']==int(folder_name[-1])].reset_index()
    label=int(dst_label.loc[0]['OTE 원인'])
    train_id_list.append(folder_name)
    train_label_list.append(label-1) 
    image_file_list = glob(train_image_list[i]+'/*.jpg')
    if len(image_file_list)>image_count:
        image_index = torch.randint(low=0, high=len(
            image_file_list)-image_count, size=(1,))
        count = 0
        for index in range(image_count):
            image = 1-tf(Image.open(image_file_list[index]).resize((img_size,img_size)))
            train_image_tensor[i,count] = image
            count += 1
    else:
        count = 0
        for index in range(len(image_file_list)):
            image = 1-tf(Image.open(image_file_list[index]).resize((img_size,img_size)))
            train_image_tensor[i,count] = image
            count += 1
        for j in range(image_count-count-1):
            image = 1-tf(Image.open(image_file_list[j]).resize((img_size,img_size)))
            train_image_tensor[i,count] = image
            count += 1
            

test_data=pd.read_csv('../../data/mteg_data/internal/1-fold_test.csv') 
file_path='../../data/mteg_data/internal/all/'
test_image_list=[]
for i in range(len(test_data)):
    file_name=test_data.loc[i]['File Name']
    id=file_name[:file_name.find('_')]
    test_image_list.append(file_path+id)
label_data=pd.read_csv('../../data/mteg_data/internal/label.csv')  
test_label_list=[]
test_id_list=[]
test_image_tensor = torch.empty((len(test_image_list),image_count,3, img_size, img_size))
for i in tqdm(range(len(test_image_list))):
    folder_name=os.path.basename(test_image_list[i])
    dst_label=label_data.loc[label_data['일련번호']==int(folder_name[:-1])]
    dst_label=dst_label.loc[dst_label['구분값']==int(folder_name[-1])].reset_index()
    label=int(dst_label.loc[0]['OTE 원인'])
    test_id_list.append(folder_name)
    test_label_list.append(label-1) 
    image_file_list = glob(test_image_list[i]+'/*.jpg')
    if len(image_file_list)>image_count:
        image_index = torch.randint(low=0, high=len(
            image_file_list)-image_count, size=(1,))
        count = 0
        for index in range(image_count):
            image = 1-tf(Image.open(image_file_list[index]).resize((img_size,img_size)))
            test_image_tensor[i,count] = image
            count += 1
    else:
        count = 0
        for index in range(len(image_file_list)):
            image = 1-tf(Image.open(image_file_list[index]).resize((img_size,img_size)))
            test_image_tensor[i,count] = image
            count += 1
        for j in range(image_count-count):
            image = 1-tf(Image.open(image_file_list[j]).resize((img_size,img_size)))
            test_image_tensor[i,count] = image
            count += 1


train_dataset = CustomDataset(train_id_list,train_image_tensor, F.one_hot(torch.tensor(train_label_list).to(torch.int64)))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = CustomDataset(test_id_list,test_image_tensor, F.one_hot(torch.tensor(test_label_list).to(torch.int64)))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

class FeatureExtractor(nn.Module):
    """Feature extoractor block"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        cnn1= timm.create_model('efficientnet_b2', pretrained=True)
        self.feature_ex = nn.Sequential(*list(cnn1.children())[:-1])

    def forward(self, inputs):
        features = self.feature_ex(inputs)
        
        return features
    
class AttentionMILModel(nn.Module):
    def __init__(self, num_classes, image_feature_dim,feature_extractor_scale1: FeatureExtractor):
        super(AttentionMILModel, self).__init__()
        self.num_classes = num_classes
        self.image_feature_dim = image_feature_dim

        # Remove the classification head of the CNN model
        self.feature_extractor = feature_extractor_scale1
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(image_feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classification layer
        self.classification_layer = nn.Linear(image_feature_dim, num_classes)
        self.dropout=torch.nn.Dropout(0.2)
    def forward(self, inputs):
        batch_size, num_tiles, channels, height, width = inputs.size()
        
        # Flatten the inputs
        inputs = inputs.view(-1, channels, height, width)
        
        # Feature extraction using the pre-trained CNN
        features = self.feature_extractor(inputs)  # Shape: (batch_size * num_tiles, 2048, 1, 1)
        
        # Reshape features
        features = features.view(batch_size, num_tiles, -1)  # Shape: (batch_size, num_tiles, 2048)
        
        # Attention mechanism
        attention_weights = self.attention(features)  # Shape: (batch_size, num_tiles, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize attention weights
        
        # Apply attention weights to features
        attended_features = torch.sum(features * attention_weights, dim=1)  # Shape: (batch_size, 2048)
        attended_features=self.dropout(attended_features)
        attended_features=F.relu(attended_features)
        # Classification layer
        logits = self.classification_layer(attended_features)  # Shape: (batch_size, num_classes)
        
        return logits
    
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
            
import transformers
Feature_Extractor=FeatureExtractor()
model = AttentionMILModel(3,1408,Feature_Extractor)
model = model.to(device)
class_counts = [train_label_list.count(0), train_label_list.count(1), train_label_list.count(2)]
class_weights = [1 - (x / sum(class_counts)) for x in class_counts]
class_weights =  torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=3).to(device)
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, lr=2e-2, momentum=0.9)
optimizer1 = torch.optim.NAdam(model.parameters(), lr=2e-5)
start = time.time()
d = datetime.datetime.now()
now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
print(f'[deeplearning Start]')
print(f'deeplearning Start Time : {now_time}')
MIN_loss=5000
train_loss_list=[]
val_loss_list=[]
train_acc_list=[]
sig=nn.Sigmoid()
val_acc_list=[]
MIN_acc=0
check_val=1000

for epoch in range(300):
    train=tqdm(train_dataloader)
    count=0
    running_loss = 0.0
    acc_loss=0
    model.train()
    for x, y in train:
        y = y.float().to(device)
        count+=1
        x=x.float().to(device)
        enable_running_stats(model)
        predict = model(x).to(device)
        cost = criterion(predict.softmax(dim=1), y.argmax(dim=1)) # cost 구함
        acc=accuracy(predict.softmax(dim=1), y.argmax(dim=1))
        cost.backward() # cost에 대한 backward 구함
        optimizer.first_step(zero_grad=True)
        disable_running_stats(model)
        predict = model(x).to(device)
        cost1 = criterion(predict.softmax(dim=1), y.argmax(dim=1)) # cost 구함
        cost1.backward() # cost에 대한 backward 구함
        optimizer.second_step(zero_grad=True)
        running_loss += cost.item()
        acc_loss+=acc
        train.set_description(f"epoch: {epoch+1}/{300} Step: {count+1} loss : {running_loss/count:.4f} accuracy: {acc_loss/count:.4f}")
    train_loss_list.append((running_loss/count))
    train_acc_list.append((acc_loss/count).cpu().detach().numpy())  
    val=tqdm(test_dataloader)
    path_list=[]
    model.eval()
    val_count=0
    val_running_loss = 0.0
    val_acc_loss=0
    with torch.no_grad():
        total_y = torch.zeros((len(test_dataloader)*batch_size+batch_size, 3)).to(device)
        total_prob = torch.zeros((len(test_dataloader), 3)).to(device)
        for x,y in val:
            y = y.to(device).float()
            val_count+=1
            x=x.to(device).float()
            predict = model(x).to(device)
            cost = criterion(predict.softmax(dim=1), y.argmax(dim=1)) # cost 구함
            acc=accuracy(predict.softmax(dim=1), y.argmax(dim=1))
            total_y[val_count*batch_size:val_count*batch_size+batch_size] = y.squeeze(dim=1)
            total_prob[val_count*batch_size:val_count*batch_size+batch_size] = predict.softmax(dim=1)
            val_running_loss+=cost.item()
            val_acc_loss+=acc
            val.set_description(f"val_epoch: {epoch+1}/{300} Step: {count+1} loss : {val_running_loss/val_count:.4f} accuracy: {val_acc_loss/val_count:.4f}")
        val_loss_list.append((val_running_loss/val_count))
        val_acc_list.append((val_acc_loss/val_count).cpu().detach().numpy())  
        f1 = f1_score(total_y.cpu().argmax(axis=1),total_prob.cpu().argmax(axis=1), average='micro')
        torch.save(model.state_dict(), '../../model/mteg/internal/1-fold/f1_'+str(round(f1,4))+'.pt')
end = time.time()
d = datetime.datetime.now()
now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
print(f'deeplearning Time : {now_time}s Time taken : {start-end}')
print(f'[deeplearning End]')
