import torch.nn as nn
import torchvision.models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torchvision.utils
import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm.notebook import tqdm
import os
import torchmetrics
import timm
import time
import datetime
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
image_count = 50
img_size = 256
tf = ToTensor()


class CustomDataset(Dataset):
    def __init__(self, image_list, label_list):
        self.img_path = image_list

        self.label = label_list

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image_tensor = torch.empty((image_count, 3, img_size, img_size))

        image_file_list = glob(self.img_path[idx]+'/*.jpg')
        if len(image_file_list) > image_count:

            count = 0
            for index in range(image_count):
                image = 1 - \
                    tf(Image.open(image_file_list[index]).resize(
                        (img_size, img_size)))
                image_tensor[count] = image
                count += 1
        else:
            count = 0
            for index in range(len(image_file_list)):
                image = 1 - \
                    tf(Image.open(image_file_list[index]).resize(
                        (img_size, img_size)))
                image_tensor[count] = image
                count += 1
            for i in range(image_count-count):
                image = 1 - \
                    tf(Image.open(image_file_list[i]).resize(
                        (img_size, img_size)))
                image_tensor[count] = image
                count += 1
        label_tensor = self.label[idx]
        return self.img_path[idx], image_tensor, label_tensor


class FeatureExtractor(nn.Module):
    """Feature extoractor block"""

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        cnn1 = timm.create_model('efficientnet_b2', pretrained=True)
        self.feature_ex = nn.Sequential(*list(cnn1.children())[:-1])

    def forward(self, inputs):
        features = self.feature_ex(inputs)

        return features


class AttentionMILModel(nn.Module):
    def __init__(self, num_classes, image_feature_dim, feature_extractor_scale1: FeatureExtractor):
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
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, inputs):
        batch_size, num_tiles, channels, height, width = inputs.size()

        # Flatten the inputs
        inputs = inputs.view(-1, channels, height, width)

        # Feature extraction using the pre-trained CNN
        # Shape: (batch_size * num_tiles, 2048, 1, 1)
        features = self.feature_extractor(inputs)

        # Reshape features
        # Shape: (batch_size, num_tiles, 2048)
        features = features.view(batch_size, num_tiles, -1)

        # Attention mechanism
        # Shape: (batch_size, num_tiles, 1)
        attention_weights = self.attention(features)
        # Normalize attention weights
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention weights to features
        # Shape: (batch_size, 2048)
        attended_features = torch.sum(features * attention_weights, dim=1)
        attended_features = self.dropout(attended_features)
        attended_features = F.relu(attended_features)
        # Classification layer
        # Shape: (batch_size, num_classes)
        logits = self.classification_layer(attended_features)

        return logits


def predict(file_list, label_data):
    start = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'[Predict Start]')
    print(f'Predict Start Time : {now_time}')
    test_label_list = []
    for i in range(len(file_list)):
        label = label_data.loc[label_data['Serial Number']
                               == int(os.path.basename(file_list[i]))]
        label = label.loc[label_data['wake'] == 1]
        label = label['label'].item()
        if label == 'Oropharynx':
            test_label_list.append(0)
        elif label == 'Tonguebase':
            test_label_list.append(1)
        elif label == 'Epiglottis':
            test_label_list.append(2)

    test_dataset = CustomDataset(file_list, F.one_hot(
        torch.tensor(test_label_list).to(torch.int64)))
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    Feature_Extractor = FeatureExtractor()
    model = AttentionMILModel(3, 1408, Feature_Extractor)
    model = model.to(device)
    accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=3).to(device)
    model.load_state_dict(torch.load('./model/Best_model.pt'))
    total_y = torch.zeros((len(test_dataloader), 3)).to(device)
    total_prob = torch.zeros((len(test_dataloader), 3)).to(device)
    model.eval()
    count = 0
    val_running_loss = 0.0
    acc_loss = 0
    test = tqdm(test_dataloader)
    path_list = []
    with torch.no_grad():
        for path, x, y in test:
            y = y.to(device).float()
            x = x.to(device).float()
            predict = model(x).to(device)
            cost = F.cross_entropy(predict.softmax(dim=1), y)  # cost 구함
            acc = accuracy(predict.softmax(
                dim=1).argmax(dim=1), y.argmax(dim=1))
            val_running_loss += cost.item()
            acc_loss += acc
            prob_pred = predict.softmax(dim=1)
            total_y[count] = y.squeeze(dim=1)
            total_prob[count] = prob_pred
            count += 1
            path_list.append(path)
    test_score = roc_auc_score(total_y.cpu().argmax(
        axis=1), total_prob.cpu(), multi_class='ovr')
    print(f'total AUC score= {test_score}')
    cm = confusion_matrix(total_y.cpu().argmax(
        axis=1), total_prob.cpu().argmax(axis=1))
    print(f'total confusion matrix array : \n {cm}')
    f1 = f1_score(total_y.cpu().argmax(axis=1),
                  total_prob.cpu().argmax(axis=1), average='macro')
    print(f'total f1-score= {f1}')
    end = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'Predict Time : {now_time}s Time taken : {end-start}')
    print(f'[Predict End]')
