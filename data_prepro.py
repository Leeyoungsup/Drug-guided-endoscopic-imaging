
import warnings
import json
import torchvision
import numpy as np
import time
import datetime
import os
import warnings
import pandas as pd
from PIL import Image
from glob import glob
from tqdm.auto import tqdm
import warnings

import torchvision

import matplotlib.pyplot as plt
# 경고 메시지를 무시하도록 설정
warnings.filterwarnings("ignore")
import numpy as np
import time
import datetime
import os
import warnings
import pandas as pd
from PIL import Image
from glob import glob
from tqdm.auto import tqdm
import warnings

import torchvision

import matplotlib.pyplot as plt
# 경고 메시지를 무시하도록 설정
warnings.filterwarnings("ignore")


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def Preprocessing(file_list,dataset_calss, label_data):
    start = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'[Preprocessing Start]')
    print(f'Preprocessing Start Time : {now_time}')
    frame_path = '../../data/mteg_data/frame/'+dataset_calss+'/'    
    for i in tqdm(range(len(file_list))):
       
        count = 0
        vidcap = torchvision.io.read_video(file_list[i])
        fps = int(vidcap[2]['video_fps'])
        video = np.array(vidcap[0], dtype=np.uint8)
        video_crop = np.zeros(
            (len(video)-1, video.shape[1], video.shape[2], 3))
        for j in range(len(video_crop)):
            video_crop[j] = video[j+1]-video[j]
        video_crop = video_crop.sum(axis=0)
        video_crop = video_crop.sum(axis=2)
        video_crop = ((video_crop/video_crop.max())*255).astype(np.uint8)
        y1 = np.where(video_crop > 100)[0].min()
        y2 = np.where(video_crop > 100)[0].max()
        x1 = np.where(video_crop > 100)[1].min()
        x2 = np.where(video_crop > 100)[1].max()
        video_name = os.path.basename(file_list[i])
        dst_label = label_data.loc[label_data["File Name"] == video_name]
        wake = str(dst_label['구분값'].item())
        Serial_Number = str(dst_label['일련번호'].item())
        file_name = Serial_Number+wake
        createDirectory(frame_path+file_name)
        for k in range(0, len(video), fps//5):
            img = Image.fromarray(video[k, y1:y2, x1:x2])
            im_new = expand2square(img, (0, 0, 0))
            im_new.resize((256, 256)).save(
                frame_path+file_name+"/%06d.jpg" % count)
            count += 1
    
    end = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'Preprocessing Time : {now_time}s Time taken : {end-start}')
    print(f'[Preprocessing End]')
label_data = pd.DataFrame(columns=['구분값', '일련번호', 'OTE 원인','File Name'])    
train_label_data = pd.DataFrame(columns=['구분값', '일련번호', 'OTE 원인','File Name'])
test_label_data = pd.DataFrame(columns=['구분값', '일련번호', 'OTE 원인','File Name'])
val_label_data = pd.DataFrame(columns=['구분값', '일련번호', 'OTE 원인','File Name'])
classes = ['Oropharynx_posterior_lateral_walls', 'Tongue_Base', 'Epiglottis']
label_list=glob('../../../../YS_Baik/5.NIA_42/02_2_classification_gachon/01_data/train/*.json')
for i in range(len(label_list)):
    with open(label_list[i], 'r') as f:
        json_data = json.load(f)
        f.close()
    separation_value= json_data['videos']['id'][json_data['videos']['id'].find('_')-1:json_data['videos']['id'].find('_')]
    serial_number=json_data['videos']['id'][:json_data['videos']['id'].find('_')-1]
    causes_of_OTE=json_data['videos']['id']
    label_data.loc[i]=[int(separation_value),int(serial_number),classes.index(json_data['metas']['cause'])+1,json_data['videos']['filename']]
    train_label_data.loc[i]=[int(separation_value),int(serial_number),classes.index(json_data['metas']['cause'])+1,json_data['videos']['filename']]
train_label_data.to_csv('../../data/mteg_data/train_label.csv',index=False)
label_list=glob('../../../../YS_Baik/5.NIA_42/02_2_classification_gachon/01_data/test/*.json')
for i in range(len(label_list)):
    with open(label_list[i], 'r') as f:
        json_data = json.load(f)
        f.close()
    separation_value= json_data['videos']['id'][json_data['videos']['id'].find('_')-1:json_data['videos']['id'].find('_')]
    serial_number=json_data['videos']['id'][:json_data['videos']['id'].find('_')-1]
    causes_of_OTE=json_data['videos']['id']
    label_data.loc[i]=[int(separation_value),int(serial_number),classes.index(json_data['metas']['cause'])+1,json_data['videos']['filename']]
    test_label_data.loc[i]=[int(separation_value),int(serial_number),classes.index(json_data['metas']['cause'])+1,json_data['videos']['filename']]
test_label_data.to_csv('../../data/mteg_data/test_label.csv',index=False)
label_list=glob('../../../../YS_Baik/5.NIA_42/02_2_classification_gachon/01_data/val/*.json')
for i in range(len(label_list)):
    with open(label_list[i], 'r') as f:
        json_data = json.load(f)
        f.close()
    separation_value= json_data['videos']['id'][json_data['videos']['id'].find('_')-1:json_data['videos']['id'].find('_')]
    serial_number=json_data['videos']['id'][:json_data['videos']['id'].find('_')-1]
    causes_of_OTE=json_data['videos']['id']
    label_data.loc[i]=[int(separation_value),int(serial_number),classes.index(json_data['metas']['cause'])+1,json_data['videos']['filename']]
    val_label_data.loc[i]=[int(separation_value),int(serial_number),classes.index(json_data['metas']['cause'])+1,json_data['videos']['filename']]
val_label_data.to_csv('../../data/mteg_data/val_label.csv',index=False)
label_data.to_csv('../../data/mteg_data/label.csv',index=False)
file_list = glob('../../../../YS_Baik/5.NIA_42/02_2_classification_gachon/01_data/train/*.mp4')
Preprocessing(file_list,'train', train_label_data)
file_list = glob('../../../../YS_Baik/5.NIA_42/02_2_classification_gachon/01_data/test/*.mp4')
Preprocessing(file_list,'test', test_label_data)
file_list = glob('../../../../YS_Baik/5.NIA_42/02_2_classification_gachon/01_data/val/*.mp4')
Preprocessing(file_list,'val', val_label_data)