import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch
import pathlib
import random

class loader(Dataset):
    def __init__(self, dataset_name, dataset_paths, dataset_type, random_size=None, subject_id='all', img_num=None):
        self.lines        = []
        self.dataset_path = dataset_paths[dataset_name]
        self.type         = dataset_type
        self.dataset_name = dataset_name
        subjects = os.listdir(f'{self.dataset_path}/Image')
        subjects.sort()
        if self.dataset_name == 'Gaze360':
            subjects = [self.type]
            self.gaze_col = 5
            self.headpose_col = 5
        elif dataset_name == 'ETH':
            self.gaze_col = 1
            self.headpose_col = 2
        elif dataset_name == 'MPII':
            if subject_id != 'all':
                subjects = [subject_id]
            self.gaze_col = 7
            self.headpose_col = 8
        elif dataset_name == 'EyeDiap':
            if subject_id != 'all':
                subjects = [subject_id]
            self.gaze_col = 6
            self.headpose_col = 7

        for subject in subjects:
            with open(os.path.join(self.dataset_path, 'Label', subject+'.label'), 'r') as label_file:
                label_content = label_file.readlines()
                label_content.pop(0)
                for label_single_line in label_content:
                    label_single_line = label_single_line.split(' ')
                    for i in range(self.gaze_col, self.headpose_col+1):
                        label_single_line[i] = list(map(float, label_single_line[i].split(',')))
                    self.lines.append(label_single_line)


        if random_size is not None:
            temp = []
            for i in range(random_size):
                a = random.randint(0, len(self.lines)-1)
                temp.append(self.lines[a])

            self.lines = temp


        if img_num is not None:
            self.lines = self.lines[:img_num]
            

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        
        if self.dataset_name != "ETH":
            face_img = cv2.imread(os.path.join(self.dataset_path, 'Image', line[0]))
        else:
            face_img = cv2.imread(os.path.join(self.dataset_path, 'Image/train', line[0]))
      
        label = line[self.gaze_col]
        face_img = face_img / 255.0
        face_img = face_img.transpose(2, 0, 1)
        img =  {"face": torch.from_numpy(face_img).type(torch.FloatTensor),
                "head_pose": torch.tensor(line[self.headpose_col]).type(torch.FloatTensor),
                "name": line[0]}



        return img, torch.tensor(label).type(torch.FloatTensor)


def txtload(dataset_name, dataset_path, dataset_type, batch_size, shuffle=False, num_workers=8, random_size=None,
            subject_id='all', img_num=None):
    print(f"[{dataset_name} Dataset] [{dataset_type} Set]: Loading......")
    dataset = loader(dataset_name, dataset_path, dataset_type, random_size, subject_id, img_num)
    print(f"[{dataset_name} Dataset] [{dataset_type} Set]: Data Loaded! Image Num: {len(dataset)}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load
