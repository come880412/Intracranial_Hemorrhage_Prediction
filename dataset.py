"""
Reference: https://github.com/darraghdog/rsna/blob/master/scripts/trainorig.py
"""

import cv2
from albumentations.pytorch import ToTensorV2
import os
from albumentations import (Compose, Normalize, HorizontalFlip,ShiftScaleRotate, Transpose)
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tqdm
cv2.setNumThreads(0)

label_name_dict = {'intraventricular':0, 'subarachnoid':1, 'subdural':2, 'intraparenchymal':3, 'epidural':4,
                  0:'intraventricular', 1:'subarachnoid', 2:'subdural', 3:'intraparenchymal', 4:'epidural'}

# Data loaders
mean_img = [0.485, 0.456, 0.406]
std_img = [0.229, 0.224, 0.225]
transform_train = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                         rotate_limit=20, p=0.3, border_mode = cv2.BORDER_REPLICATE),
    Transpose(p=0.5),
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensorV2()
])

transform_test= Compose([
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensorV2()
])

def autocrop(image, threshold=0):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    cols = np.where(np.max(flatImage, 1) > threshold)[0]
    image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    #logger.info(image.shape)
    sqside = max(image.shape)
    imageout = np.zeros((sqside, sqside, 3), dtype = 'uint8')
    imageout[:image.shape[0], :image.shape[1],:] = image.copy()
    return imageout


class DCM_data(Dataset):
    def __init__(self, data_root, sub_folder, csv_root, mode, in_channel, image_size, semi=0):
        self.in_channel = in_channel
        self.mode = mode
        self.image_size = image_size
        self.file = []
        if self.mode == 'train' or self.mode == 'val' or self.mode == 'visualization':
            label_count = {'0':0, '1':0, '2':0, '3':0, '4':0}
            self.label_count = []

            if self.mode == 'train':
                dcm_file_list = np.loadtxt(csv_root, delimiter=',', dtype=np.str)
                self.transform = transform_train

            elif self.mode == 'val' or self.mode == 'visualization':
                dcm_file_list = np.loadtxt(csv_root, delimiter=',', dtype=np.str)
                self.transform = transform_test

            for i in tqdm.tqdm(range(len(dcm_file_list))):
                file_list = dcm_file_list[i]
                file_name = file_list[0]
                dcm_file_path = os.path.join(data_root, sub_folder, file_name)
                label = file_list[1:]
                for i in range(len(label)):
                    label_sub = int(label[i])
                    if label_sub == 1:
                        label_count[str(i)] += 1
                self.file.append([dcm_file_path, label])
            # Use semi-superviesd learning(th=0.5)
            if semi:
                print('use unlabeled data!')
                semi_file_list = np.loadtxt(os.path.join(data_root, 'train2_png.csv'), delimiter=',', dtype=np.str)[1:]
                for i in tqdm.tqdm(semi_file_list):
                    file_name = i[0]
                    file_path = os.path.join(data_root, 'train2_png', file_name)
                    logits = i[1:].astype(float)
                    logits[logits >= 0.5] = 1
                    logits[logits < 0.5] = 0
                    logits = logits.astype(int)
                    label = np.any(logits)
                    if label:
                        self.file.append([file_path, logits])

            # print('------label_count------')
            for _, values in label_count.items():
                num_label = values
                self.label_count.append(num_label)
            
        elif self.mode == 'test':
            self.file_name = []
            test_csv_file = np.loadtxt('./sample_submission.csv', delimiter=',', dtype=np.str)[1:]
            test_csv_file = test_csv_file[:100]
            self.transform = transform_test
            for i in tqdm.tqdm(range(0, len(test_csv_file), 5)):
                file_list = test_csv_file[i:i+5, :]
                file_name_list = list(file_list[:, 0])
                file_name = file_list[0,0].split('.')[0]
                
                file_path = os.path.join(data_root, sub_folder, file_name + '.png')
                self.file_name.append(file_name_list)
                self.file.append(file_path)
        
        elif self.mode == 'semi': # Generate psuedo-label
            self.file_name = []
            folder_path = os.path.join(data_root, sub_folder)
            img_name_list = os.listdir(folder_path)
            img_name_list.sort()
            self.transform = transform_test
            for img_name in img_name_list:
                img_path = os.path.join(folder_path, img_name)
                self.file.append(img_path)
                self.file_name.append(img_name)

    def __getitem__(self, index):  
        if self.mode == 'train' or self.mode == 'val':
            file_path, label = self.file[index]
            label = np.array(label).astype(int)
            image = cv2.imread(file_path)
            # cv2.imwrite('./image_beforecroped.png', image)
            try:
                image = autocrop(image, threshold=0)
            except:
                1
            image = cv2.resize(image,(self.image_size,self.image_size))
            # cv2.imwrite('./image_aftercroped.png', image)
            augmented = self.transform(image= image)
            image = augmented['image']
            return image, label
        elif self.mode == 'test':
            file_path = self.file[index]
            file_name = list(self.file_name[index])
            image = cv2.imread(file_path)
            try:
                image = autocrop(image, threshold=0)
            except:
                1
            image = cv2.resize(image,(self.image_size,self.image_size))
            augmented = self.transform(image= image)
            image = augmented['image']
            
            return image, file_name
        
        elif self.mode == 'semi':
            file_path = self.file[index]
            file_name = self.file_name[index]
            image = cv2.imread(file_path)
            try:
                image = autocrop(image, threshold=0)
            except:
                1
            image = cv2.resize(image,(self.image_size,self.image_size))
            augmented = self.transform(image= image)
            image = augmented['image']
            
            return image, file_name
        
        elif self.mode == 'visualization':
            file_path, label = self.file[index]
            label = np.array(label).astype(int)
            image = cv2.imread(file_path)
            try:
                image = autocrop(image, threshold=0)
            except:
                1
            image = cv2.resize(image,(self.image_size,self.image_size))
            augmented = self.transform(image= image)
            image = augmented['image']
            return image, label, file_path
    def __len__(self):
        return len(self.file)

if __name__ == '__main__':
    root = '../dataset'
    data = DCM_data(root, 'train1_png', 'train', in_channel=3)
    data_loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=0)
    for batch,(image, label) in tqdm.tqdm(enumerate(data_loader)):
        print(image.shape)
    # iter_data = iter(data_loader)
    # image, label = iter_data.next()
    # print(image.shape, label)