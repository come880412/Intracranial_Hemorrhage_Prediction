"""
Reference : https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/blob/master/2DNet/src/prepare_data.py
"""

import pydicom
import numpy as np
import os
import cv2
import warnings
from train_val_split import train_val_split
import cv2
from tqdm import tqdm
from glob import glob
import argparse
warnings.filterwarnings('ignore')

label_name_list = {'intraventricular':0, 'subarachnoid':1, 'subdural':2, 'intraparenchymal':3, 'epidural':4}

def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    return int(x)

def get_id(img_dicom):
    return str(img_dicom.SOPInstanceUID)

def get_metadata_from_dicom(img_dicom):
    metadata = {
        "intercept": img_dicom.RescaleIntercept,
        "slope": img_dicom.RescaleSlope,
    }
    return {k: get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}

def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image

def window_image(img, intercept, slope):
    img = img * slope + intercept # get_hu_pixels
    image1 = apply_window(img, 40, 80) # brain
    image2 = apply_window(img, 80, 200) # subdural
    image3 = apply_window(img, 40, 380) # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)

    return image

def save_img(image, subfolder, name):
    cv2.imwrite(subfolder+name+'.png', image)

def prepare_image(img_path):
    img_dicom = pydicom.read_file(img_path)
    img_id = get_id(img_dicom)
    study_instance_uid = img_dicom.StudyInstanceUID
    metadata = get_metadata_from_dicom(img_dicom)
    image = window_image(img_dicom.pixel_array, **metadata)
    image -= image.min((0,1))
    image = (255*image).astype(np.uint8)
    return img_id, study_instance_uid, image

def prepare_and_save(img_path, subfolder, folder, file_name):
    if folder == 'train1':
        img_id, study_instance_uid, combined_img = prepare_image(img_path)
        save_img(combined_img, subfolder, img_id)
        return img_id, study_instance_uid
    elif folder == 'train2' or folder == 'test':
        _, _, combined_img = prepare_image(img_path)
        save_img(combined_img, subfolder, file_name[:-4])

def prepare_images(data_path, dcm_path, subfolder, train_val_ratio, folder):
    if folder == 'train1':
        train_csv_file = np.loadtxt(os.path.join(data_path, 'Public_train_label.csv'), delimiter=',', dtype=np.str)[1:]
        file_dict = {}
        for i in range(0, len(train_csv_file), 5):
            label = np.zeros(len(label_name_list), dtype=str)
            file_list = train_csv_file[i:i+5]
            subtype_list = file_list[:,1]
            for idx,subtype in enumerate(subtype_list):
                label[label_name_list[subtype]] = file_list[idx,2]
            dcm_name = file_list[0][0]   
            if dcm_name not in file_dict.keys():
                file_dict[dcm_name] = label
        
        dcm_list = os.listdir(dcm_path)
        save_csv = [['dcm_file_name', 'filename', 'study_instance_uid', 'intraventricular', 'subarachnoid', 'subdural', 'intraparenchymal', 'epidural']]
        for dcm_name in tqdm(dcm_list):
            label = file_dict[dcm_name]
            dcm_image_path = os.path.join(dcm_path, dcm_name)
            ID, study_instance_uid = prepare_and_save(dcm_image_path, subfolder, folder, dcm_name)
            filename = ID + '.png'
            save_csv.append([dcm_name, filename, study_instance_uid, label[0], label[1], label[2], label[3], label[4]])
        
        # Train_val_split
        save_path = os.path.join(data_path, folder+'_png.csv')
        np.savetxt(save_path,  save_csv, fmt='%s', delimiter=',')
        nosick_ratio = 1.0  # Ratio of no sick images
        random_split = True # True:random split  False: split by ID
        train_save_name = os.path.join(data_path, 'train.csv')
        val_save_name = os.path.join(data_path, 'val.csv')
        train_val_split(save_path, train_val_ratio, nosick_ratio, train_save_name, val_save_name, random_split)

    elif folder == 'train2' or folder == 'test':
        dcm_list = os.listdir(dcm_path)
        for dcm_name in tqdm(dcm_list):
            dcm_image_path = os.path.join(dcm_path, dcm_name)
            prepare_and_save(dcm_image_path, subfolder, folder, dcm_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-data_path", default='../dataset', type=str)
    parser.add_argument("-folder", default='test', type=str, help='train1/train2/test')
    args = parser.parse_args()
    dcm_path = os.path.join(args.data_path, args.folder)
    png_path = os.path.join(args.data_path, args.folder + '_png')

    np.random.seed(500)
    train_val_ratio = 0.8

    if not os.path.exists(png_path):
        os.makedirs(png_path)

    prepare_images(args.data_path, dcm_path, png_path+'/', train_val_ratio, args.folder)