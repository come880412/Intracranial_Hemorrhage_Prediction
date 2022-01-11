import os
import tqdm
from dataset import DCM_data
from torch.utils.data import DataLoader
import argparse
from model import resnext50_32x4d, Ensemble_net, SEresnet50
import torch
import numpy as np
import random
from dcm_to_png import prepare_images
from utils import grad_cam
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image

label_name_list = {'intraventricular':0, 'subarachnoid':1, 'subdural':2, 'intraparenchymal':3, 'epidural':4,
                   0:'intraventricular', 1:'subarachnoid', 2:'subdural', 3:'intraparenchymal', 4:'epidural'}

def visualize_layer(opt, model, test_loader):
    model.eval()
    for image, label, file_path in tqdm.tqdm(test_loader):
        image = image.cuda()
        pred = model(image) #(B, num_classes)
        pred_label = pred.cpu().detach().numpy()

        grayscale_cam = grad_cam(image, model)
        for i, gray_cam in enumerate(grayscale_cam):
            if label[i][4] == 1:
                file_path_ = file_path[i]
                image_bgr = cv2.imread(file_path_)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_rgb = (image_rgb / 255.0).astype(np.float32)
                visualization = show_cam_on_image(image_rgb, gray_cam, use_rgb=True)
                plt.imshow(visualization)
                print(pred_label[i], label[i])
                plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model", type=str, default='./model_epoch23_f10.85_threshold0.50.pth', help="save path of model")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--image_size", type=int, default=512, help="size of the batches")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold for multi-label classification")
    """base_options"""
    parser.add_argument("--data_path", type=str, default='../dataset', help="path to dataset")
    parser.add_argument("--sub_folder", type=str, default='train1', help="path to sub_folder(train1/test/train2)")
    parser.add_argument("--test_csv_root", type=str, default='val.csv', help="name of csv")
    parser.add_argument("--train_val_ratio", type=float, default=0.8, help="train/val split ratio")
    parser.add_argument("--ensemble", type=int, default=1, help="ensemble net(1/0)")
    parser.add_argument("--model", type=str, default='resnext50', help="which model you want to use(resnext50/SEresnet50)")
    parser.add_argument("--test_mode", type=str, default='val', help="test/val/semi/visualize")
    parser.add_argument("--num_classes", type=int, default=5, help="number of classes")
    parser.add_argument("--in_channel", type=int, default=3, help="the number of input channels")
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id")
    opt = parser.parse_args()
    dcm_path = os.path.join(opt.data_path, opt.sub_folder)
    png_path = os.path.join(opt.data_path, opt.sub_folder + '_png')
    opt.test_csv_root = os.path.join(opt.data_path, opt.test_csv_root)

    os.makedirs(png_path, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    random_seed_general = 500
    random.seed(random_seed_general) 
    torch.manual_seed(random_seed_general)
    torch.cuda.manual_seed_all(random_seed_general)
    np.random.seed(random_seed_general)
    random.seed(random_seed_general)
    torch.backends.cudnn.deterministic = True
    # prepare_images(opt.data_path, dcm_path, png_path+'/', opt.train_val_ratio, opt.sub_folder)
    
    if opt.model == 'resnext50':
        model = resnext50_32x4d(opt).cuda()
    elif opt.model == 'SEresnet50':
        model = SEresnet50(opt)
    
    if opt.ensemble:
        print('Use ensemble model!')
        model_list = []
        model_name_list = [
                           './model_resnext50.pth',
                           './model_SEresnet50.pth']
        for idx, model_name in enumerate(model_name_list):
            if idx == 0 :
                model = resnext50_32x4d(opt).cuda()
            elif idx ==1:
                model = SEresnet50(opt).cuda()
            model.load_state_dict(torch.load(model_name))
            model_list.append(model)
        model = Ensemble_net(model_list).cuda()
    else:
        model.load_state_dict(torch.load(opt.saved_model))
    

    val_data = DCM_data(opt.data_path, opt.sub_folder + '_png', opt.test_csv_root, 'visualization', opt.in_channel, opt.image_size)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    visualize_layer(opt, model, val_loader)
    