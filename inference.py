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
from utils import val_f1
import matplotlib.pyplot as plt
import cv2

label_name_list = {'intraventricular':0, 'subarachnoid':1, 'subdural':2, 'intraparenchymal':3, 'epidural':4,
                   0:'intraventricular', 1:'subarachnoid', 2:'subdural', 3:'intraparenchymal', 4:'epidural'}

def val(opt, val_loader, model):
    cuda = True if torch.cuda.is_available() else False
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    if cuda:
        criterion = criterion.cuda()
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="val", unit="step")
    total_loss = 0.
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    with torch.no_grad():
        for image, label in val_loader:
            if cuda:
                image = image.cuda()
                label = label.cuda()
            
            pred = model(image) #(B, num_classes)
            loss = criterion(pred, label.type(torch.float))
            total_loss += loss.item()
            if not opt.ensemble:
                pred = pred.sigmoid()

            outPRED = torch.cat((outPRED, pred.data), 0)
            outGT = torch.cat((outGT, label.float()), 0)
            
            pbar.update()
            pbar.set_postfix(
            loss=f"{total_loss:.4f}",
            )
    max_result_f1, optimal_precision, optimal_recall = val_f1(outPRED, outGT, opt.threshold)
    pbar.update()
    pbar.set_postfix(
        loss=f"{total_loss:.4f}",
        precision_avg=f"{optimal_precision:.2f}",
        recall_avg=f"{optimal_recall:.2f}",
        F1_score=f"{max_result_f1:.2f}"
    )
    pbar.close()

def test(opt, test_loader, model):
    cuda = True if torch.cuda.is_available() else False
    model.eval()
    save_csv = [['id', 'label']]
    with torch.no_grad():
        for image, dcm_file_name in tqdm.tqdm(test_loader):
            dcm_file_name = np.array(dcm_file_name).T
            if cuda:
                image = image.cuda()

            pred = model(image) #(B, num_classes)
            if not opt.ensemble:
                pred = pred.sigmoid()
            pred_label = pred.cpu().detach().numpy()
            for i in range(len(pred_label)):
                label = np.round(pred_label[i], 3)
                dcm_file_name_list = dcm_file_name[i]
                for saved_name in dcm_file_name_list:
                    subtype = saved_name.split('_')[1]
                    pred_result = label[label_name_list[subtype]]
                    if pred_result >= opt.threshold:
                        save_csv.append([saved_name, '1'])
                    else:
                        save_csv.append([saved_name, '0'])

    np.savetxt(opt.output,  save_csv, fmt='%s', delimiter=',')

def semi_supervised_extract(opt, train2_loader, model):
    cuda = True if torch.cuda.is_available() else False
    model.eval()
    save_csv = [['filename', 'intraventricular', 'subarachnoid', 'subdural', 'intraparenchymal', 'epidural']]
    with torch.no_grad():
        for image, file_name_list in tqdm.tqdm(train2_loader):
            if cuda:
                image = image.cuda()

            pred = model(image) #(B, num_classes)
            pred = pred.cpu().detach().numpy()
            pred = np.round(pred, 4)
            for i in range(len(pred)):
                logits = pred[i]
                file_name = file_name_list[i]
                save_csv.append([file_name, logits[0], logits[1], logits[2], logits[3], logits[4]])

    np.savetxt(os.path.join(opt.data_path, 'train2_png.csv'), save_csv, fmt='%s', delimiter=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model", type=str, default='./model_epoch23_f10.85_threshold0.50.pth', help="save path of model")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--image_size", type=int, default=512, help="size of the batches")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold for multi-label classification")
    """base_options"""
    parser.add_argument("--data_path", type=str, default='../dataset', help="path to dataset")
    parser.add_argument("--output", type=str, default='./output.csv', help="path to output csv")
    parser.add_argument("--sub_folder", type=str, default='test', help="path to sub_folder(train1/test/train2)")
    parser.add_argument("--test_csv_root", type=str, default='val.csv', help="name of csv")
    parser.add_argument("--train_val_ratio", type=float, default=0.8, help="train/val split ratio")
    parser.add_argument("--ensemble", type=int, default=1, help="ensemble net(1/0)")
    parser.add_argument("--model", type=str, default='resnext50', help="which model you want to use(resnext50/SEresnet50)")
    parser.add_argument("--test_mode", type=str, default='test', help="test/val/semi")
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
    prepare_images(opt.data_path, dcm_path, png_path+'/', opt.train_val_ratio, opt.sub_folder)
    
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
    

    if opt.test_mode == 'val':   # Test model performance
        val_data = DCM_data(opt.data_path, opt.sub_folder + '_png', opt.test_csv_root, 'val', opt.in_channel, opt.image_size)
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
        val(opt, val_loader, model)

    elif opt.test_mode == 'test': # Generate public csv file
        test_data = DCM_data(opt.data_path, opt.sub_folder + '_png', None, 'test', opt.in_channel, opt.image_size)
        test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
        test(opt, test_loader, model)

    elif opt.test_mode == 'semi': # Generate psuedo labels
        train2_data = DCM_data(opt.data_path, opt.sub_folder + '_png', None, 'semi', opt.in_channel, opt.image_size)
        train2_loader = DataLoader(train2_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
        semi_supervised_extract(opt, train2_loader, model)
    