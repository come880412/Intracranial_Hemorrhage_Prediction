'''
Modified Date: 2022/01/11
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import os
import tqdm
from dataset import DCM_data
from torch.utils.data import DataLoader
import argparse
from model import resnext50_32x4d, SEresnet50
from tensorboardX import SummaryWriter
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, StepLR
from dcm_to_png import prepare_images
import random
from utils import val_f1, calculate_pos_weights

def train(opt, model, train_loader, val_loader):
    writer = SummaryWriter('runs/%s' % opt.saved_name)
    cuda = True if torch.cuda.is_available() else False
    if opt.use_weight:
        print('Use unbalanced weight')
        pos_weight = calculate_pos_weights(opt.label_count, opt.num_train_data)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        print('Use normal weight')
        criterion = torch.nn.BCEWithLogitsLoss()

    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay = opt.weight_decay)
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay = opt.weight_decay, momentum=0.9)

    """lr_scheduler"""
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - opt.lr_decay_epoch) / float(opt.lr_decay_epoch + 1)
        return lr_l
    if opt.scheduler == 'linear':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=opt.lr_decay_epoch, gamma=opt.gamma)
    
    if cuda:
        criterion = criterion.cuda()
        model = model.cuda()
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    """training"""
    print('Start training!')
    max_f1 = 0.
    optimal_threshold = 0
    train_update = 0
    for epoch in range(opt.initial_epoch, opt.n_epochs):
        model.train()
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, opt.n_epochs), unit=" step")
        total_loss = 0.
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        for image, label in train_loader:
            if cuda:
                image = image.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(image) #(B, num_classes)
                loss = criterion(pred, label.type(torch.float))
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            
            total_loss += loss.item()
            pred_label = pred.sigmoid()

            outPRED = torch.cat((outPRED, pred_label.data), 0)
            outGT = torch.cat((outGT, label.float()), 0)
            
            pbar.update()
            pbar.set_postfix(
            loss_total=f"{total_loss:.4f}",
            loss=f"{loss:.2f}"
            )
            writer.add_scalar('training loss', loss.item(), train_update)
            train_update += 1

        max_result_f1, max_precision, max_recall = val_f1(outPRED, outGT, opt.threshold)

        writer.add_scalar('training f1_score', max_result_f1, epoch)
        writer.add_scalar('training precision', max_precision, epoch)
        writer.add_scalar('training recall', max_recall, epoch)

        pbar.update()
        pbar.set_postfix(
        loss=f"{total_loss:.4f}",
        precision_avg=f"{max_precision:.2f}",
        recall_avg=f"{max_recall:.2f}",
        F1_score=f"{max_result_f1:.2f}"
        )
        pbar.close()

        f1 = val(val_loader, model, writer, epoch, opt.threshold, criterion)
        if f1 > max_f1:
            max_f1 = f1
            optimal_threshold = opt.threshold
            torch.save(model.state_dict(), '%s/%s/model_epoch%d_f1%.2f_threshold%.2f.pth' % (opt.save_model_path, opt.saved_name, epoch, max_f1, optimal_threshold))
            print('save model!!')
        scheduler.step()

def val(val_loader, model, writer, epoch, threshold, criterion):
    cuda = True if torch.cuda.is_available() else False
    model.eval()
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
            pred_label = pred.sigmoid()

            outPRED = torch.cat((outPRED, pred_label.data), 0)
            outGT = torch.cat((outGT, label.float()), 0)
            
            pbar.update()
            pbar.set_postfix(
            loss=f"{total_loss:.4f}",
            )
    max_result_f1, optimal_precision, optimal_recall = val_f1(outPRED, outGT, threshold)
    pbar.update()
    pbar.set_postfix(
        loss=f"{total_loss:.4f}",
        precision_avg=f"{optimal_precision:.2f}",
        recall_avg=f"{optimal_recall:.2f}",
        F1_score=f"{max_result_f1:.2f}"
    )
    pbar.close()
    writer.add_scalar('validation precision', optimal_precision, epoch)
    writer.add_scalar('validation recall', optimal_recall, epoch)
    writer.add_scalar('validation f1_score', max_result_f1, epoch)
    writer.add_scalar('validation loss', total_loss, epoch)
    return max_result_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """train_model"""
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--initial_epoch", type=int, default=0, help="Start epoch")
    parser.add_argument("--lr_decay_epoch", type=int, default=50, help="Start lr_decay")
    parser.add_argument("--optimizer", type=str, default='adam', help="which optimizer(adam/sgd)")
    parser.add_argument("--saved_model", type=str, default='', help="save path of model")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument('--scheduler', default='step', help='linear/step')
    parser.add_argument("--gamma", type=float, default=0.412, help="decay factor of 'step' scheduler")
    parser.add_argument("--train_val_ratio", type=float, default=0.8, help="train/val split ratio")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold for pred_label")
    parser.add_argument("--weight_decay", type=float, default=0, help="L2 regularization")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    """base_options"""
    parser.add_argument("--data_path", type=str, default='../dataset', help="path to dataset")
    parser.add_argument("--train_csv_root", type=str, default='../dataset/train.csv', help="path to training csv")
    parser.add_argument("--val_csv_root", type=str, default='../dataset/val.csv', help="path to validation csv")
    parser.add_argument("--sub_folder", type=str, default='train1', help="path to sub_folder(train1/test/train2)")
    parser.add_argument("--num_classes", type=int, default=5, help="number of classes(5 for multicalss and 2 for binary)")
    parser.add_argument("--semi_supervised", type=int, default=0, help="whether do semi_supervised learning(1/0)")
    parser.add_argument("--use_weight", type=int, default=1, help="use unbalanced data weight(1/0)")
    parser.add_argument("--image_size", type=int, default=512, help="Size of image")
    parser.add_argument("--model", type=str, default='resnext50', help="which model you want to use(resnext50/SEresnet50)")
    parser.add_argument("--in_channel", type=int, default=3, help="the number of input channels")
    parser.add_argument("--saved_name", type=str, default='model_resnext50_multiclass', help="the name to save")
    parser.add_argument("--save_model_path", type=str, default='./checkpoints', help="name of the dataset_list")
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id")
    opt = parser.parse_args()
    dcm_path = os.path.join(opt.data_path, opt.sub_folder)
    png_path = os.path.join(opt.data_path, opt.sub_folder + '_png')
    os.makedirs(os.path.join(opt.save_model_path, opt.saved_name), exist_ok=True)
    os.makedirs(png_path, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    # Set random seed
    random_seed_general = 500
    random.seed(random_seed_general) 
    torch.manual_seed(random_seed_general)
    torch.cuda.manual_seed_all(random_seed_general)
    np.random.seed(random_seed_general)
    random.seed(random_seed_general)
    torch.backends.cudnn.deterministic = True

    # Prepare data
    prepare_images(opt.data_path, dcm_path, png_path+'/', opt.train_val_ratio, opt.sub_folder)

    # Load data
    print('Load multiclass training data')
    train_data = DCM_data(opt.data_path, opt.sub_folder + '_png', opt.train_csv_root, 'train', opt.in_channel, opt.image_size)
    print('Load multiclass validation data')
    val_data = DCM_data(opt.data_path, opt.sub_folder + '_png', opt.val_csv_root, 'val', opt.in_channel, opt.image_size)
    print('number of training data : ', len(train_data))
    print('number of validation data : ', len(val_data))

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, pin_memory=True)

    # For weighted BCE
    opt.num_train_data = len(train_data)
    opt.label_count = train_data.label_count

    # Load model
    if opt.model == 'resnext50':
        model = resnext50_32x4d(opt)
    elif opt.model == 'SEresnet50':
        model = SEresnet50(opt)
    
    if opt.saved_model:
        print('load pretrained model!')
        model.load_state_dict(torch.load(opt.saved_model))
    else:
        print('training from scratch')

    train(opt, model, train_loader, val_loader)

