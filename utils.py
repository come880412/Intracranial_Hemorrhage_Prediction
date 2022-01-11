import torch
import numpy as np
from pytorch_grad_cam import GradCAM

def calculate_pos_weights(class_counts, num_of_data): # ref: https://stackoverflow.com/questions/57021620/how-to-calculate-unbalanced-weights-for-bcewithlogitsloss-in-pytorch
    pos_weights = np.ones_like(class_counts)
    neg_counts = [num_of_data-pos_count for pos_count in class_counts]
    for cdx, (pos_count, neg_count) in enumerate(zip(class_counts,  neg_counts)):
        pos_weights[cdx] = neg_count / (pos_count + 1e-5)

    return torch.as_tensor(pos_weights, dtype=torch.float)

def val_f1(output, target, threshold):  # ref: https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/blob/master/2DNet/src/tuils/tools.py
    eps=1e-20
    target = target.type(torch.cuda.ByteTensor)

    precision_total = 0
    recall_total = 0
    f1_total = 0
    for i in range(output.shape[1]):

        output_class = output[:, i]
        target_class = target[:, i]
        
        prob = output_class > threshold
        label = target_class > 0.5
        # print(prob, label)
        TP = (prob & label).sum().float()
        TN = ((~prob) & (~label)).sum().float()
        FP = (prob & (~label)).sum().float()
        FN = ((~prob) & label).sum().float()

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)

        result_f1 = 2 * precision  * recall / (precision + recall + eps)

        precision_total += precision.item()
        recall_total += recall.item()
        f1_total += result_f1.item()

    f1_avg = f1_total / output.shape[1]
    precision_avg = precision_total / output.shape[1]
    recall_avg = recall_total / output.shape[1]

    return f1_avg, precision_avg, recall_avg

def grad_cam(input_tensor, model):
    target_layers = [model.model_ft.layer4[-1]]
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    target_category = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category, aug_smooth=True)

    return grayscale_cam
    