# Intracranial_Hemorrhage_Prediction
This repo is for the midtern project of the course Deep Learning in Medical Imaging (DLMI). Please refer to the technical document if you are interested in this task.

# Task description
This task is intracranial hemorrhage prediction. There are 5 classes in total: subarachnoid, epidural, intraventricular, intraparenchymal, and subdural. Given a Dicom file, we should determine whether the image is a positive sample or not. If it is a positive sample, we should further determine which class of sick he/she obtains. This task is a multi-label classification problem since the image may not have only one class of sick. 
# Preprocessing
First, we transform the Dicom format image into a CT image according to "intercept" and "slope" information in the Dicom file. The visualization figure is illustrated below:
<p align="center">
<img src="https://github.com/come880412/Intracranial_Hemorrhage_Prediction/blob/main/images/Preprocessing.png" width=40% height=40%>
</p>
In addition, we found that the image has many black edges. We also apply the black edge cropping algorithm to reduce the black borders. The figure below visualizes that the image before cropping and after cropping.
<p align="center">
<img src="https://github.com/come880412/Intracranial_Hemorrhage_Prediction/blob/main/images/edge_cut.png" width=40% height=40%>
</p>

# Method
We applied the ensemble model to make our predicted result more robust. The ensemble model consists of the two popular models: ResNext50 and SEResNet50. The model architecture is illustrated below:
<p align="center">
<img src="https://github.com/come880412/Intracranial_Hemorrhage_Prediction/blob/main/images/model.png" width=78% height=78%>
</p>

# Experiments
- The metric in this task is F1_score\
  _**F1_score = 2 * (precision * recall) / (precision + recall)**_

- Our experimental results are listed below:

| Model(F1_score) | Validation | Public data |
|:----------:|:----------:|:----------:|
| ResNet18  | 0.82 | 0.66583 |
| ResNext50 | 0.84 | 0.69819|
| SEResNet50| 0.84 | 0.69075|
| Ensemble | _**0.86**_ | _**0.72167**_ |

- Grad_cam visualization
To make the model interpretable, we apply the Grad_cam to visualize where the model focuses to make the model interpretable. The visualization result is shown below:
<p align="center">
<img src="https://github.com/come880412/Intracranial_Hemorrhage_Prediction/blob/main/images/Grad_cam.png" width=50% height=50%>
</p>

# Getting started
### Requirements 
1. torch 1.3.1+
2. grad-cam 
3. opencv 4.5
4. tensorboradX
5. tqdm

### Download the pretrained models
```bash
$ bash download.sh
```

### Training from scratch
SEresnet50 training
```bash
$ python train.py --data_path ../dataset --n_cpu 8 --batch_size 48 --model SEresnet50 --saved_name model_SEresnet50
```

resnext50 training
```bash
$ python train.py --data_path ../dataset --n_cpu 8 --batch_size 64 --model resnext50 --saved_name model_resnext50
```

### Inference
```bash
$ python inference.py --data_path ../dataset --output ./output.csv
```

If you have any implementation problem, feel free to E-mail me! come880412@gmail.com
