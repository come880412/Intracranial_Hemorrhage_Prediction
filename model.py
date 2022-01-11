import torch.nn as nn
import torchvision
import torch
import torch.hub

class SEresnet50(nn.Module):
    def __init__(self, opt):
        super(SEresnet50, self).__init__()
        self.model = torch.hub.load(
        'moskomule/senet.pytorch',
        'se_resnet50',
        pretrained=True,)
        in_feature = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feature, opt.num_classes, bias=True)
    def forward(self, x):
        x = self.model(x)
        return x

class Ensemble_net(nn.Module):
    def __init__(self, model_list):
        super(Ensemble_net, self).__init__()
        self.model = model_list
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for idx, m in enumerate(self.model):
            if idx == 0:
                out = self.sigmoid(m(x)).unsqueeze(2)
            else:
                out = torch.cat([out, self.sigmoid(m(x)).unsqueeze(2)], dim=2)
        return out.mean(dim=2)

class resnext50_32x4d(nn.Module):
    def __init__(self, opt):
        super(resnext50_32x4d, self).__init__()
        self.model_ft = torchvision.models.resnext50_32x4d(pretrained=True)
        if opt.in_channel == 1:
            self.model_ft.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, opt.num_classes, bias=True))

    def forward(self, x):
        x = self.model_ft(x)
        return x
