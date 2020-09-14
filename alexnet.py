'''
    学习pytorch 和 学习 alexnet
'''

import torch
import torch.nn as nn
from torchvision import transforms,utils
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import pdb

# 建立模型
class AlexNet(nn.Module):

    # 初始化
    def __init__(self,num_class = 5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True) ,#inplace:原地　　不创建新对象，直接对传入数据进行修改
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*13*13,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_class),
        )

    # 向前传播
    def forward(self, x):
        x = self.features(x)
        #函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        #2163200
        # print(x.shape)
        x = x.view(x.size(0),256*13*13)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # print("x :",x)
        x = self.classifier(x)
        return x


