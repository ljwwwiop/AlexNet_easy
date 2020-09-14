from torch.utils.data import Dataset,DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms,utils

# 数据加载预处理
def default_loader(path):
    im = Image.open(path).convert('RGB')
    im = np.asarray(im.resize((224,224)))
    # print("im.shape",im.shape)
    return im

class MyDataset(Dataset):
    def __init__(self,txt,transform = None,target_transform=None,loader = default_loader):
        f = open(txt,'r')
        # 从train.txt获取 txt ,标签类型
        self.folder = txt.split('/')[-1].split('.')[0]
        imgs = []
        for line in f.readlines():
            img_name = line.split()[0]
            label = line.split()[1]
            imgs.append((img_name,int(label)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self,index):
        # 类对象的处理方法
        # 类似C++ 对象重载
        img_name ,label = self.imgs[index]
        img_path = './data/'+ self.folder +'/'+img_name
        img = self.loader(img_path)
        if self.transform is not None :
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

