import torch
import numpy as np
from torch.autograd import Variable
from data_process import default_loader

classes = ['bus','dinosaur','elephant','flower','horse']

# begin predict
# model = torch.load('./models/alexnet.pkl').cuda()
model = torch.load('./models/alexnet.pkl')
im = default_loader('473.jpg')
# 扩展一个维度 （Ｎ，Ｃ，Ｈ，Ｗ）
im = np.expand_dims(im,0)
# n h w c  -> n c h w
im = im.transpose(0,3,1,2)
im = torch.from_numpy(im).float()
# x = Variable(im).cuda()
x = Variable(im)

print("x Variable:",x)
pred = model(x)
print("predict pred:",pred)
index = torch.max(pred,1)[1].data[0]
print('预测结果:%s'%(classes[index]))


