from alexnet import AlexNet
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from data_process import MyDataset
from torchvision import transforms,utils
from torch.utils.data import DataLoader
import torch,time

train_data = MyDataset(txt='./data/train.txt',transform=transforms.ToTensor())
train_loader = DataLoader(train_data,batch_size=50,shuffle=True) # 返回的是迭代器
test_data = MyDataset(txt='./data/val.txt',transform = transforms.ToTensor())
test_loader = DataLoader(test_data,batch_size=50)

# 这里使用 GPU，将模型加载到显存
# model = AlexNet().cuda()
model = AlexNet()
# print("model :",model)
# print("model parameters:",list(model.parameters()))
# Adam优化器，学习率0.001
optimizer = optim.Adam(model.parameters(),lr = 0.001)
# 5种分类，使用交叉熵损失比较方便
loss_func = nn.CrossEntropyLoss()

# 开始训练
start = time.time()
for epoch in range(30):
    print('epoch {}'.format(epoch + 1))

    # training -------------------------------
    train_loss = 0
    train_accu = 0
    for batch_x,batch_y in train_loader:
        # print("batch_x,batch_y",len(batch_x),len(batch_y))
        # batch_x -> img_data batch_y -> img_label
        # batch_x, batch_y = batch_x.cuda(), batch_y.cuda()  # 数据加载到显存
        out = model(batch_x)
        loss = loss_func(out,batch_y)
        train_loss += loss.item()
        pred = torch.max(out,1)[1]
        # print("pred",pred)
        train_correct = (pred == batch_y).sum()
        train_accu += train_correct.item()
        print("train_loss:",train_loss ," " ,"train_accu:",train_accu)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    mean_loss = train_loss/(len(train_data))
    mean_accu = train_accu/(len(train_data))
    print('Training Loss : %.6f,Accu: %.6f' % (mean_loss, mean_accu))

    #evaluation------------------------
    model.eval()
    eval_loss = 0
    eval_accu = 0
    for batch_x,batch_y in test_loader:
        # batch_x, batch_y = Variable(batch_x, volatile=True).cuda(), Variable(batch_y, volatile=True).cuda()
        batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
        out = model(batch_x)
        loss = loss_func(out,batch_y)
        eval_loss += loss.item()
        num_correct = (pred == batch_y).sum()
        eval_accu += num_correct.data[0]
    mean_loss = eval_loss / (len(test_data))
    mean_accu = eval_accu / (len(test_data))
    print('Testing Loss:%.6f,Accu:%.6f' % (mean_loss, mean_accu))

print(u'训练结束，执行时间：', time.time() - start)
torch.save(model,'./models/alexnet.pkl')