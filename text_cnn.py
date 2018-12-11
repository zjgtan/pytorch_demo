# coding: utf8
import torch
from torch import nn
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(10, 3)
        self.conv1 = nn.Conv1d(3, 100, 2) # 窗口大小位2的卷积
        self.conv2 = nn.Conv1d(3, 100, 3) # 窗口大小位2的卷积
        self.conv3 = nn.Conv1d(3, 100, 4) # 窗口大小位2的卷积

        self.fc = nn.Linear(300, 2)
        self.sorfmax = nn.LogSoftmax()


    def forward(self, x):
        """
        x: tensor.LongTensor, [batch_size, length]

        """
        emb1 = self.embedding(x) # [batch_size, length, embedding_size]
        emb1 = emb1.permute(0, 2, 1) # [batch_size, embedding_size, length], 为了进行卷积
        print emb1.shape # [1, 3, 9]

        # 对emb1的每个样本的每一行进行卷积，也即embedding的某一维构成的序列
        conv1 = self.conv1(emb1) # [batch_size, 通道数, 序列上的卷积窗口滑动数]
        print conv1.shape
        m1 = nn.MaxPool1d(conv1.size()[-1])(conv1).view(x.size()[0], 100)

        conv2 = self.conv2(emb1)
        m2 = nn.MaxPool1d(conv2.size()[-1])(conv2).view(x.size()[0], 100)

        conv3 = self.conv3(emb1)
        m3 = nn.MaxPool1d(conv3.size()[-1])(conv3).view(x.size()[0], 100)

        x = torch.cat((m1, m2, m3), 1)
        x = self.fc(x)
        x = self.sorfmax(x)

        return x




net = Net()
x = torch.LongTensor([[1,2,3, 4, 5, 6, 7, 8, 9]])
net.forward(x)

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 1e-3,
        momentum = 0.9)

x = torch.LongTensor([[1,2,3, 4, 5, 6, 7, 8, 9], 
    [9, 8, 7, 6, 5, 4, 3, 2, 1]])

y = torch.LongTensor([1, 0])

for epoch in range(10000):
    out = net(x)

    loss = criterion(out, y)

    print loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()







