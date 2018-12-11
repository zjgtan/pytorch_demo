# coding: utf8
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # embedding层
        # num_embeddings = 10, 词表大小
        # embedding_dim = 3, 词向量维度
        # padding_idx, 需要mask的word序号
        self.embedding = nn.Embedding(10, 3)
        self.lstm = nn.LSTM(3, 5)
        self.softmax = nn.LogSoftmax()

        self.linear = nn.Linear(5, 2)
        self.softmax = nn.LogSoftmax()


    def forward(self, x):
        """
        x: tensor.LongTensor, [batch_size, length]

        """
        # embedding层
        emb1 = self.embedding(x) # [batch_size, length, embedding_size]
        # 矩阵的转置
        # 由于nn.Conv1d默认是在最后一维上进行卷积，通过转置将让最后一维表示sequence
        #emb1 = emb1.permute(0, 2, 1) # [batch_size, embedding_size, length], 为了进行卷积

        # LSTM层
        # 将每个batch的embedding序列转换为lstm的隐层序列输出
        out, (_, _) = self.lstm(emb1) #out [batch_size, length, lstm_hidden_size]
        # 将各个时间片的隐向量求均值合并成一个，也即 out[1, 1] = avg(out[1, :, 1])
        # 由于avgpool1d默认在最后一维进行，需要进行转置
        out = out.permute(0, 2, 1) # 
        out = nn.AvgPool1d(9)(out).view(x.size()[0], 5) # avg(out[1, 1, :])
        out = self.linear(out)
        out = self.softmax(out)

        return out



net = Net()

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 1,
        momentum = 0.9)

x = torch.LongTensor([[1,2,3, 4, 5, 6, 7, 8, 9], 
    [9, 8, 7, 6, 5, 4, 3, 2, 1]])

y = torch.LongTensor([1, 0])

for epoch in range(100):
    out = net(x)

    loss = criterion(out, y)

    print loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 预测
output = net(x)
preds = list(np.argmax(output.data.numpy(), axis=1).flatten())
print preds

