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
        self.lstm = nn.LSTM(3, 5, batch_first = True)
        self.softmax = nn.LogSoftmax()

        self.linear = nn.Linear(5, 2)
        self.softmax = nn.LogSoftmax()


    def forward(self, sentence):
        """
        x: tensor.LongTensor, [batch_size, length]

        """
        # embedding层
        emb1 = self.embedding(sentence) # [length, embedding_size]
        # 矩阵的转置
        # 由于nn.Conv1d默认是在最后一维上进行卷积，通过转置将让最后一维表示sequence
        #emb1 = emb1.permute(1, 0, 2) # [batch_size, embedding_size, length], 为了进行卷积

        emb1 = emb1.view(len(sentence), 1, -1)
        # LSTM层的操作
        # 
        out, hidden = self.lstm(emb1)
        out = out.view(len(sentence), 5)
        #out = out.permute(1, 0, 2)
        # LSTM层
        # 将每个batch的embedding序列转换为lstm的隐层序列输出
        out = self.linear(out)
        out = self.softmax(out)
        return out

net = Net()

'''
x = torch.LongTensor([1,2,3, 4 ])
y = torch.LongTensor([[1, 1, 0, 0], [0, 0, 1, 1]])
ret = net(x)
print ret

exit(0)
'''

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 1,
        momentum = 0.9)

sentences = [[1,2,3, 4, 5, 6, 7, 8, 9], 
    [9, 8, 7, 6, 5, 4, 3, 2, 1]]

labels = [[1, 1, 1, 1, 1, 0, 0, 0, 0], 
    [0, 0, 0, 0, 1, 1, 1, 1, 1]]

for epoch in range(100):
    for ix in range(2):
        x = torch.LongTensor(sentences[ix])
        y = torch.LongTensor(labels[ix])

        out = net(x)

        loss = criterion(out, y.view(-1))

        print loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 预测
output = net(torch.LongTensor(sentences[0]))
print output
preds = np.argmax(output.data.numpy(), axis=1)
print preds

output = net(torch.LongTensor(sentences[1]))
print output
preds = np.argmax(output.data.numpy(), axis=1)
print preds

