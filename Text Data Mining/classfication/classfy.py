from __future__ import print_function
from logging import exception
from transformers import pipeline

import numpy as np

import torch
import torch.nn as nn
import time
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 50)
        self.fc3 = nn.Linear(50,num_classes)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self,x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

input_size = 768
hidden_size = 100
num_classes = 2
model = NeuralNet(input_size,hidden_size,num_classes)

try:
    checkpoint = torch.load('./model_binary_3800.pth')
    model.load_state_dict(checkpoint['net'])        # 从字典中依次读取
    model.eval()
    start_epoch = checkpoint['epoch']
    print('===> Load last checkpoint data')
except FileNotFoundError:
    print('Can\'t found autoencoder.t7')

nlp_features = pipeline('feature-extraction')

file = open('new_whole.txt', 'r',encoding='utf8')
compab = file.readlines()
file.close()

num = len(compab)
print('There are {} sens.'.format(num))
start = time.time()
for i in range(num):
    compab[i] = compab[i][0:-1]
    try:
        feature = np.array(nlp_features(compab[i]))[:,0,:]
        data=torch.tensor(feature,dtype=torch.float32)
        pred = model(data)
        preds = pred.argmax(dim=1)
        if preds == 0:
            f=open('rela_0713.txt','a',encoding='utf8')
            f.write(compab[i])
            f.write('\n')
            f.close
        if (i+1)%1000==0:
            stop = time.time()
            print('After {} min :{} has been checked'.format((stop-start)/60,i+1))
    except Exception as e:
        f = open('error.txt','a')
        recall = str(i) + ':' + str(e) + '\n' 
        f.write(recall)
        f.close()
