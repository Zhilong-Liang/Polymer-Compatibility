from __future__ import print_function
import ipywidgets as widgets
from transformers import pipeline

import numpy as np
from sklearn import svm

nlp_features = pipeline('feature-extraction')

file = open('positive_sample', 'r' )
positive = file.readlines()
file.close()
num = len(positive)
for i in range(num):
    positive[i] = positive[i][0:-1]
print(len(positive))

file = open('negative_sample', 'r' )
negative = file.readlines()
file.close()
num = len(negative)
for i in range(num):
    negative[i] = negative[i][0:-1]
print(len(negative))

file = open('change_sample', 'r' )
change = file.readlines()
file.close()
num = len(change)
for i in range(num):
    change[i] = change[i][0:-1]
print(len(change))

file = open('non_rela', 'r' )
norela = file.readlines()
file.close()
num = len(norela)
for i in range(num):
    norela[i] = norela[i][0:-1]
print(len(norela))

import random
list = range(0,1832)
gt = random.sample(list,120)
norela_1 = []
for i in range(300):
    norela_1.append(norela[i])
print(len(norela_1))
norela_2 = []
for i in range(300):
    norela_2.append(norela[i+300])
print(len(norela_2))
norela_3 = []
for i in range(300):
    norela_3.append(norela[i+600])
print(len(norela_3))
norela_4 = []
for i in range(300):
    norela_4.append(norela[i+900])
print(len(norela_4))
norela_5 = []
for i in range(300):
    norela_5.append(norela[i+1200])
print(len(norela_5))
norela_6 = []
for i in range(300):
    norela_6.append(norela[i+1500])
print(len(norela_6))

output_pos = nlp_features(positive)
print(np.array(output_pos).shape)   # (Samples, Tokens, Vector Size)

output_neg = nlp_features(negative)
print(np.array(output_neg).shape)   # (Samples, Tokens, Vector Size)

output_cha = nlp_features(change)
print(np.array(output_cha).shape)   # (Samples, Tokens, Vector Size)

output_norela_1 = nlp_features(norela_1)
print(np.array(output_norela_1).shape)   # (Samples, Tokens, Vector Size)

output_norela_2 = nlp_features(norela_2) 
print(np.array(output_norela_2).shape) 

output_norela_3 = nlp_features(norela_3) 
print(np.array(output_norela_3).shape) 

output_norela_4 = nlp_features(norela_4) 
print(np.array(output_norela_4).shape) 

output_norela_5 = nlp_features(norela_5) 
print(np.array(output_norela_5).shape) 

output_norela_6 = nlp_features(norela_6) 
print(np.array(output_norela_6).shape) 

rela = np.concatenate((np.array(output_pos)[:,0,:],np.array(output_neg)[:,0,:],np.array(output_cha)[:,0,:]))
rela = np.tile(rela,[10,1])

import torch

data = np.concatenate((np.array(rela),np.array(output_norela_1)[:,0,:],np.array(output_norela_2)[:,0,:],np.array(output_norela_3)[:,0,:],np.array(output_norela_4)[:,0,:],np.array(output_norela_5)[:,0,:],np.array(output_norela_6)[:,0,:]))
print(np.array(data).shape)
label = [0]*1880 + [1]*1800

import torch.nn as nn

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
num_epochs = 1000
batch_size = 300
learning_rate = 0.00005

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data,label,train_size=0.6)
print(len(x_train))

class MyDataset(Dataset):
    def __init__(self):
        lines = [[y_train[i], x_train[i]] for i in range(2207)]
        self.df = pd.DataFrame(lines, columns=['label', 'data'])
    def __getitem__(self, item):
        single_item = self.df.iloc[item, :]
        return single_item.values[0], single_item.values[1]
    def __len__(self):
        return self.df.shape[0]

dataset = MyDataset()

train_loader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
model_binary = NeuralNet(input_size,hidden_size,num_classes)

import pandas as pd
print(len(x_test))
class MyTestset(Dataset):
    def __init__(self):
        lines = [[y_test[i], x_test[i]] for i in range(1471)]
        self.df = pd.DataFrame(lines, columns=['label', 'data'])
    def __getitem__(self, item):
        single_item = selftest_loader = single_item = self.df.iloc[item, :]
        return single_item.values[0], single_item.values[1]
    def __len__(self):
        return self.df.shape[0]

testset = MyTestset()
test_loader = DataLoader(testset,batch_size=batch_size, shuffle=True)
def test_func(model,test_loader=test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for i,(slabel, sdata) in enumerate(test_loader):
            sdata=sdata.to(torch.float32)
            outputs = model(sdata)
            _,predicted = torch.max(outputs.data,1)
            total += slabel.size(0)
            correct +=(predicted == slabel).sum().item()
    print('Accuracy : {}%'.format(100*correct/total))
    return 100*correct/total
import torch.optim as optim
import torch
total_step = len(train_loader)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_binary.parameters(),lr=learning_rate)

best = 0
acc = 0
for epoch in range(num_epochs):
    if acc > best:
        state = {'net':model_binary.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, f'./binary_1880_1800/model{acc}.pth')
    for i, (slabel,sdata) in enumerate(train_loader):
        sdata=sdata.to(torch.float32)
        outputs = model_binary(sdata)
        loss = criterion(outputs,slabel)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}],Loss:{:.4f}'.format(epoch,num_epochs,loss.item()))
    acc = test_func(model_binary)