from time import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.data as Data
import matplotlib.pyplot as plt
import os
import time

def whole_net(x_train,y_train,x_test,y_test,folder,lrate,hidden_size,kind,itera,cuda):

    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available:
        print('GPU',device)
    [h1,h2,h3,h4,h5,h6,h7] = hidden_size

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train_c = x_train[:,4096:4098]/100
    x_train = np.hstack((x_train[:,0:4096],x_train_c))
    x_test_c = x_test[:,4096:4098]/100
    x_test = np.hstack((x_test[:,0:4096],x_test_c))
    
    train_xt = torch.from_numpy(x_train.astype(np.float32))
    train_yt = torch.from_numpy(y_train.astype(np.float32))
    test_xt = torch.from_numpy(x_test.astype(np.float32))
    test_yt = torch.from_numpy(y_test.astype(np.float32))
    train_xt = train_xt.to(device)
    train_yt = train_yt.to(device)
    test_xt = test_xt.to(device)
    test_yt = test_yt.to(device)


    train_data = Data.TensorDataset(train_xt,train_yt)
    test_data = Data.TensorDataset(test_xt,test_yt)

    train_loader = Data.DataLoader(dataset=train_data,batch_size=20,shuffle=True,num_workers=0,drop_last=False)
    test_loader = Data.DataLoader(dataset=test_data,batch_size=20,shuffle=True,num_workers=0,drop_last=False)
    print(train_xt.shape,train_yt.shape,test_xt.shape,test_yt.shape)

    class HDDN_noc(nn.Module):
        def __init__(self,h1,h2,h3,h4,h5,h6,h7):
            super(HDDN_noc, self).__init__()
            
            self.MLP1 = nn.Linear(2048,h1)
            self.MLP2 = nn.Linear(h1,h2)
            self.MLP3 = nn.Linear(h2,h3)
            self.MLP4 = nn.Linear(h3,h4)
            self.MLP5 = nn.Linear(h4,h5)
            self.MLP6 = nn.Linear(h5,h6)
            self.MLP7 = nn.Linear(h6,h7)
            self.MLP8 = nn.Linear(h7,1)
            self.nolinear = nn.Sigmoid()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self,x):
            A = x[:,0:2048]
            B = x[:,2048:4096]
            C = x[:,4096:4098]
            A1 = self.MLP1(A); B1 = self.MLP1(B)
            A1 = self.dropout(A1); B1 = self.dropout(B1)
            A1 = self.nolinear(A1)*10; B1 = self.nolinear(B1)*10
            A2 = self.MLP2(A1); B2 = self.MLP2(B1)
            A2 = self.dropout(A2); B2 = self.dropout(B2)
            A2 = self.nolinear(A2)*10;B2 = self.nolinear(B2)*10
            A3 = self.MLP3(A2+A1);B3 = self.MLP3(B2+B1)
            A3 = self.dropout(A3); B3 = self.dropout(B3)
            A3 = self.nolinear(A3)*10;B3 = self.nolinear(B3)*10
            A4 = self.MLP4(A3+A2+A1);B4 = self.MLP4(B3+B2+B1)
            A4 = self.dropout(A4); B4 = self.dropout(B4)
            A4 = self.nolinear(A4)*10;B4 = self.nolinear(B4)*10
            delta = abs(A4-B4)           
            D1=self.MLP5(delta)            
            D1 = self.dropout(D1)
            D1 = self.nolinear(D1)*10            
            D2 = self.MLP6(D1)
            D2 = self.dropout(D2)
            D2 = self.nolinear(D2)*10                
            D3 = self.MLP7(D2)
            D3 = self.dropout(D3)
            D3 = self.nolinear(D3)*10           
            D4 = self.MLP8(D3)
            D4 = self.dropout(D4)
            output = self.nolinear(D4)*10
            return output[:,0]
    

    class HDDN(nn.Module):
        def __init__(self,h1,h2,h3,h4,h5,h6,h7):
            super(HDDN, self).__init__()
            
            self.MLP1 = nn.Linear(2048,h1)
            self.MLP2 = nn.Linear(h1,h2)
            self.MLP3 = nn.Linear(h2,h3)
            self.MLP4 = nn.Linear(h3,h4)
            self.MLP5 = nn.Linear(h4+2,h5)
            self.MLP6 = nn.Linear(h5,h6)
            self.MLP7 = nn.Linear(h6,h7)
            self.MLP8 = nn.Linear(h7,1)
            self.nolinear = nn.Sigmoid()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self,x):
            A = x[:,0:2048]
            B = x[:,2048:4096]
            C = x[:,4096:4098]
            A1 = self.MLP1(A); B1 = self.MLP1(B)
            A1 = self.dropout(A1); B1 = self.dropout(B1)
            A1 = self.nolinear(A1)*10; B1 = self.nolinear(B1)*10
            A2 = self.MLP2(A1); B2 = self.MLP2(B1)
            A2 = self.dropout(A2); B2 = self.dropout(B2)
            A2 = self.nolinear(A2)*10;B2 = self.nolinear(B2)*10
            A3 = self.MLP3(A2+A1);B3 = self.MLP3(B2+B1)
            A3 = self.dropout(A3); B3 = self.dropout(B3)
            A3 = self.nolinear(A3)*10;B3 = self.nolinear(B3)*10
            A4 = self.MLP4(A3+A2+A1);B4 = self.MLP4(B3+B2+B1)
            A4 = self.dropout(A4); B4 = self.dropout(B4)
            A4 = self.nolinear(A4)*10;B4 = self.nolinear(B4)*10
            delta = abs(A4-B4)
            delta = torch.cat((delta,C),1)            
            D1=self.MLP5(delta)            
            D1 = self.dropout(D1)
            D1 = self.nolinear(D1)*10           
            D2 = self.MLP6(D1)
            D2 = self.dropout(D2)
            D2 = self.nolinear(D2)*10                
            D3 = self.MLP7(D2)
            D3 = self.dropout(D3)
            D3 = self.nolinear(D3)*10             
            D4 = self.MLP8(D3)
            D4 = self.dropout(D4)
            output = self.nolinear(D4)*10
            return output[:,0]
    
    class HDDN_noabs(nn.Module):
        def __init__(self,h1,h2,h3,h4,h5,h6,h7):
            super(HDDN_noabs, self).__init__()
            
            self.MLP1 = nn.Linear(2048,h1)
            self.MLP2 = nn.Linear(h1,h2)
            self.MLP3 = nn.Linear(h2,h3)
            self.MLP4 = nn.Linear(h3,h4)
            self.MLP5 = nn.Linear(h4+2,h5)
            self.MLP6 = nn.Linear(h5,h6)
            self.MLP7 = nn.Linear(h6,h7)
            self.MLP8 = nn.Linear(h7,1)
            self.nolinear = nn.Sigmoid()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self,x):
            A = x[:,0:2048]
            B = x[:,2048:4096]
            C = x[:,4096:4098]
            A1 = self.MLP1(A); B1 = self.MLP1(B)
            A1 = self.dropout(A1); B1 = self.dropout(B1)
            A1 = self.nolinear(A1)*10; B1 = self.nolinear(B1)*10
            A2 = self.MLP2(A1); B2 = self.MLP2(B1)
            A2 = self.dropout(A2); B2 = self.dropout(B2)
            A2 = self.nolinear(A2)*10;B2 = self.nolinear(B2)*10
            A3 = self.MLP3(A2+A1);B3 = self.MLP3(B2+B1)
            A3 = self.dropout(A3); B3 = self.dropout(B3)
            A3 = self.nolinear(A3)*10;B3 = self.nolinear(B3)*10
            A4 = self.MLP4(A3+A2+A1);B4 = self.MLP4(B3+B2+B1)
            A4 = self.dropout(A4); B4 = self.dropout(B4)
            A4 = self.nolinear(A4)*10;B4 = self.nolinear(B4)*10
            delta = A4-B4
            delta = torch.cat((delta,C),1)            
            D1=self.MLP5(delta)            
            D1 = self.dropout(D1)
            D1 = self.nolinear(D1)*10           
            D2 = self.MLP6(D1)
            D2 = self.dropout(D2)
            D2 = self.nolinear(D2)*10                
            D3 = self.MLP7(D2)
            D3 = self.dropout(D3)
            D3 = self.nolinear(D3)*10             
            D4 = self.MLP8(D3)
            D4 = self.dropout(D4)
            output = self.nolinear(D4)*10
            return output[:,0]

    #搭建MLP回归网络
    class MLP(nn.Module):
        def __init__(self,h1,h2,h3,h4,h5,h6,h7):
            super(MLP, self).__init__()
            
            self.MLP1 = nn.Linear(4098,h1)
            self.MLP2 = nn.Linear(h1,h2)
            self.MLP3 = nn.Linear(h2,h3)
            self.MLP4 = nn.Linear(h3,h4)
            self.MLP5 = nn.Linear(h4,h5)
            self.MLP6 = nn.Linear(h5,h6)
            self.MLP7 = nn.Linear(h6,h7)
            self.MLP8 = nn.Linear(h7,1)
            self.nolinear = nn.Sigmoid()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self,x):
            A = x        
            A1 = self.MLP1(A)
            A1 = self.dropout(A1)
            A1 = self.nolinear(A1)*10
            A2 = self.MLP2(A1)
            A2 = self.dropout(A2)
            A2 = self.nolinear(A2)*10
            A3 = self.MLP3(A2)
            A3 = self.dropout(A3)
            A3 = self.nolinear(A3)*10
            A4 = self.MLP4(A3)
            A4 = self.dropout(A4)
            A4 = self.nolinear(A4)*10         
            D1=self.MLP5(A4)            
            D1 = self.dropout(D1)
            D1 = self.nolinear(D1)*10            
            D2 = self.MLP6(D1)
            D2 = self.dropout(D2)
            D2 = self.nolinear(D2)*10                
            D3 = self.MLP7(D2)
            D3 = self.dropout(D3)
            D3 = self.nolinear(D3)*10          
            D4 = self.MLP8(D3)
            D4 = self.dropout(D4)
            output = self.nolinear(D4)*10
            return output[:,0]
        
    class CDN(nn.Module):
        def __init__(self,h1,h2,h3,h4,h5,h6,h7):
            super(CDN, self).__init__()
            self.MLP1 = nn.Linear(2048,h1)
            self.MLP2 = nn.Linear(h1,h2)
            self.MLP3 = nn.Linear(h2,h3)
            self.MLP4 = nn.Linear(h3,h4)
            self.MLP5 = nn.Linear(h2+h3+h4+2,h5)
            self.MLP6 = nn.Linear(h5,h6)
            self.MLP7 = nn.Linear(h6,h7)
            self.MLP8 = nn.Linear(h7,1)
            self.tanh = nn.Sigmoid()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self,x):
            A = x[:,0:2048]
            B = x[:,2048:4096]
            C = x[:,4096:4098]         
            A1 = self.MLP1(A); B1 = self.MLP1(B)
            A1 = self.dropout(A1); B1 = self.dropout(B1)
            A1 = self.tanh(A1)*10; B1 = self.tanh(B1)*10            
            A2 = self.MLP2(A1); B2 = self.MLP2(B1)
            A2 = self.dropout(A2); B2 = self.dropout(B2)
            A2 = self.tanh(A2)*10;B2 = self.tanh(B2)*10
            A3 = self.MLP3(A2);B3 = self.MLP3(B2)
            A3 = self.dropout(A3); B3 = self.dropout(B3)
            A3 = self.tanh(A3)*10;B3 = self.tanh(B3)*10
            A4 = self.MLP4(A3);B4 = self.MLP4(B3)
            A4 = self.dropout(A4); B4 = self.dropout(B4)
            A4 = self.tanh(A4)*10;B4 = self.tanh(B4)*10
            delta = torch.cat((A2,A3,A4),1) - torch.cat((B2,B3,B4),1)
            delta = torch.cat((delta,C),1)
            delta = abs(delta)
            D1 = self.MLP5(delta)
            D1 = self.dropout(D1)
            D1 = self.tanh(D1)*10            
            D2 = self.MLP6(D1)
            D2 = self.dropout(D2)
            D2 = self.tanh(D2)*10    
            D3 = self.MLP7(D2)
            D3 = self.dropout(D3)
            D3 = self.tanh(D3)*10 
            D4 = self.MLP8(D3)
            D4 = self.dropout(D4)
            output = self.tanh(D4)*10
            return output[:,0]
        
    class DN(nn.Module):
        def __init__(self,h1,h2,h3,h4,h5,h6,h7):
            super(DN, self).__init__()
            
            self.MLP1 = nn.Linear(4098,h1)
            self.MLP2 = nn.Linear(h1,h2)
            self.MLP3 = nn.Linear(h2,h3)
            self.MLP4 = nn.Linear(h3,h4)
            self.MLP5 = nn.Linear(h4,h4)
            self.MLP6 = nn.Linear(h4,h4)
            self.MLP7 = nn.Linear(h4,h4)
            self.MLP8 = nn.Linear(h4,1)
            self.nolinear = nn.Sigmoid()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self,x):
            A = x        
            A1 = self.MLP1(A)
            A1 = self.dropout(A1)
            A1 = self.nolinear(A1)*10
            A2 = self.MLP2(A1)
            A2 = self.dropout(A2)
            A2 = self.nolinear(A2)*10
            A3 = self.MLP3(A2+A1)
            A3 = self.dropout(A3)
            A3 = self.nolinear(A3)*10
            A4 = self.MLP4(A3+A2+A1)
            A4 = self.dropout(A4)
            A4 = self.nolinear(A4)*10         
            D1=self.MLP5(A4+A3+A2+A1)            
            D1 = self.dropout(D1)
            D1 = self.nolinear(D1)*10            
            D2 = self.MLP6(D1+A4+A3+A2+A1)
            D2 = self.dropout(D2)
            D2 = self.nolinear(D2)*10                
            D3 = self.MLP7(D2+D1+A4+A3+A2+A1)
            D3 = self.dropout(D3)
            D3 = self.nolinear(D3)*10          
            D4 = self.MLP8(D3+D2+D1+A4+A3+A2+A1)
            D4 = self.dropout(D4)
            output = self.nolinear(D4)*10
            return output[:,0]

    class HDDN_nodense(nn.Module):
        def __init__(self,h1,h2,h3,h4,h5,h6,h7):
            super(HDDN_nodense, self).__init__()
            
            self.MLP1 = nn.Linear(2048,h1)
            self.MLP2 = nn.Linear(h1,h2)
            self.MLP3 = nn.Linear(h2,h3)
            self.MLP4 = nn.Linear(h3,h4)
            self.MLP5 = nn.Linear(h4+2,h5)
            self.MLP6 = nn.Linear(h5,h6)
            self.MLP7 = nn.Linear(h6,h7)
            self.MLP8 = nn.Linear(h7,1)
            self.nolinear = nn.Sigmoid()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self,x):
            A = x[:,0:2048]
            B = x[:,2048:4096]
            C = x[:,4096:4098]
            A1 = self.MLP1(A); B1 = self.MLP1(B)
            A1 = self.dropout(A1); B1 = self.dropout(B1)
            A1 = self.nolinear(A1)*10; B1 = self.nolinear(B1)*10
            A2 = self.MLP2(A1); B2 = self.MLP2(B1)
            A2 = self.dropout(A2); B2 = self.dropout(B2)
            A2 = self.nolinear(A2)*10;B2 = self.nolinear(B2)*10
            A3 = self.MLP3(A2);B3 = self.MLP3(B2)
            A3 = self.dropout(A3); B3 = self.dropout(B3)
            A3 = self.nolinear(A3)*10;B3 = self.nolinear(B3)*10
            A4 = self.MLP4(A3);B4 = self.MLP4(B3)
            A4 = self.dropout(A4); B4 = self.dropout(B4)
            A4 = self.nolinear(A4)*10;B4 = self.nolinear(B4)*10
            delta = abs(A4-B4)
            delta = torch.cat((delta,C),1)            
            D1=self.MLP5(delta)            
            D1 = self.dropout(D1)
            D1 = self.nolinear(D1)*10           
            D2 = self.MLP6(D1)
            D2 = self.dropout(D2)
            D2 = self.nolinear(D2)*10                
            D3 = self.MLP7(D2)
            D3 = self.dropout(D3)
            D3 = self.nolinear(D3)*10             
            D4 = self.MLP8(D3)
            D4 = self.dropout(D4)
            output = self.nolinear(D4)*10
            return output[:,0]

    class HDDN_nodiff(nn.Module):
        def __init__(self,h1,h2,h3,h4,h5,h6,h7):
            super(HDDN_nodiff, self).__init__()
            
            self.MLP1 = nn.Linear(4096,h1)
            self.MLP2 = nn.Linear(h1,h2)
            self.MLP3 = nn.Linear(h2,h3)
            self.MLP4 = nn.Linear(h3,h4)
            self.MLP5 = nn.Linear(h4+2,h5)
            self.MLP6 = nn.Linear(h5,h6)
            self.MLP7 = nn.Linear(h6,h7)
            self.MLP8 = nn.Linear(h7,1)
            self.nolinear = nn.Sigmoid()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self,x):
            A = x[:,0:4096]
            C = x[:,4096:4098]
            A1 = self.MLP1(A)
            A1 = self.dropout(A1)
            A1 = self.nolinear(A1)*10
            A2 = self.MLP2(A1)
            A2 = self.dropout(A2)
            A2 = self.nolinear(A2)*10
            A3 = self.MLP3(A2+A1)
            A3 = self.dropout(A3)
            A3 = self.nolinear(A3)*10
            A4 = self.MLP4(A3+A2+A1)
            A4 = self.dropout(A4)
            A4 = self.nolinear(A4)*10
            delta = torch.cat((A4,C),1)            
            D1=self.MLP5(delta)            
            D1 = self.dropout(D1)
            D1 = self.nolinear(D1)*10           
            D2 = self.MLP6(D1)
            D2 = self.dropout(D2)
            D2 = self.nolinear(D2)*10                
            D3 = self.MLP7(D2)
            D3 = self.dropout(D3)
            D3 = self.nolinear(D3)*10             
            D4 = self.MLP8(D3)
            D4 = self.dropout(D4)
            output = self.nolinear(D4)*10
            return output[:,0]
        
    if kind == 0:
        model = HDDN(h1,h2,h3,h4,h5,h6,h7)
    if kind == 1:
        model = HDDN_noc(h1,h2,h3,h4,h5,h6,h7)
    if kind == 2:
        model = HDDN_nodense(h1,h2,h3,h4,h5,h6,h7)
    if kind == 3:
        model = HDDN_nodiff(h1,h2,h3,h4,h5,h6,h7)
    if kind == 4:
        model = HDDN_noabs(h1,h2,h3,h4,h5,h6,h7)
    if kind == 5:
        model = MLP(h1,h2,h3,h4,h5,h6,h7)
    if kind == 6:
        model = CDN(h1,h2,h3,h4,h5,h6,h7)
    if kind == 7:
        model = DN(h1,h2,h3,h4,h5,h6,h7)

    model = model.to(device)
    optimizer = Adam(model.parameters(),lr=lrate,weight_decay=0.001)
    loss_func = nn.MSELoss()
    train_loss_all = []
    test_loss_all = []
    test_accuracy_all = []
    TFPN = []
    best = 0

    for epoch in range(1000):
        train_loss = 0
        train_num = 0
        test_error=0
        test_num=0
        for step,(b_x,b_y) in enumerate(train_loader):
            output = model(b_x)
            loss = loss_func(output,b_y) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)
 
        with torch.no_grad():
            total =0 
            correct=0
            TN,TP,FN,FP = 0,0,0,0
            model.eval()

            pre_y_test = model(test_xt)
            pre_y_test = np.array(pre_y_test.data.cpu().numpy())

            pre_y_train = model(train_xt)
            pre_y_train = np.array(pre_y_train.data.cpu().numpy())

            mae = mean_absolute_error(y_test,pre_y_test)
            correct +=(abs(pre_y_test-y_test)<5).sum().item()
            rate = 100*correct/len(y_test)
            
            if rate > best:
                best = rate
                state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                torch.save(state, f'./model/{folder}/kind_{kind}/{itera}/model_{epoch}_{rate}.pth')
                
                for i in range(len(y_test)):
                    if abs(pre_y_test[i]-y_test[i])<5 and y_test[i]==10:
                        TP += 1
                    if abs(pre_y_test[i]-y_test[i])<5 and y_test[i]==0:
                        TN += 1
                    if abs(pre_y_test[i]-y_test[i])>5 and y_test[i]==0:
                        FP += 1
                    if abs(pre_y_test[i]-y_test[i])>5 and y_test[i]==10:
                        FN += 1

                print('epoch',epoch,'accu:','%.2f'%best,'mae:','%.2f'%mae,'TP:',TP,'TN',TN,'FP',FP,'FN',FN,'mse:','%.2f'%train_loss_all[-1])
                best_string = str(['epoch',epoch,'accu:','%.2f'%best,'mae:','%.2f'%mae,'TP:',TP,'TN',TN,'FP',FP,'FN',FN,'mse:','%.2f'%train_loss_all[-1]])
                file = open(f'./model/{folder}/kind_{kind}/{itera}/TFPN.txt','a',encoding='utf8')
                file.write(best_string)
                file.write('\n')
                file.close
                

                index = np.argsort(y_test)
                plt.figure(figsize=(12,5))
                plt.plot(np.arange(len(y_test)),y_test[index],"r",label="Original Y")
                plt.scatter(np.arange(len(pre_y_test)),pre_y_test[index],s=3,c="b",label="Prediction")
                plt.legend(loc="upper left")
                plt.grid()
                plt.xlabel("index")
                plt.ylabel("Y")
#                 plt.show()
                plt.savefig(f'./model/{folder}/kind_{kind}/{itera}/test_{epoch}.png')
                plt.close()

                
                index = np.argsort(y_train)
                plt.figure(figsize=(12,5))
                pre_y =  model(train_xt)
                pre_y = pre_y.data.cpu().numpy()
                plt.plot(np.arange(len(y_train)),y_train[index],"r",label="Original Y")
                plt.scatter(np.arange(len(pre_y_train)),pre_y_train[index],s=3,c="b",label="Prediction")
                plt.legend(loc="upper left")
                plt.grid()
                plt.xlabel("index")
                plt.ylabel("Y")
#                 plt.show()
                plt.savefig(f'./model/{folder}/kind_{kind}/{itera}/train_{epoch}.png')
                plt.close()
            test_accuracy_all.append(rate)
            test_loss_all.append(mae)

        if epoch%200==0:
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, f'./model/{folder}/kind_{kind}/{itera}/model_{epoch}_{rate}.pth')
            TN,TP,FN,FP = 0,0,0,0
            for i in range(len(y_test)):
                if abs(pre_y_test[i]-y_test[i])<5 and y_test[i]==10:
                    TP += 1
                if abs(pre_y_test[i]-y_test[i])<5 and y_test[i]==0:
                    TN += 1
                if abs(pre_y_test[i]-y_test[i])>5 and y_test[i]==0:
                    FP += 1
                if abs(pre_y_test[i]-y_test[i])>5 and y_test[i]==10:
                    FN += 1

            print('epoch',epoch,'best_accu:','%.2f'%rate,'mae:','%.2f'%mae,'TP:',TP,'TN',TN,'FP',FP,'FN',FN,'mse:','%.2f'%train_loss_all[-1])
            file = open(f'./model/{folder}/kind_{kind}/{itera}/TFPN.txt','a',encoding='utf8')
            string = str(['epoch',epoch,'accu:','%.2f'%rate,'mae:','%.2f'%mae,'TP:',TP,'TN',TN,'FP',FP,'FN',FN,'mse:','%.2f'%train_loss_all[-1]])
            file.write(string)
            file.write('\n')
            file.close()

            index = np.argsort(y_test)
            plt.figure(figsize=(12,5))
            plt.plot(np.arange(len(y_test)),y_test[index],"r",label="Original Y")
            plt.scatter(np.arange(len(pre_y_test)),pre_y_test[index],s=3,c="b",label="Prediction")
            plt.legend(loc="upper left")
            plt.grid()
            plt.xlabel("index")
            plt.ylabel("Y")
#                 plt.show()
            plt.savefig(f'./model/{folder}/kind_{kind}/{itera}/test_{epoch}.png')
            plt.close()
            
            index = np.argsort(y_train)
            plt.figure(figsize=(12,5))
            plt.plot(np.arange(len(y_train)),y_train[index],"r",label="Original Y")
            plt.scatter(np.arange(len(pre_y_train)),pre_y_train[index],s=3,c="b",label="Prediction")
            plt.legend(loc="upper left")
            plt.grid()
            plt.xlabel("index")
            plt.ylabel("Y")
#                 plt.show()
            plt.savefig(f'./model/{folder}/kind_{kind}/{itera}/train_{epoch}.png')
            plt.close()

    file = open(f'./model/{folder}/kind_{kind}/TFPN.txt','a',encoding='utf8')
    file.write(f'{itera}:')
    file.write(best_string)
    file.write('\n')
    file.close

    plt.figure(figsize=(10,6))
    plt.plot(train_loss_all,"r",label="Train loss")
    plt.legend()
    plt.grid()
    plt.xlabel("eopch")
    plt.ylabel("loss")
#     plt.show()
    plt.savefig(f'./model/{folder}/kind_{kind}/{itera}/train_loss')
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(test_loss_all,"r",label="Test loss")
    plt.legend()
    plt.grid()
    plt.xlabel("eopch")
    plt.ylabel("loss")
#     plt.show()
    plt.savefig(f'./model/{folder}/kind_{kind}/{itera}/test_loss')
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(test_accuracy_all,"r",label="Test Accuracy")
    plt.legend()
    plt.grid()
    plt.xlabel("eopch")
    plt.ylabel("accuracy")
#     plt.show()
    plt.savefig(f'./model/{folder}/kind_{kind}/{itera}/accuracy')
    plt.close()
    
    index = np.argsort(y_test)
    plt.figure(figsize=(12,5))
    plt.plot(np.arange(len(y_test)),y_test[index],"r",label="Original Y")
    plt.scatter(np.arange(len(pre_y_test)),pre_y_test[index],s=3,c="b",label="Prediction")
    plt.legend(loc="upper left")
    plt.grid()
    plt.xlabel("index")
    plt.ylabel("Y")
#     plt.show()
    plt.savefig(f'./model/{folder}/kind_{kind}/{itera}/pre_test')
    plt.close()
    
    index = np.argsort(y_train)
    plt.figure(figsize=(12,5))
    plt.plot(np.arange(len(y_train)),y_train[index],"r",label="Original Y")
    plt.scatter(np.arange(len(pre_y_train)),pre_y_train[index],s=3,c="b",label="Prediction")
    plt.legend(loc="upper left")
    plt.grid()
    plt.xlabel("index")
    plt.ylabel("Y")
#     plt.show()
    plt.savefig(f'./model/{folder}/kind_{kind}/{itera}/pre_train')
    plt.close()
    
    f = open(f'./model/{folder}/kind_{kind}/{itera}/Result.txt','w',encoding='utf8')
    f.write(str(train_loss_all))
    f.write('\n')
    f.write(str(test_loss_all))
    f.write('\n')
    f.write(str(test_accuracy_all))
    f.write('\n')
    f.close()


#import data from database
f = open('info_train.txt','r',encoding='utf8')
Info = f.readlines()
f.close()
Info_2 = []
for item in Info:
    entry = eval(item)
    Info_2.append(entry)

data = []
label = []
for info in Info_2:
    try:
        if info['score']==10 or info['score']==0:
            if float(info['Composition C1'])+float(info['Composition C2']) == 100.0:
                vector = info['SMILE C1']+info['SMILE C2']+[float(info['Composition C1']),float(info['Composition C2'])]
                data.append(vector)
                label.append(info['score'])
        if len(vector)!=4098:
            print(info['Sample ID:'],len(vector))
    except:
        continue
print('len(label):',len(label))
print('len(data):',len(data))


# For Balanced Division
data_set = [['BD001995', 'BD001566', 'BD000642', 'BD000187', 'BD000172', 'BD000368', 'BD000639', 'BD000764', 'BD001702', 'BD001609', 'BD000295', 'BD000299', 'BD002129', 'BD000636', 'BD000418', 'BD000782', 'BD000303', 'BD000442', 'BD001527', 'BD001988', 'BD000485', 'BD000705', 'BD000212', 'BD000153', 'BD000709', 'BD000288', 'BD001576', 'BD000394', 'BD000437', 'BD001520', 'BD000744', 'BD000081', 'BD000367', 'BD000259', 'BD001524', 'BD000438', 'BD000273', 'BD001911', 'BD000902', 'BD000280', 'BD000362', 'BD001636', 'BD000500', 'BD000766', 'BD001565', 'BD000717', 'BD002133', 'BD001508', 'BD000841', 'BD001042', 'BD001871', 'BD000392', 'BD001680', 'BD000600', 'BD000207', 'BD000289', 'BD000342', 'BD002250', 'BD000986', 'BD000695', 'BD000298', 'BD001496', 'BD000176', 'BD000790', 'BD000192', 'BD000089', 'BD000681', 'BD000602', 'BD000794', 'BD001481', 'BD000852', 'BD002255', 'BD000488', 'BD000583', 'BD000748', 'BD000868', 'BD000425', 'BD002125', 'BD000918', 'BD000644', 'BD000687', 'BD000196', 'BD000173', 'BD000307', 'BD001983', 'BD000302', 'BD000354', 'BD000169', 'BD000112', 'BD000366', 'BD001498', 'BD000504', 'BD001252', 'BD001677', 'BD000085', 'BD000746', 'BD000278', 'BD001696', 'BD000421', 'BD000152', 'BD000431', 'BD000811', 'BD000154', 'BD000503', 'BD000716', 'BD001542', 'BD000286', 'BD000183', 'BD000251', 'BD000111', 'BD000386', 'BD000903', 'BD000641', 'BD001523', 'BD000190', 'BD000194','BD001995', 'BD001566', 'BD000642', 'BD000187', 'BD000172', 'BD000368', 'BD000764', 'BD000418', 'BD000442', 'BD001527', 'BD000437', 'BD001520', 'BD000259', 'BD001524', 'BD000438', 'BD000362', 'BD000500', 'BD000766', 'BD002133', 'BD001508', 'BD001042', 'BD001871', 'BD001680', 'BD000600', 'BD000207', 'BD002250', 'BD000986', 'BD000790', 'BD000192', 'BD000089', 'BD000602', 'BD001481', 'BD000852', 'BD000748', 'BD000868', 'BD000918', 'BD000687', 'BD000196', 'BD000112', 'BD001252', 'BD000085', 'BD000746', 'BD000152', 'BD000154', 'BD000503', 'BD000183', 'BD000111', 'BD000903', 'BD000641', 'BD001523','BD000642','BD000442','BD001995','BD001566'],
['BD000715', 'BD000732', 'BD000098', 'BD000377', 'BD000206', 'BD000813', 'BD002114', 'BD001518', 'BD000094', 'BD000150', 'BD000582', 'BD000371', 'BD000439', 'BD000926', 'BD000668', 'BD001497', 'BD000332', 'BD000665', 'BD000309', 'BD000158', 'BD000253', 'BD000420', 'BD001996', 'BD002074', 'BD000737', 'BD000213', 'BD000838', 'BD000505', 'BD001492', 'BD001567', 'BD000423', 'BD000561', 'BD001966', 'BD000205', 'BD000925', 'BD000271','BD000582'],
['BD000942', 'BD000491', 'BD000422', 'BD002047', 'BD001634', 'BD000181', 'BD000365', 'BD000211', 'BD000941', 'BD001695', 'BD000542', 'BD001689', 'BD000258', 'BD000218', 'BD000287', 'BD000328', 'BD000762', 'BD000531', 'BD000270', 'BD000188', 'BD000846', 'BD000697', 'BD000274', 'BD000497', 'BD001483', 'BD000638', 'BD000723', 'BD000640', 'BD000415', 'BD000290', 'BD000763', 'BD000869', 'BD001026', 'BD000924', 'BD001503', 'BD000428', 'BD000256', 'BD000786', 'BD000885','BD000942', 'BD000491', 'BD002047', 'BD000181', 'BD000365', 'BD000211', 'BD000542', 'BD000762', 'BD000846', 'BD000497', 'BD001483', 'BD000640', 'BD001026', 'BD000428', 'BD000786', 'BD000885','BD000542']]

#check the balance of dataset
x_balTrain = []
y_balTrain = []
for BID in data_set[0]:
    for info in Info_2:
        try:
            if info['score']==10 or info['score']==0:
                if info['Blend ID(BDID):'] ==BID :
                    vector = info['SMILE C1']+info['SMILE C2']+[float(info['Composition C1']),float(info['Composition C2'])]
                    x_balTrain.append(vector)
                    y_balTrain.append(info['score'])
        except:
            continue 
for BID in data_set[1]:
    for info in Info_2:
        try:
            if info['score']==10 or info['score']==0:
                if info['Blend ID(BDID):'] ==BID :
                    vector = info['SMILE C1']+info['SMILE C2']+[float(info['Composition C1']),float(info['Composition C2'])]
                    x_balTrain.append(vector)
                    y_balTrain.append(info['score'])
        except:
            continue   
print('x_balTrain:',np.shape(x_balTrain),'y_balTrain:',np.mean(y_balTrain))

x_baltrain = []
y_baltrain = []
for BID in data_set[0]:
    for info in Info_2:
        try:
            if info['score']==10 or info['score']==0:
                if info['Blend ID(BDID):'] ==BID :
                    vector = info['SMILE C1']+info['SMILE C2']+[float(info['Composition C1']),float(info['Composition C2'])]
                    x_baltrain.append(vector)
                    y_baltrain.append(info['score'])
        except:
            continue 
print('x_baltrain:',np.shape(x_baltrain),'y_baltrain:',np.mean(y_baltrain))

x_balvalid = []
y_balvalid = []
for BID in data_set[1]:
    for info in Info_2:
        try:
            if info['score']==10 or info['score']==0:
                if info['Blend ID(BDID):'] ==BID :
                    vector = info['SMILE C1']+info['SMILE C2']+[float(info['Composition C1']),float(info['Composition C2'])]
                    x_balvalid.append(vector)
                    y_balvalid.append(info['score'])
        except:
            continue   
print('x_balvalid:',np.shape(x_balvalid),'y_balvalid:',np.mean(y_balvalid))

x_baltest = []
y_baltest = []
for BID in data_set[2]:
    for info in Info_2:
        try:
            if info['score']==10 or info['score']==0:
                if info['Blend ID(BDID):'] ==BID :
                    vector = info['SMILE C1']+info['SMILE C2']+[float(info['Composition C1']),float(info['Composition C2'])]
                    x_baltest.append(vector)
                    y_baltest.append(info['score'])
        except:
            continue   
print('x_baltest',np.shape(x_baltest),'y_baltest',np.mean(y_baltest))


# For Random Division
x_ranTrain,x_rantest,y_ranTrain,y_rantest = train_test_split(data,label,train_size=0.8,random_state=1)
x_rantrain,x_ranvalid,y_rantrain,y_ranvalid = train_test_split(x_ranTrain,y_ranTrain,train_size=0.8,random_state=1)
print('x_rantrain:',np.shape(x_rantrain),'y_rantrain:',np.mean(y_rantrain))
print('x_ranvalid:',np.shape(x_ranvalid),'y_ranvalid:',np.mean(y_ranvalid))
print('x_rantest',np.shape(x_rantest),'y_rantest',np.mean(y_rantest))


itera = 10
cuda = 0
lrate = 0.0001
print('input kind')
kind = int(input())
for itera in range(10):

    for kind in range(8):

        if kind ==0:
            h1,h2=300,300
        elif kind==1:
            h1,h2 = 300,200
        elif kind==2:
            h1,h2 = 500,500
        elif kind==3:
            h1,h2 = 100,500
        elif kind==4:
            h1,h2 = 300,100
        elif kind==5:
            h1,h2 = 200,500
        elif kind==6:
            h1,h2 = 400,400
        else:
            h1,h2 = 100,100

        H1 = [h1,h1,h1,h1,200,200,100]
        H2 = [h2,h2,h2,h2,200,200,100]

        folder1 = f'ran/test;{h1};{lrate}'
        if os.path.exists(f'./model/{folder1}') is False:
            os.mkdir(f'./model/{folder1}')
        print(f'This training uses {kind} as model, h1={h1},lrate={lrate},itera={itera}')
        if os.path.exists(f'./model/{folder1}/kind_{kind}') is False:
            os.mkdir(f'./model/{folder1}/kind_{kind}')
        if os.path.exists(f'./model/{folder1}/kind_{kind}/{itera}') is False:
            os.mkdir(f'./model/{folder1}/kind_{kind}/{itera}')

        folder2 = f'bal/test;{h2};{lrate}'
        if os.path.exists(f'./model/{folder2}') is False:
            os.mkdir(f'./model/{folder2}')
        print(f'This training uses {kind} as model, h1={h2},lrate={lrate},itera={itera}')
        if os.path.exists(f'./model/{folder2}/kind_{kind}') is False:
            os.mkdir(f'./model/{folder2}/kind_{kind}')
        if os.path.exists(f'./model/{folder2}/kind_{kind}/{itera}') is False:
            os.mkdir(f'./model/{folder2}/kind_{kind}/{itera}')

        start = time.time()
        whole_net(x_ranTrain,y_ranTrain,x_rantest,y_rantest,folder1,lrate,H1,kind,itera,cuda)
        stop = time.time()
        print('%.2f'%((stop-start)/60),'min')

        start = time.time()
        whole_net(x_balTrain,y_balTrain,x_baltest,y_baltest,folder2,lrate,H2,kind,itera,cuda)
        stop = time.time()
        print('%.2f'%((stop-start)/60),'min')
