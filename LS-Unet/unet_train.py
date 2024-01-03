#https://github.com/DiegoLeon96/Neural-Speech-Dereverberation

import time, os, pickle, argparse
import numpy as np
import random
import psutil
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
import sys
from convolutional_models import UNetRev, LateSupUnet
import pdb


iter_num = 100

def train(train_loader, path, device):
    #os.system("mkdir " + path)
    if not os.path.exists(path):
        os.makedirs(path)
    model = LateSupUnet(n_channels=1, bilinear=False).to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5,0.999))    
    start = time.time()
    loss_value = []  
    outputs = [] 
    for epoch in range(1, iter_num + 1):      
        run_loss=0
        for batch_idx, (xt, yt) in enumerate(train_loader):            
            xt, yt = xt.to(device), yt.to(device)          
            xa = xt.unsqueeze(0)
            ya = yt.squeeze(0)
            
            #pdb.set_trace()
            optimizer.zero_grad()                
            output = model(xa)
            loss = criterion(output, ya)
            #loss = F.mse_loss(output, yb)
            loss.backward()
            optimizer.step()                  
            run_loss+=loss.item()
            if batch_idx % 100 == 0:
                torch.save(model, path+'model.pt')
                torch.save(model.state_dict(), path+'parameter.pth')
                print('Train Epoch: [{}] [{}/{} ({:.0f}%)] Time: [{:.6f}] \nLoss: [{:.6f}]'.format(
                    epoch, batch_idx , len(train_loader.dataset),               
                    100. * batch_idx / len(train_loader), time.time() - start, loss.item()))
        #pdb.set_trace()       
        loss_value.append(((run_loss/len(train_loader.dataset))))                 
    plt.plot(loss_value, 'r')
    plt.title("unet pytorch Loss") # title
    plt.ylabel("loss") # y label
    plt.xlabel("epoch") # x label
    #y_major_locator=MultipleLocator(0.05)#把y軸區間設定為0.05的倍數
    
    #ax=plt.gca()
    #ax.yaxis.set_major_locator(y_major_locator)
    #plt.ylim(0, 0.6)
    plt.show()
    plt.savefig(path+"unet pytorch Loss.png")
    

  




def test(test_loader, path, result, device):
    model = LateSupUnet(n_channels=1, bilinear=False).to(device)
    model.load_state_dict(torch.load(path+'parameter.pth',map_location=device))
    #model.load_state_dict(torch.load(path+'parameter.pth', map_location=lambda storage, loc: storage.cuda(0)))
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    i = 0
    preds = []
    with torch.no_grad():
        for batch_idx, (xt, yt) in enumerate(test_loader):
            xt, yt = xt.to(device), yt.to(device)
            xa = xt.unsqueeze(0)
            ya = yt.squeeze(0)             
            pred = model(xa)
            preds.append(pred)
            loss = criterion(pred, ya).item()
            i+=1
            print('Loss[{}]: {:.6f}'.format(i, loss))
            test_loss += loss
    test_loss /= len(test_loader.dataset)
    print('\n\nAverage Loss: {:.6f}\n\n'.format(test_loss))
    #print('\ndata: {}'.format(preds))        
    os.chdir(path)
    #os.system("mkdir " + data)
    if not os.path.exists(result):
        os.makedirs(result)
    with open(result + '/pred.txt', 'wb') as fp:
        torch.save(preds, fp)
        
    #with open(data+'/pred.txt', 'rb') as fp:
        # d = torch.load(fp)
        #print('{}' .format(d))
    os.chdir('../')        
        




















            
            
            
            











