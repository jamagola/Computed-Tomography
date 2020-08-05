# -*- coding: utf-8 -*-
"""
Created on Thu Jul  20 10:09:00 2020

@Last modified by: Golam
"""

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from multiprocessing import Process, Manager
from torch.optim import lr_scheduler
import os
import pickle
import random
import numpy as np
import os
import argparse
from model2D import *

# Change to correct directory
target=torch.from_numpy(np.load('/usr/WS2/jaman1/summer2020/target2D.npy'))
CTimage=torch.from_numpy(np.load('/usr/WS2/jaman1/summer2020/CTdata2D.npy'))
with open('/usr/WS2/jaman1/summer2020/lotmap2D.pkl','rb') as f:
 lotmap=pickle.load(f)
print('Processing...\n')
# feed e.g. python stress2D.py --leaveoutlots 'AM','D'
parser = argparse.ArgumentParser()
parser.add_argument('--leaveoutlots')
args = parser.parse_args()
leaveouts = args.leaveoutlots.split(',')

out_file_path=r'/usr/workspace/jaman1/summer2020/'
model_file_name=out_file_path

## Parameters ##
torch.manual_seed(0)
random.seed(0)
image_size=300 # 300 by 300 from preprocess
batch_size=26  # If layer step size is 3 (1/3 of images), each lot will carry 260 images. 
droprate = 0.0
widenfactor = 2
depth = 28
epochs = 99 #######################
num_epochs = epochs
nepochs=epochs
lr=0.0001
lotlen=260 # This will vary depending on number of images taken from each lot
imageSize=image_size
batch=batch_size
channel=1

for gpuindex, leaveout in enumerate(leaveouts):

    print('leaving out ',leaveout,' on gpuindex', gpuindex)
    out_file=open(out_file_path+'processed2D_'+leaveout+'.out','a+')
    device = torch.device('cuda:'+str(gpuindex))
    cudnn.benchmark = True    
    leaveOut=leaveout
    index=lotmap.index(leaveOut)

    test_=CTimage[index:index+lotlen,:,:,:]
    test_=test_.reshape(-1,channel,imageSize,imageSize)
    desire=target[index:index+lotlen]

    CTdata_=torch.cat((CTimage[:index,:,:,:],CTimage[index+lotlen:,:,:,:]))
    CTdata_=CTdata_.reshape(-1,channel,imageSize,imageSize)
    target_=torch.cat((target[:index,:],target[index+lotlen:,:]))
    devi=np.std(target_.numpy())
    meu=np.mean(target_.numpy())
    target_=(target_-meu)/devi #Normalization
    
    train=torch.utils.data.TensorDataset(CTdata_, target_)
    test=torch.utils.data.TensorDataset(test_, desire)
    ts_load=torch.utils.data.DataLoader(test, batch_size=int(lotlen/10), shuffle=True)
    tr_load=torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True) 
	
    model = WideResNet(depth=depth, widen_factor=widenfactor,drop_rate=droprate)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    for aa in model.modules():
        if isinstance(aa, nn.BatchNorm2d):
            aa.momentum = 1e-2
			
    print(device)
    print(model)
    out_file.write(str(model))
	
	
    criterion=nn.MSELoss()
    optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # Clear gradients
    optimizer.zero_grad()
    mini=1 # Number of mini batches from gradient accumulation
    count = 0
    loss_=0
    threshold=0.1
	
    for epoch in range(num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        running_loss = []
        # Iterate over data.
        for inputs, labels in tr_load:
            inputs = inputs.to(device)
            labels = labels.to(device)
            #debug
            #print(inputs.shape)
            # zero the parameter gradients
                    
            outputs = model(inputs).view(-1,1)
            labels = labels.view(-1,1).float()
            loss = criterion(outputs, labels)           
            if count==0:
                loss_=loss.data
            if loss.data/loss_ <= threshold:
              lr=lr*1 ##################if 1: Not doing any adaptive LR######
              optimizer=torch.optim.SGD(model.parameters(), lr=lr)
              loss_=loss.data
              print(f'new lr= {lr}') ##########################################
    
            # Calculating gradients
            loss.backward()
            if (count+1)%mini==0: #Every # mini- batches(each of size batch), update weights
               loss=loss/mini
               # Update parameters
               optimizer.step()
               # Clear gradients
               optimizer.zero_grad()
            
            count += 1
            if count % mini == 0:
                #Print Loss
                print('Epoch: {} Iteration: {}  Loss: {}'.format(epoch, count, loss.data)) #####
            
        out_file.write('\nEpoch: {}  Loss: {}'.format(epoch, loss.data))	## Report after each epoch only		      


# Test
################################################################################
    
    for i, (images, labels) in enumerate(ts_load):
        images = images.to(device)
        labels = labels.to(device)
        model.eval()
        outputs = model(images).view(-1,1)
        outputs=outputs*devi+meu
        labels=labels.view(-1,1).float()
        #labels=labels*devi+meu
        print(images.shape)
        print('\nEvaluating leftout lot:\n')
        print(outputs.mean())
        print('\nActual:\n')
        print(labels.mean())
        print('\nPE: \n')
        print(abs((outputs.mean())-(labels.mean()))*100/(labels.mean()))
        out_file.write('\nEvaluating leftout lot:\n')
        out_file.write(str(outputs.mean()))
        out_file.write('\nActual:\n')
        out_file.write(str(labels.mean()))
        out_file.write('\nPE:\n')
        out_file.write(str(abs((outputs.mean())-(labels.mean()))*100/(labels.mean())))
        break

    mPath=model_file_name+'model2D_'+leaveOut+'.pth'
    torch.save(model, mPath)

    out_file.write('\nDone!\n')
    out_file.close()
print('Done\n')    
