# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 17:26:41 2020

@Last modified by: Golam
"""

import numpy as np
import os
import argparse
from multiprocessing import Process, Manager
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import *
import torch.backends.cudnn as cudnn
import cv2
import torchvision
import pickle
from model3D import *

# Change to correct directory
target=torch.from_numpy(np.load('/usr/WS2/jaman1/summer2020/target3D.npy'))
CTdata=torch.from_numpy(np.load('/usr/WS2/jaman1/summer2020/CTdata3D.npy'))
with open('/usr/WS2/jaman1/summer2020/lotmap3D.pkl','rb') as f:
 lotmap=pickle.load(f)
print('Processing...\n')
# feed e.g. python stress3D.py --leaveoutlots 'AM','D'
parser = argparse.ArgumentParser()
parser.add_argument('--leaveoutlots')
args = parser.parse_args()
leaveouts = args.leaveoutlots.split(',')

out_file_path=r'/usr/workspace/jaman1/summer2020/'
model_file_name=out_file_path
lotlen=1 # 8 rotations+flip : keep 1 if no data augmentation otherwise 8
w_=300
h_=300
layer_=260

for gpuindex, leaveout in enumerate(leaveouts):

    print('leaving out ',leaveout,' on gpuindex', gpuindex)
    out_file=open(out_file_path+'processed_'+leaveout+'.out','a+')
    device = torch.device('cuda:'+str(gpuindex))
    cudnn.benchmark = True
    leaveOut=leaveout
    index=lotmap.index(leaveOut)
    test_=CTdata[index:index+lotlen,:,:,:,:]
    #test_=test_.unsqueeze(0)
    desire=target[index:index+lotlen]
    
    CTdata_=torch.cat((CTdata[:index,:,:,:,:],CTdata[index+lotlen:,:,:,:,:]))
    target_=torch.cat((target[:index,:],target[index+lotlen:,:]))
    devi=np.std(target_.numpy())
    meu=np.mean(target_.numpy())
    target_=(target_-meu)/devi
    
    batch=1 # Batch normalization requires > 1
    train=torch.utils.data.TensorDataset(CTdata_, target_)
    test=torch.utils.data.TensorDataset(test_, desire)
    ts_load=torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)
    tr_load=torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True)
    
    nepochs=100
    model=CNNModel(False) # Default batch norm: True
    model=model.to(device) ################################################
    print(device)
    print(model)
    out_file.write(str(model))
    error=nn.MSELoss()
    #Parameters #####################################################
    lr=0.00001  #####################################################
    optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99) #############
    # Clear gradients
    optimizer.zero_grad()
    mini=4 # Number of mini batches from gradient accumulation
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    loss_=0
    threshold=0.1
    for epoch in range(nepochs):
        for i, (images, labels) in enumerate(tr_load):
		
            train_ = Variable(images.view(batch,1,layer_,w_,h_))#260 for layer step 3
            labels = Variable(labels)
            train_ = train_.to(device)
            labels = labels.to(device)

            # Forward propagation
            outputs = model(train_)
            # Calculate MSE loss
            loss = error(outputs, labels)
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
                # Print Loss
                print('Epoch: {} Iteration: {}  Loss: {}'.format(epoch, count, loss.data)) #####
               
        out_file.write('\nEpoch: {}  Loss: {}'.format(epoch, loss.data))
    
    #output_mean=0
    #pe_mean=0
    for i, (images, labels) in enumerate(ts_load): #change to ts_load if not there!!!!!!!!!!!!!!!
        test_ = Variable(images.view(1,1,layer_,w_,h_)) #(batch,channel,layers,w,h)
        test_ = test_.to(device)
        labels = Variable(labels)
        labels = labels.to(device)
        model.eval()
        outputs = model(test_)
        outputs=outputs*devi+meu
        #output_mean=output_mean+outputs
        #labels=labels*devi+meu
        out_file.write('\nEvaluating leftout lot:\n')
        out_file.write(str(outputs))
        out_file.write('\nActual:\n')
        out_file.write(str(labels))
        out_file.write('\nPE:\n')
        out_file.write(str(((outputs)-(labels))*100/(labels)))
        break
    #output_mean=output_mean/lotlen
    #pe_mean=(output_mean-labels)*100/labels
    #out_file.write('\nMean PE:\n')
    #out_file.write(str(pe_mean))   

    mPath=model_file_name+'model3D_'+leaveOut+'.pth'
    torch.save(model, mPath)

    out_file.write('\nDone!\n')
    out_file.close()
print('Done\n')    
