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

nclass=1

class CNNModel(nn.Module):
  def __init__(self, batchNorm=True):
    super(CNNModel, self).__init__()
    self.batchNorm=batchNorm

    self.conv_layer1=self._conv_layer_set(1,32)
    self.conv_layer2=self._conv_layer_set2(32,64)
    self.conv_layer3=self._conv_layer_set2(64,64)
    self.conv_layer4=self._conv_layer_set2(64,64)
    self.conv_layer5=self._conv_layer_set2(64,64)
    self.conv_layer6=self._conv_layer_set3(64,64)
    self.fc1=nn.Linear((2**3)*64, 128) #Make adaptive:2**3 came from calculation: in Conv or minMaxPool ((image_space)-(kernel_size)+1)/stride(steps)
    self.fc2=nn.Linear(128, nclass)

    self.relu=nn.LeakyReLU()
    self.rrelu=nn.ReLU()
    self.sigm=nn.Sigmoid()
    self.batch = nn.BatchNorm1d(128)
    self.batch2 = nn.BatchNorm1d(1)
    self.drop=nn.Dropout(p=0.15)
    self.drop2=nn.Dropout(p=0.5)

  def _conv_layer_set(self, in_c, out_c):
    conv= nn.Sequential(nn.Conv3d(in_c, out_c, kernel_size=(3,3,3),stride=(1,1,1), padding=0), nn.LeakyReLU(), nn.MaxPool3d((2,2,2)) )
    return conv

  def _conv_layer_set2(self, in_c, out_c):
    conv= nn.Sequential(nn.Conv3d(in_c, out_c, kernel_size=(3,3,3),stride=(1,1,1), padding=0), nn.LeakyReLU(), nn.MaxPool3d((2,2,2)) )
    return conv

  def _conv_layer_set3(self, in_c, out_c):
    conv= nn.Sequential(nn.Conv3d(in_c, out_c, kernel_size=(3,3,3),stride=(1,1,1), padding=0), nn.LeakyReLU(), nn.AdaptiveMaxPool3d((2,2,2)) ) ######
    return conv

  def forward(self, x):
    out=self.conv_layer1(x)
    out=self.conv_layer2(out)
    out=self.conv_layer3(out)
    out=self.conv_layer4(out)
    out=self.conv_layer5(out)
    out=self.conv_layer6(out)
    out=out.view(out.size(0),-1)
    out=self.fc1(out)
    out=self.relu(out)
    if self.batchNorm==True:
        out=self.batch(out)
    else:
        out=out
    out=self.drop(out)
    out=self.fc2(out)
    out=self.rrelu(out)
    #out=self.relu(out)
    #out=self.batch2(out)
    #out=self.sigm(out)
    return out
