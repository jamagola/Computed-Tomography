# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:07:02 2020

@Last modified by: Golam
"""

import os
import cv2
import torch
import torchvision
import numpy as np
import pickle

peak_stress = {
    'A': 1518.0,
    'AJ': 1904.5,
    'AK': 1645.0,
    'AL': 1586.5,
    'AM': 2302.5,
    'AN': 1793.0,
    'AO': 1141.5,
    'AP': 1481.5,
    'AQ': 1915.0,
    'AR': 1370.5,
    'AS': 1780.0,
    'AT': 1620.0,
    'AU': 1875.0,
    'AV': 1800.0,
    'AW': 1589.0,
    'AX': 1734.0,
    'AY': 1709.5,
    'AZ': 1675.0,
    'B': 1233.5,
    'C': 987.5,
    'D': 1027.5,
    'E': 1023.5,
    'F': 1935.5,
    'H': 908.5,
    'I': 924.5,
    'J': 1688.0,
    'K': 1480.0,
    'L': 2256.5,
    'M': 2167.0,
    'N': 822.0,
    'O': 2131.0,
    'P': 1932.0,
    'Q': 2148.5,
    'R': 1005.0,
    'S': 1469.5,
    'T': 1701.0,
    'U': 1324.0,
    'V': 1915.0,
    'W': 1222.0,
    'X': 1647.5,
    'Y': 1784.5
}

peak_strain = {
    'A': 8561.0,
    'AJ': 12738.0,
    'AK': 13065.5,
    'AL': 9710.5,
    'AM': 15256.5,
    'AN': 14917.0,
    'AO': 13049.0,
    'AP': 11215.0,
    'AQ': 13175.5,
    'AR': 7724.0,
    'AS': 16422.5,
    'AT': 15784.5,
    'AU': 12537.0,
    'AV': 10407.0,
    'AW': 8691.0,
    'AX': 13991.5,
    'AY': 13398.0,
    'AZ': 13733.0,
    'B': 9132.5,
    'C': 14290.5,
    'D': 14105.0,
    'E': 12770.0,
    'F': 9084.5,
    'H': 9736.0,
    'I': 15916.5,
    'J': 11055.0,
    'K': 13861.5,
    'L': 10886.0,
    'M': 10591.5,
    'N': 15703.0,
    'O': 12923.5,
    'P': 9027.5,
    'Q': 13181.5,
    'R': 10881.5,
    'S': 10604.5,
    'T': 9230.0,
    'U': 13978.0,
    'V': 8647.0,
    'W': 17363.5,
    'X': 8987.5,
    'Y': 9352.0
}

slope = {
    'A': 0.59,
    'AJ': 0.535,
    'AK': 0.47,
    'AL': 0.53,
    'AM': 0.57,
    'AN': 0.455,
    'AO': 0.325,
    'AP': 0.485,
    'AQ': 0.535,
    'AR': 0.435,
    'AS': 0.445,
    'AT': 0.415,
    'AU': 0.555,
    'AV': 0.53,
    'AW': 0.515,
    'AX': 0.48,
    'AY': 0.49,
    'AZ': 0.445,
    'B': 0.52,
    'C': 0.335,
    'D': 0.395,
    'E': 0.38,
    'F': 0.695,
    'H': 0.425,
    'I': 0.295,
    'J': 0.55,
    'K': 0.395,
    'L': 0.705,
    'M': 0.645,
    'N': 0.23,
    'O': 0.6,
    'P': 0.685,
    'Q': 0.63,
    'R': 0.33,
    'S': 0.47,
    'T': 0.66,
    'U': 0.39,
    'V': 0.625,
    'W': 0.225,
    'X': 0.62,
    'Y': 0.645
}

percent_anomalous = {
    'I': 0.00436,
    'R': 0.00189,
    'D': 0.00415,
    'AO': 0.00600,
    'AR': 0.05289,
    'U': 0.00364,
    'W': 0.00649,
    'AP': 0.00306,
    'AW': 0.02800,
    'AT': 0.00261,
    'X': 0.00395,
    'AZ': 0.01177,
    'AX': 0.00503,
    'T': 0.00396,
    'AS': 0.00410,
    'AV': 0.01489,
    'AU': 0.00550,
    'AQ': 0.00872,
    'V': 0.01300,
    'F': 0.00106,
    'AM': 0.00936
}

percent_cracks = {
    'I': 0.00431,
    'R': 0.00186,
    'D': 0.00389,
    'AO': 0.0059,
    'AR': 0.05273,
    'U': 0.00335,
    'W': 0.00646,
    'AP': 0.00252,
    'AW': 0.02797,
    'AT': 0.00255,
    'X': 0.00391,
    'AZ': 0.01175,
    'AX': 0.00498,
    'T': 0.00249,
    'AS': 0.00409,
    'AV': 0.01486,
    'AU': 0.00549,
    'AQ': 0.00867,
    'V': 0.01291,
    'F': 0.00071,
    'AM': 0.00899
}

percent_foreigns = {
    'I': 0.0000491,
    'R': 3.59E-5,
    'D': 0.00025874,
    'AO': 9.66E-5,
    'AR': 0.000165608,
    'U': 0.000290098,
    'W': 3.46E-5,
    'AP': 0.000542298,
    'AW': 3.28E-5,
    'AT': 4.61E-5,
    'X': 3.97E-5,
    'AZ': 2.55E-5,
    'AX': 4.74E-5,
    'T': 0.001464138,
    'AS': 1.12E-5,
    'AV': 2.09E-5,
    'AU': 7.86E-6,
    'AQ': 4.88E-5,
    'V': 8.92E-5,
    'F': 0.000354675,
    'AM': 0.000366845
}

anomalies = {
    'I': 213,
    'R': 321,
    'D': 563,
    'AO': 183,
    'AR': 1911,
    'U': 1077,
    'W': 526,
    'AP': 1003,
    'AW': 1478,
    'AT': 360,
    'X': 327,
    'AZ': 375,
    'AX': 368,
    'T': 4166,
    'AS': 83,
    'AV': 318,
    'AU': 193,
    'AQ': 403,
    'V': 506,
    'F': 355,
    'AM': 1094
}

cracks = {
    'I': 176,
    'R': 264,
    'D': 285,
    'AO': 127,
    'AR': 1505,
    'U': 775,
    'W': 472,
    'AP': 515,
    'AW': 1397,
    'AT': 304,
    'X': 295,
    'AZ': 344,
    'AX': 312,
    'T': 389,
    'AS': 64,
    'AV': 264,
    'AU': 171,
    'AQ': 339,
    'V': 361,
    'F': 198,
    'AM': 225
}

objects = {
    'I': 37,
    'R': 57,
    'D': 278,
    'AO': 56,
    'AR': 406,
    'U': 302,
    'W': 54,
    'AP': 488,
    'AW': 81,
    'AT': 56,
    'X': 32,
    'AZ': 31,
    'AX': 56,
    'T': 3777,
    'AS': 19,
    'AV': 54,
    'AU': 22,
    'AQ': 64,
    'V': 145,
    'F': 157,
    'AM': 869
}

out_file_path=r'/usr/workspace/jaman1/summer2020/'
path=r'/usr/WS2/jaman1/store/Neat_CT_Data'
out_file=open(out_file_path+'preprocess3D.out','a+')
folder=os.listdir(path)
config_=[0,0,0] #17,19,20
len_=len(os.listdir(path+'/'+folder[0]))
step=3; #layers step (for downsampling layers)
state0=True
state1=True
image=[]
image90=[]
image180=[]
image270=[]
data=[]
target=[]
lotmap=[]
mew=[0,0,0] #17,19,20
sigma=[0,0,0] #17,19,20
count=0
w_=300
h_=300
temp_circle=[]
blur=True
data_aug=False #disabled data augmentation

if data_aug:
  stack_=8
else:
  stack_=1

target_param=percent_anomalous #change target here

for i in folder:
  if len(os.listdir(path+'/'+i)) < 800: #800: #Number of images from 803-827(remove 5 if there!)
    continue
  if i.split('_')[1] in target_param.keys():
    out_file.write('Loading: {}\n'.format(i))
    print('Loading: {}\n'.format(i))
  else:
    continue

  count=count+1

  files=os.listdir(path+'/'+i)
  files=sorted(files)

  #for j in np.arange(0,len(os.listdir(path+'/'+i)),step):
  for j in np.arange(20,799,step):
    source=path+'/'+i+'/'+files[j]
    # Masking
    temp = cv2.imread(source, cv2.IMREAD_UNCHANGED)
    #print(temp.shape) 
    temp_ = cv2.medianBlur(temp,5)
    circles = cv2.HoughCircles(temp_,cv2.HOUGH_GRADIENT,1,100, param1=50,param2=30,minRadius=400,maxRadius=450)
    #In case of no circle found, use last known best setup
    if circles is not None:
      temp_circle=circles
     # print(circles)
      print(source)
    else:
      circles=temp_circle
      print('Alert!')
      print(source)
    
    if blur:
      temp=temp_
    else:
      temp=temp
    
    circles = np.uint16(np.around(circles))
    for n in circles[0,:]:
      # draw the outer circle, only one
      cv2.circle(temp,(n[0],n[1]),n[2],(0,255,0),2)

    height,width = temp.shape
    dim=(height,width)
    mask = np.zeros(dim, np.uint8)
    cv2.circle(mask,(n[0],n[1]),n[2],(255,255,255),thickness=-1)
    temp = cv2.bitwise_and(temp, mask, mask=mask)
    #############################################
    #cv2.imshow(temp)
    #Seperate lot normalization
    #Normalization setup
    if i.split('_')[0]=='17':
      if config_[0]==0 or config_[0]==1: #Remove || config_[x] to enable brightness based normalization
        mew[0]=np.mean(temp[mask==255])
        sigma[0]=np.std(temp[mask==255]) 
        #print(mew[0],' ', sigma[0])
        config_[0]=1
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([mew[0]], [sigma[0]])]) ##update
      else:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((mew[0]), (sigma[0]))]) ##update
    elif i.split('_')[0]=='19':
      if config_[1]==0 or config_[1]==1: #Remove || config_[x] to enable brightness based normalization
        mew[1]=np.mean(temp[mask==255])
        sigma[1]=np.std(temp[mask==255]) 
        #print(mew[1],' ', sigma[1])
        config_[1]=1
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([mew[1]], [sigma[1]])])
      else: 
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([mew[1]], [sigma[1]])])
    else:
      if config_[2]==0 or config_[2]==1: #Remove || config_[x] to enable brightness based normalization
        mew[2]=np.mean(temp[mask==255])
        sigma[2]=np.std(temp[mask==255]) 
        #print(mew[2],' ', sigma[2])
        config_[2]=1
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([mew[2]], [sigma[2]])])
      else:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([mew[2]], [sigma[2]])])
   

    #############################################
    x=n[0]-n[2]
    y=n[1]-n[2]
    mask=np.zeros(dim, np.uint8)
    crop=temp[y:y+n[2]*2,x:x+n[2]*2]
    p_=(len(mask)/2)-n[2]
    mask[int(p_):int(p_+len(crop)), int(p_):int(p_+len(crop))]=crop
    temp=mask
 
    #Resize
    scale_percent = 20 # percent of original size
    width = int(temp.shape[1] * scale_percent / 100)
    height = int(temp.shape[0] * scale_percent / 100)
    dim = (300,300) # One can use scaled height and width as tuple
    #w_=300
    #h_=300
    # resize image
    temp = cv2.resize(temp, dim, interpolation = cv2.INTER_AREA)
    rot90=cv2.rotate(temp, cv2.ROTATE_90_CLOCKWISE)
    rot180=cv2.rotate(temp, cv2.ROTATE_180)
    rot270=cv2.rotate(rot90, cv2.ROTATE_180)
    temp=temp.reshape(w_,h_,1)
    rot90=rot90.reshape(w_,h_,1)
    rot180=rot180.reshape(w_,h_,1)
    rot270=rot270.reshape(w_,h_,1)
    #print(temp.shape)
    #cv2.imshow(temp) 
    #cv2.imshow(rot90) 
    #cv2.imshow(rot180) 
    #temp=torchvision.transforms.ToTensor()(temp)
    temp=transform(temp) ## Add transform to data augmentation
    rot90=transform(rot90)
    rot180=transform(rot180)
    rot270=transform(rot270)
    ############################################################################
    if state0==True:
      image=temp
      image90=rot90
      image180=rot180
      image270=rot270
      state0=False
    else:
      image=torch.cat([image,temp])
      image90=torch.cat([image90,rot90])
      image180=torch.cat([image180,rot180])
      image270=torch.cat([image270,rot270])
  
  imageF=torch.flip(image,[0,1])
  image90F=torch.flip(image90,[0,1])
  image180F=torch.flip(image180,[0,1])
  image270F=torch.flip(image270,[0,1])

  image=image.reshape(-1,w_,h_,1)
  image90=image90.reshape(-1,w_,h_,1)
  image180=image180.reshape(-1,w_,h_,1)
  image270=image270.reshape(-1,w_,h_,1)
  imageF=imageF.reshape(-1,w_,h_,1)
  image90F=image90F.reshape(-1,w_,h_,1)
  image180F=image180F.reshape(-1,w_,h_,1)
  image270F=image270F.reshape(-1,w_,h_,1)
  
  image=image.unsqueeze(0)
  image90=image90.unsqueeze(0)
  image180=image180.unsqueeze(0)
  image270=image270.unsqueeze(0)
  imageF=imageF.unsqueeze(0)
  image90F=image90F.unsqueeze(0)
  image180F=image180F.unsqueeze(0)
  image270F=image270F.unsqueeze(0)
  
  for repeat in range(stack_): 
  # for data augmenting 0,90,180, 270  rotation & Flip
      target.append(target_param[i.split('_')[1]])
      lotmap.append(i.split('_')[1])
  if state1==True:
    data=image
    if data_aug:
      data=torch.cat([data,image90])
      data=torch.cat([data,image180])
      data=torch.cat([data,image270])
      data=torch.cat([data,image270F])
      data=torch.cat([data,image180F])
      data=torch.cat([data,image90F])
      data=torch.cat([data,imageF])
      print(data.shape)
    else:
      print(data.shape)
      
    state1=False
  
  else:
    data=torch.cat([data,image])
    if data_aug:
      data=torch.cat([data,image90])
      data=torch.cat([data,image180])
      data=torch.cat([data,image270])
      data=torch.cat([data,image270F])
      data=torch.cat([data,image180F])
      data=torch.cat([data,image90F])
      data=torch.cat([data,imageF])
      print(data.shape)
    else:
      print(data.shape)
  
  state0=True
  
target=torch.FloatTensor(target)
target=torch.reshape(target,(stack_*count,1)) #Again 8 for rotations and flips
print(target.shape)
# target built
################################################################################
out_file.write('Done packing datasets.\n')
out_file.write('Target review: \n')
out_file.write(str(target))
out_file.write('\nSaving a numpy array:\n')

np.save(out_file_path+'target3D.npy', target)
np.save(out_file_path+'CTdata3D.npy',data)
with open(out_file_path+'lotmap3D.pkl','wb') as f:
  pickle.dump(lotmap, f)
  
out_file.write('\nDone!\n')
out_file.close()
