# -*- coding: utf-8 -*-
"""
Created on Thu Jul  11 15:35:41 2020

@Last modified by: Golam
"""

import numpy as np
import os
import argparse
import pickle

# Change to correct directory
with open('/usr/WS2/jaman1/summer2020/lotmap3D.pkl','rb') as f:
 lotmap=pickle.load(f)
# feed e.g. python stress3D.py --leaveoutlots 'AM','D'

list_=list(set(lotmap))
procspernode = 2 #Based on available GPU
numnodes = int(np.ceil(len(list_)/procspernode)) #11

lotsplits = np.array_split(list_, numnodes)

for group, lots in enumerate(lotsplits):
    shfilename = 'group' + str(group) + '.sh'
    fhand = open(shfilename, 'w')
    fhand.write('#!/bin/bash\n')
    fhand.write('python run3D.py' + ' --leaveoutlots ' +  ','.join('\'{0}\''.format(w) for w in lots) + '\n')
    fhand.close()
