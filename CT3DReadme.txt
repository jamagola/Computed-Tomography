#This document works as a guide towards running CT data on 3D and WideRes Network/framework using following files:

-preprocess3D.py : 

This file loads all the CT data lot by lot and prepare each lot as 3D data point along with data 
augmentation (if enabled), downsampling(skipping layers and reducing each image resolution), normalization. 

This file address Neat_CT_Data where all the CT data lots and only the CT images stored in lot based folders.
Please change the directory to source folder. Images currently reading is .tif. Step variable determines how many images
to skip (e.g. step=n skips n-1 images). Each images currently downsampled to 300x300.

-preprocess2D.py : 

Just as preprocess3D.py, preprocess2D.py prepares .tif images but for wideResNet instead of 3D network and target/labels
attached to each image and not for entire layers of images.

Note:
Once the preprocess3D/2D.py finished running, three files will be generated. 
- lotmap3D/2D.pkl : All the lot values used in preprocess.
- target3D/2D.npy : Numpy array containg all the target values in the order of lotmap*.pkl
- CTdata3D/2D.npy : All the preprocessed input used in the training and testing.

-model2D/3D.py : Network architecture for wideResNet(from Matt's framework) if model2D.py. 3D CNN architecture if 
model3D.py applied.

run2D/3D.py : These files train and test using model2D/3D.py after the completion of preprocess2D/3D.py. These files
takes --leaveoutlots and send the each lot job to available GPU. At the end of process, models are stored as .pth and .out
file generated for each --leaveoutlots with epoch results and evaluation outcomes.

gensh.py : Only for 3D process. Runs after preprocess and generates .sh for group of lots to be sent out to multiple nodes.
gensh2D.py : Only for 2D process and does same job as gensh.py

runall_.sh : Runs gensh.py and send sbatch -n 1 -N 1 -A $bank --time $time group*.sh
runall_2D.sh : Runs gensh2D.py and send sbatch -n 1 -N 1 -A $bank --time $time group*.sh

Contacts:
For any information or concerns raised from this document:

Golam Gause Jaman
jamagola@isu.edu
(208)760-7357
Lives in Mountain Time

Date created: 08/05/2020
