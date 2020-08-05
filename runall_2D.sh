#!/bin/bash 
##usage: ./runall_2D.sh  Alsp check dos2unix
runtime=12:00:00
bank=feedopt
##generate group*.sh scripts; enables usage of all gpus on a node
python gensh2D.py
for f in group2D_*.sh
do
  sbatch -n 1 -N 1 -A $bank --time $runtime $f
done
rm group2D_*.sh #safe to delete these files immediately after job is submitted
