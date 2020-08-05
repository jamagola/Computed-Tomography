#!/bin/bash 
##usage: ./runall_.sh
runtime=1:00:00
bank=feedopt
##generate group*.sh scripts; enables usage of all gpus on a node
python gensh.py
for f in group*.sh
do
  sbatch -n 1 -N 1 -A $bank --time $runtime $f
done
rm group*.sh #safe to delete these files immediately after job is submitted
