#! /bin/bash

i=1

for s in `seq 8`;
do
  CUDA_VISIBLE_DEVICES=$i screen -S bs-${s} -d -m `which python` train-replica.py $i 
  i=$((i + 1))
done
