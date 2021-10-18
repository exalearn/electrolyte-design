#! /bin/bash

for rf in sum softmax max mean; do
  python run_test.py --readout-fn $rf --random-seed 2
done
