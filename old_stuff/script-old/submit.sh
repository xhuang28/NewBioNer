#!/bin/bash

# Training
for mask in 0.0 0.2 0.4 0.6 0.8 1.0; do
  sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./train.sh $mask
done
