#!/bin/bash


sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./cexp2_train.sh 51


squeue -n cexp2_train.sh

