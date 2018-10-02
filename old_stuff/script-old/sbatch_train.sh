#!/bin/bash

sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./train.sh 1
sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./train.sh 2
sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./train.sh 3
sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./train.sh 4

squeue -n train.sh

