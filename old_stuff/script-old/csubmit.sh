#!/bin/bash

# Training
sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./ctrain.sh 0.0 47
sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./ctrain.sh 0.2 54
sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./ctrain.sh 0.4 81
sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./ctrain.sh 0.6 45
sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./ctrain.sh 0.8 65
sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./ctrain.sh 1.0 40
