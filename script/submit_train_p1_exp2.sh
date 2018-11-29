#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp2.sh P12 M nosig 100 300 0.2 49

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp2.sh P10 M nosig 100 300 0 51


sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp2.sh P13 M nosig 100 300 0.85 52
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp2.sh P13 M nosig 100 300 0.9  49
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp2.sh P13 M nosig 100 300 0.95 70



squeue -u huan183

