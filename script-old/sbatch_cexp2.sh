#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./cexp2.sh 1 3
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./cexp2.sh 2 3
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./cexp2.sh 3 2
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./cexp2.sh 4 2
sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./u_cexp2.sh 51


squeue -n cexp2.sh
squeue -n u_cexp2.sh

