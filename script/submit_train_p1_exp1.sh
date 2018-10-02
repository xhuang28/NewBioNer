#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P10 M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P10 M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P10 M nosig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P10 M relu 200 300 0


# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh Li M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh Li M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh Li M nosig 200 300 0 
sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh Li M relu 200 300 0

# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P11 M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P11 M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P11 M nosig 200 300 0
sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P11 M relu 200 300 0

# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P12 M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P12 M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 200 300 0
sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P12 M relu 200 300 0

squeue -u huan183

