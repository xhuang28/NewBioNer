#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P10 M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P10 M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P10 M nosig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P10 M relu 200 300 0


# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh Li M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh Li M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh Li M nosig 200 300 0 
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh Li M relu 200 300 0

# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P11 M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P11 M sig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P11 M nosig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp1.sh P11 M relu 200 300 0


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 100 200 0
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 100 300 0
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 100 400 0
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 200 200 0
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 200 300 0
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 200 400 0
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 300 200 0
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 300 300 0
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 300 400 0

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 100 200 0

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 100 300 0.1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 100 300 0.2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 100 300 0.3
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 100 400 0.1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 100 400 0.2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 100 400 0.3
# sbatch --partition=isi --gres=gpu:1 --time=80:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 200 400 0.1
# sbatch --partition=isi --gres=gpu:1 --time=80:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 200 400 0.2
# sbatch --partition=isi --gres=gpu:1 --time=80:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 200 400 0.3

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 200 200 0.1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 300 200 0.2
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 300 200 0.3
# sbatch --partition=isi --gres=gpu:1 --time=80:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 300 300 0.1
# sbatch --partition=isi --gres=gpu:1 --time=80:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 300 300 0.2
# sbatch --partition=isi --gres=gpu:1 --time=80:00:00 --mem=64000 ./train_p1_exp1.sh P12 M nosig 300 300 0.3


squeue -u huan183

