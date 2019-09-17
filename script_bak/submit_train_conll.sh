#!/bin/bash


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p1_conll_exp1.sh 100 300 0 1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p1_conll_exp1.sh 100 300 1 1
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p1_conll_exp1.sh 100 300 0 0

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh LOC
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh PER
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh ORG
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp1.sh MISC

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p1_conll_exp2.sh 100 300 0 1 63
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p1_conll_exp2.sh 100 300 1 1 4
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p1_conll_exp2.sh 100 300 0 0 76

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh LOC  58
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh PER  195
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh ORG  98
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_STM_exp2.sh MISC 101

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p1_conll_exp2_MTL.sh 100 300
