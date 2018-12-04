#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp2.sh P12 M nosig 100 300 0.2 49

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp2.sh P10 M nosig 100 300 0 51


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp2.sh P13 M nosig 100 300 0.85 52
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp2.sh P13 M nosig 100 300 0.9  49
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp2.sh P13 M nosig 100 300 0.95 70
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 300 0.25 0.25 56
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 300 0.2 0.2 56
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 300 0.1 0.1 50
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 300 0.3 0.3 49



squeue -u huan183

