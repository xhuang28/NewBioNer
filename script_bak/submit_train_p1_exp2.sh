#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=64000 ./train_p1_exp2.sh P12 M nosig 100 300 0.2 49

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp2.sh P10 M nosig 100 300 0 51


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp2.sh P13 M nosig 100 300 0.85 52
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp2.sh P13 M nosig 100 300 0.9  49
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p1_exp2.sh P13 M nosig 100 300 0.95 70
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 300 0.25 0.25 56
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 300 0.2 0.2 56
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 300 0.1 0.1 50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 300 0.3 0.3 49
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 300 0.05 0.05 50
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 300 0.01 0.01 57
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 300 0.001 0.001 57
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 200 300 0 0 62
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 50 300 0 0 66
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 400 0 0 57
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 200 400 0 0 54
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p1_exp2.sh P11 M nosig 100 300 0 34

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 300 0 0 84
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 100 400 0 0 55
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 200 300 0 0 60
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p14_exp2.sh P14 M nosig 200 400 0 0 66

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 ./train_p1_exp2_MTL.sh 100 300

squeue -u huan183

