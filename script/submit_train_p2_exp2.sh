#!/bin/bash


# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2.sh P22 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 9 False
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2.sh P22 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 48 True

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2.sh P23 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 29 False
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2.sh P23 P1/EXP2 P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.9665_0.9618_0.9712_49 M nosig 100 300 47 True


sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2.sh P22 P1/EXP2 P12_M_nosig_C100_W400_MV0.5_EP57 N21_0.9715_0.9641_0.9791_57 M nosig 100 400 14 False
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp2.sh P23 P1/EXP2 P12_M_nosig_C100_W400_MV0.3_EP53 N21_0.9710_0.9654_0.9766_53 M nosig 100 400 16 False


squeue -u huan183