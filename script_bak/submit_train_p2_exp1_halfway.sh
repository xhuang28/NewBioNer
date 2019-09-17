#!/bin/bash

# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_halfway.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.7375_0.7722_0.7059_1 M nosig 100 300 False
# sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_halfway.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.7375_0.7722_0.7059_1 M nosig 100 300 False


sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_halfway.sh P22 P2/EXP1 P22_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_halfway_IT1 N21_0.8245_0.8316_0.8175_110 M nosig 100 300 False 2
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_halfway.sh P23 P2/EXP1 P23_P12_M_nosig_C100_W300_MV0.2_M_nosig_C100_W300_RestartFalse_halfway_IT1 N21_0.8318_0.8395_0.8242_59 M nosig 100 300 False 2

squeue -u huan183