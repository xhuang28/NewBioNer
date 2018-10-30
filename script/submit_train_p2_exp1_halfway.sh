#!/bin/bash

sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_halfway.sh P22 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.7375_0.7722_0.7059_1 M nosig 100 300 False
sbatch --partition=isi --gres=gpu:1 --time=120:00:00 --mem=64000 ./train_p2_exp1_halfway.sh P23 P1/EXP1 P12_M_nosig_C100_W300_MV0.2 N21_0.7375_0.7722_0.7059_1 M nosig 100 300 False

squeue -u huan183