#!/bin/bash

sbatch --partition=isi --gres=gpu:1 --time=24:00:00 --mem=64000 ./upredict_exp1_halfway.sh P1/EXP1/P12_M_nosig_C100_W300_MV0.2 N21_0.7375_0.7722_0.7059_1 nosig 100 300 
sbatch --partition=isi --gres=gpu:1 --time=24:00:00 --mem=64000 ./upredict_exp2_halfway.sh P1/EXP2/P12_M_nosig_C100_W300_MV0.2_EP49 N21_0.7795_0.8549_0.7163_1 nosig 100 300


squeue -u huan183