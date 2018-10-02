#!/bin/bash

sbatch --partition=isi --gres=gpu:1 --time=336:00:00 --mem=32000 ./eval.sh P1 M Li_M_nosig_C200_W300_MV0_EP66 N21_0.9738_0.9673_0.9804_66


squeue -u huan183

