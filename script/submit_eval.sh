#!/bin/bash


# substitute ?? with file name of best checkpoint (no suffix) in the folder
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M C100_W300_MV0_0_EP200 ?? pickle2
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M C100_W300_MV0_1_EP200 ?? pickle2
sbatch --partition=isi --gres=gpu:1 --time=1:00:00 ./eval.sh P1 M C100_W300_MV1_1_EP200 ?? pickle2


