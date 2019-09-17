#!/bin/bash

sbatch --partition=isi --gres=gpu:1 --time=24:00:00 ./train_single_exp1.sh BC2GM-IOBES
sbatch --partition=isi --gres=gpu:1 --time=24:00:00 ./train_single_exp1.sh BC4CHEMD-IOBES
sbatch --partition=isi --gres=gpu:1 --time=24:00:00 ./train_single_exp1.sh BC5CDR-IOBES
sbatch --partition=isi --gres=gpu:1 --time=24:00:00 ./train_single_exp1.sh NCBI-IOBES
sbatch --partition=isi --gres=gpu:1 --time=24:00:00 ./train_single_exp1.sh JNLPBA-IOBES
sbatch --partition=isi --gres=gpu:1 --time=24:00:00 ./train_single_exp1.sh linnaeus-IOBES