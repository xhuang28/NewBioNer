#!/bin/bash

sbatch --partition=isi --gres=gpu:1 --time=24:00:00 ./train_single_exp2.sh BC2GM-IOBES    5
sbatch --partition=isi --gres=gpu:1 --time=24:00:00 ./train_single_exp2.sh BC4CHEMD-IOBES 4
sbatch --partition=isi --gres=gpu:1 --time=24:00:00 ./train_single_exp2.sh BC5CDR-IOBES   8
sbatch --partition=isi --gres=gpu:1 --time=24:00:00 ./train_single_exp2.sh NCBI-IOBES     7
sbatch --partition=isi --gres=gpu:1 --time=24:00:00 ./train_single_exp2.sh JNLPBA-IOBES   4
sbatch --partition=isi --gres=gpu:1 --time=24:00:00 ./train_single_exp2.sh linnaeus-IOBES 1